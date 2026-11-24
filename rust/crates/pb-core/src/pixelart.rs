//! Pixel-art engine: turns a photo or an illustration into pixel art that
//! looks hand-made rather than "shrunk and posterised".
//!
//! Everything after the first step runs at art resolution (a few hundred
//! pixels across), in OKLab, so it is fast and the live preview on a small
//! copy matches the full-size result (the grid is given in art pixels
//! across, not source pixels).
//!
//! 1. **Cell sampling**: each art pixel takes the *dominant* colour of its
//!    cell (2-means inside the cell, majority cluster), not the average, so
//!    edges stay crisp and no muddy in-between colours appear. Thin dark
//!    lines that cover enough of a cell win (line art survives), and the
//!    brightest point of every cell is kept for sparkles.
//! 2. **Abstraction**: edge-preserving smoothing flattens noise and texture
//!    into clean areas of colour.
//! 3. **Look**: punch (chroma / contrast) and hue shifting: shadows lean
//!    to violet-blue, highlights to warm yellow, the way pixel artists build
//!    colour ramps.
//! 4. **Palette**: adaptive (detail-weighted k-means++ in OKLab), *ramps*
//!    (a few hue families, each expanded into a hue-shifted light-to-dark
//!    ramp), a built-in palette, or the user's own colours.
//! 5. **Quantise + dither**: palette-aware pattern dither (mixes the two
//!    nearest palette colours with a Bayer pattern, only in smooth
//!    gradients, never across edges), classic ordered, error diffusion, or
//!    none.
//! 6. **Cleanup**: removes low-contrast orphan pixels and "pixel perfect"
//!    L-corners from 1-px lines; keeps high-contrast specks (eyes, stars).
//! 7. **Outlines**: ink (darkest colour) or selective (a darker, cooler
//!    shade of the colour it borders); 1 px outside the sprite for cutouts.
//!    **Rim light** brightens edges that face the light.
//! 8. **Sparkles**: bright points become 4-point pixel stars with glints.
//! 9. Scale back with hard pixel edges (same size as the input), or keep
//!    the native art size.

use crate::image::Image;
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};

// ------------------------------------------------------------------ params

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct PixelArt {
    /// Art pixels across (resolution independent). 0 = use `cell`.
    pub width: usize,
    /// Legacy: source pixels per art pixel (used when `width` is 0).
    pub cell: usize,
    /// Palette size for "auto" / "ramps".
    pub colors: usize,
    /// "auto", "ramps", or a built-in palette id (see [`PALETTES`]).
    pub palette: String,
    /// Own palette (sRGB 0..1); overrides `palette` when not empty.
    pub custom: Vec<[f32; 3]>,
    /// "none", "pattern", "bayer", "diffuse" (`true` = bayer, `false` = none).
    #[serde(deserialize_with = "dither_mode")]
    pub dither: String,
    pub dither_amount: f32,
    /// "none", "ink", "selective" (`true` = ink, `false` = none).
    #[serde(deserialize_with = "outline_mode")]
    pub outline: String,
    /// Line detail 0..1: keeps thinner dark lines and outlines weaker edges.
    pub lines: f32,
    /// Rim light 0..1 on edges facing the light.
    pub rim: f32,
    /// Where the light comes from, degrees (0 = right, 90 = top).
    pub light: f32,
    /// Hue shifting of shadows / highlights 0..1.
    pub hue_shift: f32,
    /// Chroma / contrast boost -1..1.
    pub punch: f32,
    /// Edge-preserving abstraction 0..1.
    pub smooth: f32,
    /// Sparkle stars on bright points 0..1.
    pub sparkle: f32,
    /// Remove orphan pixels and jaggies.
    pub cleanup: bool,
    /// Output the art at its native size (1 px per art pixel).
    pub native: bool,
}

impl Default for PixelArt {
    fn default() -> Self {
        PixelArt {
            width: 0,
            cell: 0,
            colors: 16,
            palette: "auto".into(),
            custom: Vec::new(),
            dither: "pattern".into(),
            dither_amount: 0.5,
            outline: "none".into(),
            lines: 0.5,
            rim: 0.0,
            light: 45.0,
            hue_shift: 0.3,
            punch: 0.15,
            smooth: 0.3,
            sparkle: 0.0,
            cleanup: true,
            native: false,
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum BoolOrStr {
    B(bool),
    S(String),
}

fn dither_mode<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    Ok(match BoolOrStr::deserialize(d)? {
        BoolOrStr::B(b) => if b { "bayer" } else { "none" }.into(),
        BoolOrStr::S(s) => s,
    })
}

fn outline_mode<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    Ok(match BoolOrStr::deserialize(d)? {
        BoolOrStr::B(b) => if b { "ink" } else { "none" }.into(),
        BoolOrStr::S(s) => s,
    })
}

pub const DITHERS: &[(&str, &str)] = &[("none", "Off"), ("pattern", "Pattern"), ("bayer", "Ordered"), ("diffuse", "Diffuse")];
pub const OUTLINES: &[(&str, &str)] = &[("none", "Off"), ("ink", "Ink"), ("selective", "Shaded")];

impl PixelArt {
    pub fn validate(&self) -> Result<(), String> {
        if !DITHERS.iter().any(|d| d.0 == self.dither) {
            return Err(format!("unknown dither {}", self.dither));
        }
        if !OUTLINES.iter().any(|d| d.0 == self.outline) {
            return Err(format!("unknown outline {}", self.outline));
        }
        if self.custom.is_empty() && self.palette != "auto" && self.palette != "ramps" && palette_by_id(&self.palette).is_none() {
            return Err(format!("unknown palette {}", self.palette));
        }
        Ok(())
    }
}

// ------------------------------------------------------------------ styles

pub struct Style {
    pub id: &'static str,
    pub name: &'static str,
    pub blurb: &'static str,
    pub make: fn() -> PixelArt,
}

pub const STYLES: &[Style] = &[
    Style {
        id: "starlight",
        name: "Starlight",
        blurb: "Hue-shifted ramps, soft dither, shaded outlines, rim light and sparkles",
        make: || PixelArt {
            width: 184,
            colors: 32,
            palette: "ramps".into(),
            dither: "pattern".into(),
            dither_amount: 0.55,
            outline: "selective".into(),
            lines: 0.55,
            rim: 0.6,
            light: 50.0,
            hue_shift: 0.55,
            punch: 0.3,
            smooth: 0.45,
            sparkle: 0.45,
            ..PixelArt::default()
        },
    },
    Style {
        id: "classic",
        name: "Classic",
        blurb: "Clean 16-colour pixel art",
        make: || PixelArt { width: 128, colors: 16, dither: "none".into(), ..PixelArt::default() },
    },
    Style {
        id: "hd",
        name: "HD",
        blurb: "Detailed 48-colour ramps, gentle dither",
        make: || PixelArt {
            width: 256,
            colors: 48,
            palette: "ramps".into(),
            dither_amount: 0.4,
            outline: "selective".into(),
            lines: 0.4,
            rim: 0.3,
            hue_shift: 0.45,
            punch: 0.2,
            smooth: 0.3,
            ..PixelArt::default()
        },
    },
    Style {
        id: "sprite",
        name: "Sprite",
        blurb: "Small, bold, inked game sprite",
        make: || PixelArt {
            width: 64,
            colors: 12,
            palette: "ramps".into(),
            dither: "none".into(),
            outline: "ink".into(),
            lines: 0.7,
            hue_shift: 0.5,
            punch: 0.4,
            smooth: 0.7,
            ..PixelArt::default()
        },
    },
    Style {
        id: "eightbit",
        name: "8-bit",
        blurb: "Chunky pixels, few colours, hard edges",
        make: || PixelArt {
            width: 96,
            colors: 10,
            dither: "bayer".into(),
            dither_amount: 0.35,
            outline: "ink".into(),
            lines: 0.5,
            hue_shift: 0.1,
            punch: 0.45,
            smooth: 0.5,
            ..PixelArt::default()
        },
    },
    Style {
        id: "nocturne",
        name: "Nocturne",
        blurb: "Night-sky palette with stars",
        make: || PixelArt {
            width: 160,
            palette: "nocturne".into(),
            dither_amount: 0.6,
            outline: "selective".into(),
            rim: 0.5,
            hue_shift: 0.5,
            punch: 0.2,
            smooth: 0.4,
            sparkle: 0.8,
            ..PixelArt::default()
        },
    },
    Style {
        id: "handheld",
        name: "Handheld",
        blurb: "Four shades of green",
        make: || PixelArt {
            width: 120,
            palette: "handheld".into(),
            dither_amount: 0.7,
            hue_shift: 0.0,
            punch: 0.3,
            smooth: 0.4,
            ..PixelArt::default()
        },
    },
    Style {
        id: "sunset",
        name: "Sunset",
        blurb: "Warm dusk palette",
        make: || PixelArt {
            width: 160,
            palette: "sunset".into(),
            dither_amount: 0.6,
            outline: "selective".into(),
            rim: 0.4,
            light: 20.0,
            hue_shift: 0.4,
            punch: 0.2,
            smooth: 0.4,
            sparkle: 0.3,
            ..PixelArt::default()
        },
    },
    Style {
        id: "ink",
        name: "Mono ink",
        blurb: "Greyscale with ink outlines",
        make: || PixelArt {
            width: 140,
            palette: "mono".into(),
            dither_amount: 0.6,
            outline: "ink".into(),
            lines: 0.6,
            hue_shift: 0.0,
            punch: 0.2,
            smooth: 0.4,
            ..PixelArt::default()
        },
    },
];

pub fn style(id: &str) -> Option<PixelArt> {
    STYLES.iter().find(|s| s.id == id).map(|s| (s.make)())
}

// ------------------------------------------------------------------ palettes

/// Built-in palettes, generated from ramp specs: (hue°, chroma, shades,
/// darkest L, lightest L). Shades are hue-shifted like hand-made ramps.
pub struct Palette {
    pub id: &'static str,
    pub name: &'static str,
    ramps: &'static [(f32, f32, usize, f32, f32)],
    /// Extra hue shift for the ramps (0..1).
    shift: f32,
}

pub const PALETTES: &[Palette] = &[
    Palette {
        id: "nocturne",
        name: "Nocturne",
        ramps: &[
            (275.0, 0.13, 6, 0.16, 0.62), // indigo / violet sky
            (240.0, 0.10, 4, 0.30, 0.78), // blue
            (330.0, 0.15, 5, 0.40, 0.85), // magenta / pink
            (20.0, 0.10, 3, 0.62, 0.90),  // peach clouds
            (95.0, 0.06, 2, 0.92, 0.99),  // star cream
            (190.0, 0.08, 2, 0.55, 0.80), // teal haze
        ],
        shift: 0.6,
    },
    Palette {
        id: "sunset",
        name: "Sunset",
        ramps: &[
            (300.0, 0.10, 5, 0.18, 0.55),
            (10.0, 0.16, 5, 0.40, 0.75),
            (55.0, 0.15, 5, 0.60, 0.92),
            (95.0, 0.10, 3, 0.85, 0.98),
            (250.0, 0.08, 3, 0.35, 0.70),
        ],
        shift: 0.5,
    },
    Palette {
        id: "meadow",
        name: "Meadow",
        ramps: &[
            (140.0, 0.13, 6, 0.25, 0.85),
            (60.0, 0.09, 5, 0.30, 0.85),
            (230.0, 0.09, 4, 0.45, 0.90),
            (25.0, 0.10, 3, 0.45, 0.80),
            (100.0, 0.04, 2, 0.15, 0.97),
        ],
        shift: 0.5,
    },
    Palette {
        id: "ember",
        name: "Ember",
        ramps: &[(25.0, 0.17, 6, 0.20, 0.70), (60.0, 0.16, 5, 0.55, 0.95), (330.0, 0.08, 3, 0.10, 0.40), (90.0, 0.05, 2, 0.95, 0.99)],
        shift: 0.5,
    },
    Palette {
        id: "frost",
        name: "Frost",
        ramps: &[(235.0, 0.10, 6, 0.20, 0.85), (200.0, 0.07, 4, 0.50, 0.95), (285.0, 0.07, 3, 0.25, 0.60), (95.0, 0.02, 2, 0.97, 1.0)],
        shift: 0.4,
    },
    Palette { id: "handheld", name: "Handheld", ramps: &[(130.0, 0.09, 4, 0.25, 0.85)], shift: 0.0 },
    Palette { id: "mono", name: "Mono", ramps: &[(0.0, 0.0, 8, 0.08, 0.98)], shift: 0.0 },
];

pub fn palette_by_id(id: &str) -> Option<&'static Palette> {
    PALETTES.iter().find(|p| p.id == id)
}

/// The palette's colours as sRGB 0..1.
pub fn palette_colors(id: &str) -> Vec<[f32; 3]> {
    palette_by_id(id).map(|p| builtin_lab(p).into_iter().map(oklab_to_srgb).collect()).unwrap_or_default()
}

fn builtin_lab(p: &Palette) -> Vec<Lab> {
    let mut out = Vec::new();
    for &(h, c, n, l0, l1) in p.ramps {
        out.extend(ramp(h, c, n, l0, l1, p.shift));
    }
    out
}

/// A light-to-dark ramp around one hue: shadows lean to violet-blue,
/// highlights to warm yellow, chroma peaks in the mid tones.
fn ramp(hue: f32, chroma: f32, n: usize, l0: f32, l1: f32, shift: f32) -> Vec<Lab> {
    (0..n)
        .map(|i| {
            let t = if n > 1 { i as f32 / (n - 1) as f32 } else { 0.5 };
            let l = l0 + (l1 - l0) * t;
            let s = t * 2.0 - 1.0; // -1 dark .. 1 light
            let h = shift_hue(hue, s, shift * 40.0);
            let c = chroma * (1.0 - 0.45 * s.abs().powf(1.6)) * if s < 0.0 { 1.0 + 0.15 * shift * -s } else { 1.0 };
            lch_to_lab(l, c, h)
        })
        .collect()
}

const SHADOW_HUE: f32 = 285.0;
const LIGHT_HUE: f32 = 95.0;

/// Rotates `hue` toward the shadow hue (s < 0) or highlight hue (s > 0) by
/// up to `max_deg * |s|`, never past the target.
fn shift_hue(hue: f32, s: f32, max_deg: f32) -> f32 {
    let target = if s < 0.0 { SHADOW_HUE } else { LIGHT_HUE };
    let mut d = target - hue;
    while d > 180.0 {
        d -= 360.0;
    }
    while d < -180.0 {
        d += 360.0;
    }
    let step = (max_deg * s.abs()).min(d.abs());
    (hue + step * d.signum()).rem_euclid(360.0)
}

// ------------------------------------------------------------------ colour

type Lab = [f32; 3];

fn srgb_to_lin(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn lin_to_srgb(v: f32) -> f32 {
    let v = v.max(0.0);
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

struct LinLut([f32; 1025]);

impl LinLut {
    fn new() -> Self {
        let mut t = [0.0; 1025];
        for (i, v) in t.iter_mut().enumerate() {
            *v = srgb_to_lin(i as f32 / 1024.0);
        }
        LinLut(t)
    }
    #[inline]
    fn get(&self, v: f32) -> f32 {
        let x = v.clamp(0.0, 1.0) * 1024.0;
        let i = (x as usize).min(1023);
        let f = x - i as f32;
        self.0[i] + (self.0[i + 1] - self.0[i]) * f
    }
}

#[inline]
fn lin_to_oklab(r: f32, g: f32, b: f32) -> Lab {
    let l = (0.412_221_46 * r + 0.536_332_55 * g + 0.051_445_995 * b).cbrt();
    let m = (0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b).cbrt();
    let s = (0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b).cbrt();
    [
        0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
        1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
    ]
}

pub fn srgb_to_oklab(c: [f32; 3]) -> Lab {
    lin_to_oklab(srgb_to_lin(c[0]), srgb_to_lin(c[1]), srgb_to_lin(c[2]))
}

pub fn oklab_to_srgb(c: Lab) -> [f32; 3] {
    let l = c[0] + 0.396_337_78 * c[1] + 0.215_803_76 * c[2];
    let m = c[0] - 0.105_561_346 * c[1] - 0.063_854_17 * c[2];
    let s = c[0] - 0.089_484_18 * c[1] - 1.291_485_5 * c[2];
    let (l, m, s) = (l * l * l, m * m * m, s * s * s);
    let r = 4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s;
    let g = -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s;
    let b = -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s;
    [lin_to_srgb(r).clamp(0.0, 1.0), lin_to_srgb(g).clamp(0.0, 1.0), lin_to_srgb(b).clamp(0.0, 1.0)]
}

/// Keeps a colour inside sRGB by lowering its chroma (not clipping, which
/// shifts hue).
fn gamut_fit(c: Lab) -> Lab {
    let inside = |c: Lab| {
        let l = c[0] + 0.396_337_78 * c[1] + 0.215_803_76 * c[2];
        let m = c[0] - 0.105_561_346 * c[1] - 0.063_854_17 * c[2];
        let s = c[0] - 0.089_484_18 * c[1] - 1.291_485_5 * c[2];
        let (l, m, s) = (l * l * l, m * m * m, s * s * s);
        let r = 4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s;
        let g = -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s;
        let b = -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s;
        [r, g, b].iter().all(|v| (-0.001..=1.001).contains(v))
    };
    if inside(c) {
        return c;
    }
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    for _ in 0..12 {
        let mid = (lo + hi) * 0.5;
        if inside([c[0], c[1] * mid, c[2] * mid]) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    [c[0], c[1] * lo, c[2] * lo]
}

fn lch_to_lab(l: f32, c: f32, h_deg: f32) -> Lab {
    let h = h_deg.to_radians();
    gamut_fit([l.clamp(0.0, 1.0), c * h.cos(), c * h.sin()])
}

fn chroma(c: Lab) -> f32 {
    (c[1] * c[1] + c[2] * c[2]).sqrt()
}

fn hue_deg(c: Lab) -> f32 {
    c[2].atan2(c[1]).to_degrees().rem_euclid(360.0)
}

#[inline]
fn d2(a: Lab, b: Lab) -> f32 {
    let (x, y, z) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    x * x + y * y + z * z
}

fn nearest(pal: &[Lab], p: Lab) -> usize {
    let mut best = (0, f32::MAX);
    for (i, c) in pal.iter().enumerate() {
        let d = d2(*c, p);
        if d < best.1 {
            best = (i, d);
        }
    }
    best.0
}

/// Deterministic 0..1 hash (palette init must not flicker between previews).
fn hash01(i: u64) -> f32 {
    let mut x = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5851_F42D_4C95_7F2D;
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 29;
    (x >> 40) as f32 / (1u64 << 24) as f32
}

// ------------------------------------------------------------------ art grid

/// The art image: OKLab colour, alpha, and the brightest point of each cell
/// (L above the cell's dominant colour) for sparkles.
struct Art {
    w: usize,
    h: usize,
    lab: Vec<Lab>,
    alpha: Vec<bool>,
    peak: Vec<f32>,
    peak_lab: Vec<Lab>,
}

fn grid(w: usize, h: usize, p: &PixelArt) -> (usize, usize, usize) {
    let cell = if p.width > 0 {
        ((w as f32 / p.width.clamp(8, 1024) as f32).round() as usize).max(1)
    } else {
        p.cell.max(1)
    };
    (cell, w.div_ceil(cell).max(1), h.div_ceil(cell).max(1))
}

/// Dominant colour of every cell.
fn sample_cells(img: &Image, cell: usize, aw: usize, ah: usize, lines: f32) -> Art {
    let lut = LinLut::new();
    // At most ~12x12 samples per cell: plenty for the dominant colour.
    let step = cell.div_ceil(12).max(1);
    let ink_share = 0.5 - 0.3 * lines.clamp(0.0, 1.0);
    let rows: Vec<Vec<(Lab, bool, f32, Lab)>> = (0..ah)
        .into_par_iter()
        .map(|ay| {
            let mut row = Vec::with_capacity(aw);
            let mut samples: Vec<Lab> = Vec::with_capacity(144);
            let (y0, y1) = (ay * cell, ((ay + 1) * cell).min(img.h));
            for ax in 0..aw {
                let (x0, x1) = (ax * cell, ((ax + 1) * cell).min(img.w));
                samples.clear();
                let (mut na, mut n) = (0usize, 0usize);
                let mut y = y0 + (step / 2).min(y1 - y0 - 1);
                while y < y1 {
                    let mut x = x0 + (step / 2).min(x1 - x0 - 1);
                    while x < x1 {
                        let q = img.px[y * img.w + x];
                        n += 1;
                        if q[3] >= 0.5 {
                            na += 1;
                            samples.push(lin_to_oklab(lut.get(q[0]), lut.get(q[1]), lut.get(q[2])));
                        }
                        x += step;
                    }
                    y += step;
                }
                if samples.is_empty() {
                    row.push(([0.0; 3], false, 0.0, [0.0; 3]));
                    continue;
                }
                let alpha = na * 2 >= n;
                let col = dominant(&samples, ink_share);
                let pk = samples.iter().copied().fold([f32::MIN, 0.0, 0.0], |a, b| if b[0] > a[0] { b } else { a });
                row.push((col, alpha, (pk[0] - col[0]).max(0.0), pk));
            }
            row
        })
        .collect();
    let mut art = Art { w: aw, h: ah, lab: Vec::with_capacity(aw * ah), alpha: Vec::new(), peak: Vec::new(), peak_lab: Vec::new() };
    for r in rows {
        for (c, a, pk, pl) in r {
            art.lab.push(c);
            art.alpha.push(a);
            art.peak.push(pk);
            art.peak_lab.push(pl);
        }
    }
    art
}

/// 2-means inside a cell: the majority cluster's mean, unless the minority
/// is clearly darker and covers `ink_share` of the cell (a line).
fn dominant(s: &[Lab], ink_share: f32) -> Lab {
    let n = s.len() as f32;
    let mean = mean_of(s.iter().copied());
    if s.len() < 3 {
        return mean;
    }
    let far = |from: Lab| *s.iter().max_by(|a, b| d2(**a, from).total_cmp(&d2(**b, from))).unwrap();
    let mut a = far(mean);
    let mut b = far(a);
    if d2(a, b) < 0.06 * 0.06 {
        return mean;
    }
    let mut na = 0.0;
    for _ in 0..4 {
        let (mut sa, mut sb) = ([0.0f32; 3], [0.0f32; 3]);
        let (mut ca, mut cb) = (0.0f32, 0.0f32);
        for &p in s {
            if d2(p, a) <= d2(p, b) {
                (0..3).for_each(|c| sa[c] += p[c]);
                ca += 1.0;
            } else {
                (0..3).for_each(|c| sb[c] += p[c]);
                cb += 1.0;
            }
        }
        if ca > 0.0 {
            a = sa.map(|v| v / ca);
        }
        if cb > 0.0 {
            b = sb.map(|v| v / cb);
        }
        na = ca;
    }
    if d2(a, b) < 0.05 * 0.05 {
        return mean;
    }
    let (big, small, share) = if na >= n - na { (a, b, (n - na) / n) } else { (b, a, na / n) };
    if small[0] < big[0] - 0.12 && share >= ink_share {
        small
    } else {
        big
    }
}

fn mean_of(it: impl Iterator<Item = Lab>) -> Lab {
    let (mut s, mut n) = ([0.0f32; 3], 0.0f32);
    for p in it {
        (0..3).for_each(|c| s[c] += p[c]);
        n += 1.0;
    }
    s.map(|v| v / n.max(1.0))
}

const N4: [(isize, isize); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const N8: [(isize, isize); 8] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)];

fn nb(w: usize, h: usize, x: usize, y: usize, d: (isize, isize)) -> Option<usize> {
    let (xx, yy) = (x as isize + d.0, y as isize + d.1);
    (xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h).then(|| yy as usize * w + xx as usize)
}

/// Edge-preserving abstraction: a few passes of a 3x3 bilateral filter in
/// OKLab (texture and noise flatten, edges stay).
fn abstract_pass(art: &mut Art, amount: f32) {
    if amount <= 0.01 {
        return;
    }
    let passes = 1 + (amount * 3.0) as usize;
    let sc = 0.03 + 0.07 * amount;
    let inv = 1.0 / (2.0 * sc * sc);
    let (w, h) = (art.w, art.h);
    for _ in 0..passes {
        let src = art.lab.clone();
        let alpha = &art.alpha;
        art.lab.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                let i = y * w + x;
                if !alpha[i] {
                    continue;
                }
                let c = src[i];
                let (mut s, mut ws) = ([0.0f32; 3], 0.0f32);
                for dy in -1..=1isize {
                    for dx in -1..=1isize {
                        let Some(j) = nb(w, h, x, y, (dx, dy)) else { continue };
                        if !alpha[j] {
                            continue;
                        }
                        let q = src[j];
                        let wt = (-d2(c, q) * inv).exp() * if dx == 0 && dy == 0 { 1.0 } else { 0.6 };
                        (0..3).for_each(|k| s[k] += q[k] * wt);
                        ws += wt;
                    }
                }
                *o = s.map(|v| v / ws);
            }
        });
    }
}

/// Punch and hue shifting, relative to the image's own tonal range.
fn look(art: &mut Art, punch: f32, hue_shift: f32) {
    let ls: Vec<f32> = art.lab.iter().zip(&art.alpha).filter(|(_, a)| **a).map(|(c, _)| c[0]).collect();
    if ls.is_empty() {
        return;
    }
    let mut sorted = ls.clone();
    sorted.sort_by(f32::total_cmp);
    let pct = |q: f32| sorted[((sorted.len() - 1) as f32 * q) as usize];
    let (lo, mid, hi) = (pct(0.05), pct(0.5), pct(0.95));
    let half = ((hi - lo) * 0.5).max(0.05);
    let punch = punch.clamp(-1.0, 1.0);
    let hs = hue_shift.clamp(0.0, 1.0);
    art.lab.par_iter_mut().for_each(|c| {
        let mut l = c[0];
        let mut ch = chroma(*c);
        let mut h = hue_deg(*c);
        l = (mid + (l - mid) * (1.0 + 0.35 * punch)).clamp(0.0, 1.0);
        ch *= 1.0 + 0.6 * punch;
        let s = ((c[0] - mid) / half).clamp(-1.0, 1.0);
        if hs > 0.0 {
            // Coloured pixels rotate toward the shadow / light hue; greys
            // (no reliable hue) get a faint tint of it instead.
            let wgt = ((ch - 0.015) / 0.045).clamp(0.0, 1.0);
            h = shift_hue(h, s, 20.0 * hs * wgt);
            // Ramps: chroma peaks in the mid tones; deep shadows stay rich.
            ch *= 1.0 - 0.2 * hs * s.max(0.0).powi(2);
            let mut out = lch_to_lab(l, ch.max(0.0), h);
            let th = (if s < 0.0 { SHADOW_HUE } else { LIGHT_HUE }).to_radians();
            let tint = 0.018 * hs * s.abs() * (1.0 - wgt);
            out[1] += th.cos() * tint;
            out[2] += th.sin() * tint;
            *c = gamut_fit(out);
            return;
        }
        *c = lch_to_lab(l, ch.max(0.0), h);
    });
}

// ------------------------------------------------------------------ palette

fn build_palette(art: &Art, p: &PixelArt) -> Vec<Lab> {
    if !p.custom.is_empty() {
        return p.custom.iter().map(|c| srgb_to_oklab(*c)).collect();
    }
    if let Some(b) = palette_by_id(&p.palette) {
        return builtin_lab(b);
    }
    let colors = p.colors.clamp(2, 256);
    let (pts, wts) = weighted_points(art);
    if pts.is_empty() {
        return vec![[0.0; 3], [1.0, 0.0, 0.0]];
    }
    if p.palette == "ramps" && colors >= 6 {
        ramps_palette(&pts, &wts, colors, p.hue_shift)
    } else {
        kmeans(&pts, &wts, colors, [1.0, 1.0, 1.0], 14)
    }
}

/// Opaque art pixels, weighted by local contrast so small details (eyes,
/// highlights) get colours of their own instead of being averaged away.
fn weighted_points(art: &Art) -> (Vec<Lab>, Vec<f32>) {
    let (w, h) = (art.w, art.h);
    let stride = ((w * h) / 40_000).max(1);
    let mut pts = Vec::new();
    let mut wts = Vec::new();
    for i in (0..w * h).step_by(stride) {
        if !art.alpha[i] {
            continue;
        }
        let (x, y) = (i % w, i / w);
        let c = art.lab[i];
        let mut m = 0.0f32;
        for d in N4 {
            if let Some(j) = nb(w, h, x, y, d) {
                m = m.max(d2(c, art.lab[j]).sqrt());
            }
        }
        pts.push(c);
        wts.push(1.0 + 6.0 * m.min(0.3));
    }
    (pts, wts)
}

/// Weighted k-means++ (deterministic seeding) in scaled OKLab.
fn kmeans(pts: &[Lab], wts: &[f32], k: usize, scale: [f32; 3], iters: usize) -> Vec<Lab> {
    let sc = |p: Lab| [p[0] * scale[0], p[1] * scale[1], p[2] * scale[2]];
    let k = k.min(pts.len()).max(1);
    let sp: Vec<Lab> = pts.iter().map(|p| sc(*p)).collect();
    // Seed: the heaviest-weighted median-ish point, then D² sampling.
    let mut cent: Vec<Lab> = vec![sp[sp.len() / 2]];
    let mut dist: Vec<f32> = sp.iter().map(|p| d2(*p, cent[0])).collect();
    for ci in 1..k {
        let total: f32 = dist.iter().zip(wts).map(|(d, w)| d * w).sum();
        if total <= 1e-12 {
            break;
        }
        let mut r = hash01(ci as u64 * 7919 + sp.len() as u64) * total;
        let mut pick = sp.len() - 1;
        for (i, (d, w)) in dist.iter().zip(wts).enumerate() {
            r -= d * w;
            if r <= 0.0 {
                pick = i;
                break;
            }
        }
        let c = sp[pick];
        cent.push(c);
        for (d, p) in dist.iter_mut().zip(&sp) {
            *d = d.min(d2(*p, c));
        }
    }
    for _ in 0..iters {
        let sums = sp
            .par_chunks(4096)
            .zip(wts.par_chunks(4096))
            .map(|(ps, ws)| {
                let mut s = vec![[0.0f32; 4]; cent.len()];
                for (p, w) in ps.iter().zip(ws) {
                    let j = nearest(&cent, *p);
                    (0..3).for_each(|c| s[j][c] += p[c] * w);
                    s[j][3] += w;
                }
                s
            })
            .reduce(
                || vec![[0.0f32; 4]; cent.len()],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b) {
                        (0..4).for_each(|c| x[c] += y[c]);
                    }
                    a
                },
            );
        let mut moved = 0.0f32;
        for (c, s) in cent.iter_mut().zip(&sums) {
            if s[3] > 0.0 {
                let n = [s[0] / s[3], s[1] / s[3], s[2] / s[3]];
                moved = moved.max(d2(*c, n));
                *c = n;
            }
        }
        if moved < 1e-8 {
            break;
        }
    }
    cent.into_iter().map(|c| gamut_fit([c[0] / scale[0], c[1] / scale[1], c[2] / scale[2]])).collect()
}

/// Ramps: an accurate k-means palette, then every colour is nudged along
/// its hue family's ramp: darker members of a family lean violet-blue and
/// gain a little chroma, lighter ones lean warm and soften, the way pixel
/// artists build ramps. Big flat areas keep their colour (members near the
/// family's centre barely move).
fn ramps_palette(pts: &[Lab], wts: &[f32], colors: usize, hue_shift: f32) -> Vec<Lab> {
    let mut pal = kmeans(pts, wts, colors, [1.0, 1.0, 1.0], 14);
    let families = (colors / 4).clamp(2, 10);
    // Families cluster on hue (a/b) with saturated colours weighted up.
    let cw: Vec<f32> = pts.iter().zip(wts).map(|(p, w)| w * (1.0 + 6.0 * chroma(*p))).collect();
    let fam = kmeans(pts, &cw, families, [0.35, 1.0, 1.0], 12);
    let of: Vec<usize> = pal.iter().map(|c| nearest_scaled(&fam, *c)).collect();
    let hs = hue_shift.clamp(0.0, 1.0);
    for (f, fc) in fam.iter().enumerate() {
        let mem: Vec<usize> = (0..pal.len()).filter(|i| of[*i] == f).collect();
        if mem.len() < 2 {
            continue;
        }
        let ls: Vec<f32> = mem.iter().map(|i| pal[*i][0]).collect();
        let lmin = ls.iter().copied().fold(f32::MAX, f32::min);
        let lmax = ls.iter().copied().fold(f32::MIN, f32::max);
        let lc = fc[0];
        let half = ((lmax - lmin) * 0.5).max(0.05);
        for &i in &mem {
            let c = pal[i];
            let s = ((c[0] - lc) / half).clamp(-1.0, 1.0);
            let ch = chroma(c);
            if ch < 0.012 {
                continue;
            }
            let h = shift_hue(hue_deg(c), s, 10.0 * hs);
            let ch = ch * if s < 0.0 { 1.0 + 0.12 * hs * -s } else { 1.0 - 0.15 * hs * s * s };
            pal[i] = lch_to_lab(c[0], ch, h);
        }
    }
    pal
}

fn nearest_scaled(fam: &[Lab], p: Lab) -> usize {
    let mut best = (0, f32::MAX);
    for (i, c) in fam.iter().enumerate() {
        let d = 0.35 * 0.35 * (c[0] - p[0]).powi(2) + (c[1] - p[1]).powi(2) + (c[2] - p[2]).powi(2);
        if d < best.1 {
            best = (i, d);
        }
    }
    best.0
}

// ------------------------------------------------------------------ quantise

const BAYER4: [[f32; 4]; 4] = [[0.0, 8.0, 2.0, 10.0], [12.0, 4.0, 14.0, 6.0], [3.0, 11.0, 1.0, 9.0], [15.0, 7.0, 13.0, 5.0]];

fn bayer(x: usize, y: usize) -> f32 {
    (BAYER4[y % 4][x % 4] + 0.5) / 16.0
}

/// Gradient strength of the art image (max OKLab step to a 4-neighbour).
fn edge_map(art: &Art) -> Vec<f32> {
    let (w, h) = (art.w, art.h);
    (0..w * h)
        .into_par_iter()
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let mut m = 0.0f32;
            for d in N4 {
                if let Some(j) = nb(w, h, x, y, d) {
                    m = m.max(if art.alpha[i] != art.alpha[j] { 1.0 } else { d2(art.lab[i], art.lab[j]).sqrt() });
                }
            }
            m
        })
        .collect()
}

/// Palette indices; `dz` marks pixels inside dither zones (cleanup leaves
/// those alone).
fn quantise(art: &Art, pal: &[Lab], p: &PixelArt, edges: &[f32]) -> (Vec<u16>, Vec<bool>) {
    let (w, h) = (art.w, art.h);
    let amt = p.dither_amount.clamp(0.0, 1.0);
    let n = w * h;
    match p.dither.as_str() {
        "pattern" if amt > 0.0 => {
            let r: Vec<(u16, bool)> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let c = art.lab[i];
                    let a = nearest(pal, c);
                    // No dither on edges: they must stay clean.
                    if edges[i] > 0.09 {
                        return (a as u16, false);
                    }
                    let ca = pal[a];
                    // Second colour: the one whose segment from `a` passes
                    // closest to the pixel, not too far apart to mix.
                    let mut best: Option<(usize, f32, f32)> = None;
                    for (b, cb) in pal.iter().enumerate() {
                        if b == a {
                            continue;
                        }
                        let ab = [cb[0] - ca[0], cb[1] - ca[1], cb[2] - ca[2]];
                        let len2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
                        if len2 < 1e-8 || len2 > 0.14 * 0.14 {
                            continue;
                        }
                        let t = ((c[0] - ca[0]) * ab[0] + (c[1] - ca[1]) * ab[1] + (c[2] - ca[2]) * ab[2]) / len2;
                        if t <= 0.0 {
                            continue;
                        }
                        let t = t.min(1.0);
                        let proj = [ca[0] + ab[0] * t, ca[1] + ab[1] * t, ca[2] + ab[2] * t];
                        let e = d2(proj, c);
                        if best.is_none_or(|b| e < b.1) {
                            best = Some((b, e, t));
                        }
                    }
                    let Some((b, e, t)) = best else { return (a as u16, false) };
                    if e > d2(ca, c) * 0.7 {
                        return (a as u16, false);
                    }
                    // Amount narrows the mixed band around t = 0.5: low =
                    // dither only right at the transition.
                    let tt = ((t - 0.5) / amt.max(0.05) + 0.5).clamp(0.0, 1.0);
                    let zone = tt > 0.0 && tt < 1.0;
                    let pick = if tt > bayer(i % w, i / w) { b } else { a };
                    (pick as u16, zone)
                })
                .collect();
            r.into_iter().unzip()
        }
        "bayer" if amt > 0.0 => {
            let spread = 0.25 * amt / (pal.len() as f32).sqrt();
            let idx = (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut c = art.lab[i];
                    c[0] += (bayer(i % w, i / w) - 0.5) * spread * 4.0;
                    nearest(pal, c) as u16
                })
                .collect();
            (idx, vec![true; n])
        }
        "diffuse" if amt > 0.0 => {
            // Serpentine error diffusion, error damped and stopped at edges.
            let mut buf = art.lab.clone();
            let mut idx = vec![0u16; n];
            let k = 0.8 * amt;
            for y in 0..h {
                let rev = y % 2 == 1;
                for xi in 0..w {
                    let x = if rev { w - 1 - xi } else { xi };
                    let i = y * w + x;
                    let c = buf[i];
                    let j = nearest(pal, c);
                    idx[i] = j as u16;
                    if !art.alpha[i] || edges[i] > 0.15 {
                        continue;
                    }
                    let err = [(c[0] - pal[j][0]) * k, (c[1] - pal[j][1]) * k, (c[2] - pal[j][2]) * k];
                    let dir: isize = if rev { -1 } else { 1 };
                    for (dx, dy, f) in [(dir, 0isize, 7.0 / 16.0), (-dir, 1, 3.0 / 16.0), (0, 1, 5.0 / 16.0), (dir, 1, 1.0 / 16.0)] {
                        if let Some(q) = nb(w, h, x, y, (dx, dy)) {
                            if art.alpha[q] && edges[q] <= 0.15 {
                                (0..3).for_each(|c| buf[q][c] += err[c] * f);
                            }
                        }
                    }
                }
            }
            (idx, vec![true; n])
        }
        _ => ((0..n).into_par_iter().map(|i| nearest(pal, art.lab[i]) as u16).collect(), vec![false; n]),
    }
}

// ------------------------------------------------------------------ cleanup

fn cleanup(idx: &mut [u16], dz: &[bool], alpha: &[bool], pal: &[Lab], w: usize, h: usize) {
    // 1. Orphans: no same-colour 8-neighbour and close in colour to the
    //    surrounding majority -> take the majority. High-contrast specks
    //    (eye glints, stars) stay.
    for _ in 0..2 {
        let src = idx.to_vec();
        let mut changed = false;
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if dz[i] || !alpha[i] {
                    continue;
                }
                let me = src[i];
                let mut counts: [(u16, u8); 8] = [(u16::MAX, 0); 8];
                let mut alone = true;
                for d in N8 {
                    let Some(j) = nb(w, h, x, y, d) else { continue };
                    if !alpha[j] {
                        continue;
                    }
                    if src[j] == me {
                        alone = false;
                        break;
                    }
                    if let Some(e) = counts.iter_mut().find(|e| e.0 == src[j] || e.1 == 0) {
                        e.0 = src[j];
                        e.1 += 1;
                    }
                }
                if !alone {
                    continue;
                }
                let Some(maj) = counts.iter().filter(|e| e.1 > 0).max_by_key(|e| e.1) else { continue };
                if d2(pal[me as usize], pal[maj.0 as usize]).sqrt() < 0.16 {
                    idx[i] = maj.0;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    // 2. Pixel-perfect lines: drop the inner corner of an L in a 1-px line
    //    (the two arms still touch diagonally).
    let src = idx.to_vec();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if dz[i] || !alpha[i] {
                continue;
            }
            let me = src[i];
            let same = |d: (isize, isize)| nb(w, h, x, y, d).is_some_and(|j| src[j] == me);
            let n8: usize = N8.iter().filter(|d| same(**d)).count();
            if n8 != 2 {
                continue;
            }
            let corner = [((0, -1), (1, 0)), ((1, 0), (0, 1)), ((0, 1), (-1, 0)), ((-1, 0), (0, -1))]
                .iter()
                .any(|(a, b)| same(*a) && same(*b));
            if !corner {
                continue;
            }
            // Replace with the most common other neighbour.
            let mut counts: Vec<(u16, u8)> = Vec::new();
            for d in N4 {
                if let Some(j) = nb(w, h, x, y, d) {
                    if src[j] != me && alpha[j] {
                        match counts.iter_mut().find(|e| e.0 == src[j]) {
                            Some(e) => e.1 += 1,
                            None => counts.push((src[j], 1)),
                        }
                    }
                }
            }
            if let Some(m) = counts.iter().max_by_key(|e| e.1) {
                if m.1 >= 2 {
                    idx[i] = m.0;
                }
            }
        }
    }
}

// ------------------------------------------------------------------ outlines

/// Palette entry nearest to `target` among those darker than `than` (L).
fn darker_entry(pal: &[Lab], target: Lab, than: f32) -> Option<usize> {
    pal.iter()
        .enumerate()
        .filter(|(_, c)| c[0] < than - 0.04)
        .min_by(|a, b| d2(*a.1, target).total_cmp(&d2(*b.1, target)))
        .map(|(i, _)| i)
}

fn lighter_entry(pal: &[Lab], target: Lab, than: f32) -> Option<usize> {
    pal.iter()
        .enumerate()
        .filter(|(_, c)| c[0] > than + 0.04)
        .min_by(|a, b| d2(*a.1, target).total_cmp(&d2(*b.1, target)))
        .map(|(i, _)| i)
}

/// The shade a pixel artist would outline / shadow `c` with: darker, a
/// bit more saturated, hue toward violet.
fn shade_of(c: Lab, depth: f32) -> Lab {
    let h = if chroma(c) < 0.02 { SHADOW_HUE } else { shift_hue(hue_deg(c), -1.0, 25.0) };
    lch_to_lab(c[0] - depth, (chroma(c) * 1.1).max(0.03), h)
}

fn light_of(c: Lab, lift: f32) -> Lab {
    let h = if chroma(c) < 0.02 { LIGHT_HUE } else { shift_hue(hue_deg(c), 1.0, 30.0) };
    lch_to_lab(c[0] + lift, chroma(c) * 0.8, h)
}

/// Where the line goes: the darker side of strong edges (judged on the
/// smooth art colours, so lines are continuous), grown along weaker edges
/// that touch strong ones (hysteresis), with stray dots and L-corners
/// removed so lines read as clean 1-px strokes.
fn line_mask(lab: &[Lab], alpha: &[bool], w: usize, h: usize, lines: f32) -> Vec<bool> {
    let thr = 0.22 - 0.1 * lines.clamp(0.0, 1.0);
    // Specks (dots of 1-2 px, e.g. small stars) get no ring around them.
    let speck: Vec<bool> = (0..w * h)
        .into_par_iter()
        .map(|j| {
            let (x, y) = (j % w, j / w);
            N8.iter().filter(|d| nb(w, h, x, y, **d).is_some_and(|k| d2(lab[j], lab[k]).sqrt() < 0.06)).count() <= 1
        })
        .collect();
    let edge_to = |i: usize, x: usize, y: usize, t: f32| {
        let me = lab[i];
        N4.iter().any(|d| {
            nb(w, h, x, y, *d).is_some_and(|j| {
                alpha[j] && !speck[j] && {
                    let o = lab[j];
                    o[0] > me[0] + 0.03 && d2(o, me).sqrt() > t
                }
            })
        })
    };
    let n = w * h;
    let strong: Vec<bool> = (0..n).into_par_iter().map(|i| alpha[i] && edge_to(i, i % w, i / w, thr)).collect();
    let weak: Vec<bool> = (0..n).into_par_iter().map(|i| alpha[i] && edge_to(i, i % w, i / w, thr * 0.75)).collect();
    let mut m = strong;
    for _ in 0..2 {
        let prev = m.clone();
        let mut grew = false;
        for i in 0..n {
            if prev[i] || !weak[i] {
                continue;
            }
            let (x, y) = (i % w, i / w);
            if N8.iter().any(|d| nb(w, h, x, y, *d).is_some_and(|j| prev[j])) {
                m[i] = true;
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    // Stray dots and pairs.
    let prev = m.clone();
    for i in 0..n {
        if !prev[i] {
            continue;
        }
        let (x, y) = (i % w, i / w);
        let k = N8.iter().filter(|d| nb(w, h, x, y, **d).is_some_and(|j| prev[j])).count();
        if k == 0 {
            m[i] = false;
        } else if k == 1 {
            // A pair with nothing else attached.
            let j = N8.iter().find_map(|d| nb(w, h, x, y, *d).filter(|j| prev[*j])).unwrap();
            let (jx, jy) = (j % w, j / w);
            if N8.iter().filter(|d| nb(w, h, jx, jy, **d).is_some_and(|q| prev[q])).count() == 1 {
                m[i] = false;
            }
        }
    }
    // Pixel-perfect: the inner corner of an L is redundant.
    let prev = m.clone();
    for i in 0..n {
        if !prev[i] {
            continue;
        }
        let (x, y) = (i % w, i / w);
        let on = |d: (isize, isize)| nb(w, h, x, y, d).is_some_and(|j| prev[j]);
        if N8.iter().filter(|d| on(**d)).count() != 2 {
            continue;
        }
        if [((0, -1), (1, 0)), ((1, 0), (0, 1)), ((0, 1), (-1, 0)), ((-1, 0), (0, -1))].iter().any(|(a, b)| on(*a) && on(*b)) {
            m[i] = false;
        }
    }
    m
}

#[allow(clippy::too_many_arguments)]
fn outlines(idx: &mut [u16], alpha: &mut [bool], base: &[u16], lab: &[Lab], pal: &[Lab], w: usize, h: usize, mode: &str, lines: f32) {
    if mode == "none" {
        return;
    }
    let darkest = (0..pal.len()).min_by(|a, b| pal[*a][0].total_cmp(&pal[*b][0])).unwrap_or(0) as u16;
    let ink_for = |c: Lab| -> u16 {
        match mode {
            "selective" => darker_entry(pal, shade_of(c, 0.2), c[0]).map(|i| i as u16).unwrap_or(darkest),
            _ => darkest,
        }
    };
    let src_alpha = alpha.to_vec();
    let mask = line_mask(lab, &src_alpha, w, h, lines);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !src_alpha[i] {
                // 1 px outside a cutout sprite.
                let touch = N4.iter().find_map(|d| nb(w, h, x, y, *d).filter(|j| src_alpha[*j]));
                if let Some(j) = touch {
                    idx[i] = ink_for(pal[base[j] as usize]);
                    alpha[i] = true;
                }
            } else if mask[i] {
                idx[i] = ink_for(pal[base[i] as usize]);
            }
        }
    }
}

/// Rim light: pixels whose light-facing side is empty or clearly darker
/// take the next lighter colour of their own hue (never a different one).
#[allow(clippy::too_many_arguments)]
fn rim_light(idx: &mut [u16], alpha: &[bool], lab: &[Lab], pal: &[Lab], w: usize, h: usize, amount: f32, light_deg: f32, lines: f32) {
    if amount <= 0.01 {
        return;
    }
    let a = light_deg.to_radians();
    let (lx, ly) = (a.cos(), -a.sin());
    // Neighbours that face the light (by angle).
    let dirs: Vec<(isize, isize)> = N8
        .iter()
        .copied()
        .filter(|(dx, dy)| {
            let n = ((dx * dx + dy * dy) as f32).sqrt();
            (*dx as f32 * lx + *dy as f32 * ly) / n > 0.55
        })
        .collect();
    let thr = 0.2 - 0.1 * lines.clamp(0.0, 1.0);
    let src = idx.to_vec();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !alpha[i] {
                continue;
            }
            let me = lab[i];
            let lit = dirs.iter().any(|d| match nb(w, h, x, y, *d) {
                Some(j) => !alpha[j] || {
                    let o = lab[j];
                    o[0] < me[0] - 0.1 && d2(o, me).sqrt() > thr
                },
                None => false,
            });
            if !lit {
                continue;
            }
            let cur = pal[src[i] as usize];
            let target = light_of(cur, 0.06 + 0.14 * amount);
            if let Some(j) = lighter_entry(pal, target, cur[0]) {
                if d2(pal[j], target).sqrt() < 0.06 + 0.08 * amount {
                    idx[i] = j as u16;
                }
            }
        }
    }
}

// ------------------------------------------------------------------ sparkles

/// Bright points worth a star: (strength, art pixel), spaced apart.
fn sparkle_points(art: &Art, amount: f32) -> Vec<(f32, usize)> {
    if amount <= 0.01 {
        return Vec::new();
    }
    let (w, h) = (art.w, art.h);
    let alpha = &art.alpha;
    // Candidates: cells with a bright peak, or art pixels much brighter
    // than their surroundings.
    let mut cand: Vec<(f32, usize)> = Vec::new();
    let pk_thr = 0.28 - 0.16 * amount;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !alpha[i] {
                continue;
            }
            let l = art.lab[i][0];
            let bright = art.peak_lab[i][0].max(l);
            if bright < 0.8 {
                continue;
            }
            // Isolated: a point, not the edge of a bright area (at most one
            // bright neighbour, e.g. a star straddling two cells).
            let mut ring = 0.0f32;
            let mut n = 0.0f32;
            let mut lit = 0;
            for d in N8 {
                if let Some(j) = nb(w, h, x, y, d) {
                    let o = art.lab[j][0];
                    ring += o;
                    n += 1.0;
                    if art.lab[j][0] > bright - pk_thr || art.peak_lab[j][0] > bright - 0.05 && art.peak[j] > pk_thr {
                        lit += 1;
                    }
                }
            }
            let ring = ring / n.max(1.0);
            // Stars sit on calm backgrounds (sky), not in textured areas
            // like hair highlights.
            let var = N8.iter().filter_map(|d| nb(w, h, x, y, *d)).map(|j| (art.lab[j][0] - ring).powi(2)).sum::<f32>() / n.max(1.0);
            // ...over a wider ring too (5x5 border): gaps in hair or
            // foliage are bright specks in a busy area.
            let (mut s5, mut q5, mut n5) = (0.0f32, 0.0f32, 0.0f32);
            for dy in -2..=2isize {
                for dx in -2..=2isize {
                    if dx.abs() == 2 || dy.abs() == 2 {
                        if let Some(j) = nb(w, h, x, y, (dx, dy)) {
                            let v = art.lab[j][0];
                            s5 += v;
                            q5 += v * v;
                            n5 += 1.0;
                        }
                    }
                }
            }
            let m5 = s5 / n5.max(1.0);
            let var = var.max(q5 / n5.max(1.0) - m5 * m5);
            let s = art.peak[i].max(l - ring);
            if var.sqrt() < 0.08 && lit <= 1 && s > pk_thr && bright > ring + pk_thr {
                cand.push((s + 0.5 * (bright - 0.8), i));
            }
        }
    }
    cand.sort_by(|a, b| b.0.total_cmp(&a.0));
    let max = ((w * h) as f32 * 0.0015 * amount).ceil() as usize + 1;
    let mut taken: Vec<(f32, usize)> = Vec::new();
    let min_gap = 3isize;
    for (s, i) in cand {
        if taken.len() >= max {
            break;
        }
        let (x, y) = ((i % w) as isize, (i / w) as isize);
        if taken.iter().any(|(_, t)| ((t % w) as isize - x).abs() < min_gap && ((t / w) as isize - y).abs() < min_gap) {
            continue;
        }
        taken.push((s, i));
    }
    taken
}

/// The star colour (and its glow) for adaptive palettes, so sparkles have
/// a real highlight to use.
fn star_colors(art: &Art, stars: &[(f32, usize)]) -> Vec<Lab> {
    if stars.is_empty() {
        return Vec::new();
    }
    let pk = mean_of(stars.iter().map(|(_, i)| art.peak_lab[*i]));
    let bg = mean_of(stars.iter().map(|(_, i)| art.lab[*i]));
    let core = lch_to_lab(pk[0].max(0.95), chroma(pk) * 0.5, hue_deg(pk));
    let glow = [(core[0] + bg[0]) * 0.5 + 0.08, (core[1] + bg[1]) * 0.5, (core[2] + bg[2]) * 0.5];
    vec![core, gamut_fit(glow)]
}

fn sparkles(idx: &mut [u16], alpha: &[bool], art: &Art, pal: &[Lab], amount: f32, stars: &[(f32, usize)]) {
    let w = art.w;
    let h = art.h;
    let lightest = (0..pal.len()).max_by(|a, b| pal[*a][0].total_cmp(&pal[*b][0])).unwrap_or(0);
    for &(s, i) in stars {
        let (x, y) = (i % w, i / w);
        let arm = (1.0 + amount * 2.5 * (s / 0.4).min(1.5)).round() as isize;
        let core = pal[lightest];
        let bg = art.lab[i];
        let mix = |t: f32| -> u16 {
            let c = [core[0] + (bg[0] - core[0]) * t, core[1] + (bg[1] - core[1]) * t, core[2] + (bg[2] - core[2]) * t];
            nearest(pal, c) as u16
        };
        idx[i] = lightest as u16;
        for (dx, dy) in N4 {
            for k in 1..=arm {
                let t = (k as f32 / (arm + 1) as f32) * 0.85;
                if let Some(j) = nb(w, h, x, y, (dx * k, dy * k)) {
                    if alpha[j] {
                        let c = mix(t);
                        // Only brighten: arms never darken what's there.
                        if pal[c as usize][0] > pal[idx[j] as usize][0] {
                            idx[j] = c;
                        }
                    }
                }
            }
        }
        if arm >= 3 {
            for (dx, dy) in [(1, 1), (-1, -1), (1, -1), (-1, 1)] {
                if let Some(j) = nb(w, h, x, y, (dx, dy)) {
                    let c = mix(0.6);
                    if alpha[j] && pal[c as usize][0] > pal[idx[j] as usize][0] {
                        idx[j] = c;
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------ entry

/// Output size for `img`: the same size, or the art size when `native`.
pub fn output_size(w: usize, h: usize, p: &PixelArt) -> (usize, usize) {
    if p.native {
        let (_, aw, ah) = grid(w, h, p);
        (aw, ah)
    } else {
        (w, h)
    }
}

/// Telemetry: what the engine made of the picture with these settings —
/// palette use, dither / outline coverage, sparkles, stage timings. Full
/// renders always; live previews one in 16.
#[allow(clippy::too_many_arguments)]
fn report(p: &PixelArt, aw: usize, ah: usize, cell: usize, pal: &[Lab], idx: &[u16], dz: &[bool], line_px: usize, stars: usize, t_sample: f64, t_palette: f64, ms: f64, src_px: usize) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static N: AtomicU32 = AtomicU32::new(0);
    if !crate::telemetry::enabled() {
        return;
    }
    let preview = src_px < 2_000_000;
    if preview && N.fetch_add(1, Ordering::Relaxed) % 16 != 0 {
        return;
    }
    let n = (aw * ah).max(1) as f64;
    let mut used = vec![0u32; pal.len()];
    for i in idx {
        if let Some(u) = used.get_mut(*i as usize) {
            *u += 1;
        }
    }
    let colours = used.iter().filter(|u| **u > 0).count();
    // Share of the most used colour (flat posters vs busy scenes).
    let top = used.iter().copied().max().unwrap_or(0) as f64 / n;
    let mut e = crate::telemetry::event("pixelart", "render").ms(ms).flag("preview", preview);
    if let Ok(v) = serde_json::to_value(p) {
        e = e.json("p", &v);
    }
    e.metric("art_w", aw as f64)
        .metric("art_h", ah as f64)
        .metric("cell", cell as f64)
        .metric("palette", pal.len() as f64)
        .metric("colours_used", colours as f64)
        .metric("top_colour_share", top)
        .metric("dither_share", dz.iter().filter(|z| **z).count() as f64 / n)
        .metric("outline_share", line_px as f64 / n)
        .metric("stars", stars as f64)
        .metric("ms_sample", t_sample)
        .metric("ms_palette", t_palette)
        .metric("src_mp", src_px as f64 / 1e6)
        .emit();
}

pub fn render(img: &Image, p: &PixelArt) -> Result<Image, String> {
    p.validate()?;
    if img.w == 0 || img.h == 0 {
        return Ok(img.clone());
    }
    let t0 = std::time::Instant::now();
    let (cell, aw, ah) = grid(img.w, img.h, p);
    let mut art = sample_cells(img, cell, aw, ah, p.lines);
    abstract_pass(&mut art, p.smooth.clamp(0.0, 1.0));
    look(&mut art, p.punch, p.hue_shift);
    let t_sample = t0.elapsed().as_secs_f64() * 1000.0;
    let stars = sparkle_points(&art, p.sparkle.clamp(0.0, 1.0));
    let mut pal = build_palette(&art, p);
    if p.custom.is_empty() && palette_by_id(&p.palette).is_none() {
        pal.extend(star_colors(&art, &stars));
    }
    let t_palette = t0.elapsed().as_secs_f64() * 1000.0 - t_sample;
    let edges = edge_map(&art);
    let (mut idx, dz) = quantise(&art, &pal, p, &edges);
    let mut alpha = art.alpha.clone();
    if p.cleanup {
        cleanup(&mut idx, &dz, &alpha, &pal, aw, ah);
    }
    // Lines and light are judged on the un-dithered colours.
    let base: Vec<u16> = if dz.iter().any(|z| *z) {
        let mut b: Vec<u16> = art.lab.par_iter().map(|c| nearest(&pal, *c) as u16).collect();
        if p.cleanup {
            cleanup(&mut b, &vec![false; aw * ah], &alpha, &pal, aw, ah);
        }
        b
    } else {
        idx.clone()
    };
    rim_light(&mut idx, &alpha, &art.lab, &pal, aw, ah, p.rim, p.light, p.lines);
    let before_lines = idx.clone();
    outlines(&mut idx, &mut alpha, &base, &art.lab, &pal, aw, ah, &p.outline, p.lines);
    let line_px = idx.iter().zip(&before_lines).filter(|(a, b)| a != b).count();
    sparkles(&mut idx, &alpha, &art, &pal, p.sparkle.clamp(0.0, 1.0), &stars);
    report(p, aw, ah, cell, &pal, &idx, &dz, line_px, stars.len(), t_sample, t_palette, t0.elapsed().as_secs_f64() * 1000.0, img.w * img.h);

    let rgb: Vec<[f32; 3]> = pal.iter().map(|c| oklab_to_srgb(*c)).collect();
    let small: Vec<[f32; 4]> = idx
        .iter()
        .zip(&alpha)
        .map(|(i, a)| {
            let c = rgb[*i as usize];
            if *a { [c[0], c[1], c[2], 1.0] } else { [0.0, 0.0, 0.0, 0.0] }
        })
        .collect();
    if p.native {
        return Ok(Image { w: aw, h: ah, px: small });
    }
    let mut out = Image::new(img.w, img.h);
    out.px.par_chunks_mut(img.w).enumerate().for_each(|(y, row)| {
        let sy = (y / cell).min(ah - 1) * aw;
        for (x, o) in row.iter_mut().enumerate() {
            *o = small[sy + (x / cell).min(aw - 1)];
        }
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_image(w: usize, h: usize) -> Image {
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
                let inside = (fx - 0.5).powi(2) + (fy - 0.5).powi(2) < 0.08;
                img.px[y * w + x] = if inside { [0.9, 0.5, 0.7, 1.0] } else { [0.1 + 0.3 * fy, 0.1, 0.3 + 0.4 * fx, 1.0] };
            }
        }
        // A star.
        img.px[(h / 5) * w + w / 5] = [1.0, 1.0, 1.0, 1.0];
        img
    }

    #[test]
    fn oklab_round_trip() {
        for c in [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.2, 0.6, 0.9], [0.5, 0.5, 0.5]] {
            let r = oklab_to_srgb(srgb_to_oklab(c));
            for k in 0..3 {
                assert!((r[k] - c[k]).abs() < 2e-3, "{c:?} -> {r:?}");
            }
        }
    }

    #[test]
    fn every_style_and_mode_runs() {
        let img = test_image(200, 150);
        for s in STYLES {
            let p = (s.make)();
            let o = render(&img, &p).unwrap();
            assert_eq!((o.w, o.h), (200, 150), "{}", s.id);
        }
        for d in DITHERS {
            for o in OUTLINES {
                let p = PixelArt { width: 50, dither: d.0.into(), outline: o.0.into(), rim: 0.5, sparkle: 1.0, ..PixelArt::default() };
                let out = render(&img, &p).unwrap();
                assert!(out.px.iter().all(|q| q.iter().all(|v| v.is_finite())));
            }
        }
    }

    #[test]
    fn palette_is_respected_and_cells_are_flat() {
        let img = test_image(160, 120);
        let p = PixelArt { width: 40, palette: "handheld".into(), ..PixelArt::default() };
        let o = render(&img, &p).unwrap();
        let pal = palette_colors("handheld");
        for q in &o.px {
            assert!(pal.iter().any(|c| (0..3).all(|k| (c[k] - q[k]).abs() < 1e-4)));
        }
        // 160 / 40 = 4 px cells: every 4x4 block is one colour.
        for y in 0..120 {
            for x in 0..160 {
                assert_eq!(o.px[y * 160 + x], o.px[(y / 4 * 4) * 160 + x / 4 * 4]);
            }
        }
        let auto = render(&img, &PixelArt { width: 40, colors: 8, dither: "none".into(), ..PixelArt::default() }).unwrap();
        let mut distinct: Vec<[u32; 3]> = auto.px.iter().map(|p| [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()]).collect();
        distinct.sort();
        distinct.dedup();
        assert!(distinct.len() <= 8);
    }

    #[test]
    fn native_and_legacy() {
        let img = test_image(120, 90);
        let o = render(&img, &PixelArt { width: 30, native: true, ..PixelArt::default() }).unwrap();
        assert_eq!((o.w, o.h), (30, 23));
        assert_eq!(output_size(120, 90, &PixelArt { width: 30, native: true, ..PixelArt::default() }), (30, 23));
        let legacy: PixelArt = serde_json::from_str(r#"{"cell":4,"colors":4,"dither":true,"outline":false}"#).unwrap();
        assert_eq!((legacy.dither.as_str(), legacy.outline.as_str()), ("bayer", "none"));
        assert!(render(&img, &legacy).is_ok());
    }

    #[test]
    fn sparkle_keeps_the_star() {
        let img = test_image(200, 200);
        let p = PixelArt { width: 50, sparkle: 1.0, dither: "none".into(), ..PixelArt::default() };
        let o = render(&img, &p).unwrap();
        // The 1-px star at (40, 40) is far below a 4 px cell, yet the art
        // keeps a bright pixel there.
        let l = |q: [f32; 4]| Image::luma(q);
        let around = (36..48).flat_map(|y| (36..48).map(move |x| (x, y))).map(|(x, y)| l(o.px[y * 200 + x])).fold(0.0f32, f32::max);
        assert!(around > 0.85, "{around}");
    }

    #[test]
    fn cutout_gets_outline_outside() {
        let mut img = Image::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let inside = (16..48).contains(&x) && (16..48).contains(&y);
                img.px[y * 64 + x] = if inside { [0.8, 0.3, 0.3, 1.0] } else { [0.0, 0.0, 0.0, 0.0] };
            }
        }
        let p = PixelArt { width: 16, outline: "selective".into(), native: true, ..PixelArt::default() };
        let o = render(&img, &p).unwrap();
        assert!(o.px[4 * 16 + 3][3] > 0.5, "ring pixel left of the sprite is opaque");
        assert!(o.px[0][3] < 0.5);
    }
}
