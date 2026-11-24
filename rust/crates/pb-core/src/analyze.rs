//! Smart settings: look at the photo itself and choose every upscale
//! setting for it.
//!
//! Measured on the pixels (no metadata, no ML):
//! - **content**: drawn art vs photo/render (`classify::stats`)
//! - **noise**: Immerkær's fast noise estimate on non-edge pixels, at the
//!   native resolution (a centre crop, so downscaling can't hide it)
//! - **blocking**: JPEG 8x8 grid strength — luma steps across block
//!   boundaries vs inside blocks
//! - **softness**: how much of the image's gradient energy survives at
//!   pixel scale vs half scale; an already upscaled / out-of-focus photo has
//!   little pixel-scale detail
//! - **skin**: share of skin-toned pixels in the middle of the frame
//!   (portraits want the model that doesn't invent texture)
//!
//! The recommendation then trades quality against a compute budget: the
//! phone's measured speed and how long the user is willing to wait.

use crate::classify;
use crate::image::Image;
use crate::models;
use crate::resample::{resize, Filter};
use rayon::prelude::*;
use serde::Serialize;

#[derive(Serialize, Debug, Clone, Copy)]
pub struct Measures {
    pub width: usize,
    pub height: usize,
    /// "anime" or "photo".
    pub kind: &'static str,
    /// Estimated noise standard deviation in 8-bit levels.
    pub noise: f32,
    /// JPEG blocking, 0 (none) .. 1 (heavy).
    pub blocking: f32,
    /// Pixel-scale sharpness, ~0.45 (very soft) .. ~0.85+ (crisp).
    pub sharpness: f32,
    /// Share of skin-toned pixels in the central area, 0..1.
    pub skin: f32,
    /// Mean luma, 0..1.
    pub brightness: f32,
    pub has_alpha: bool,
    /// Share of flat (single-colour) neighbourhoods (from the classifier).
    pub flat: f32,
    /// Real resolution: the detail is this many times coarser than the
    /// pixels (1 = sharp; 3 = an image blown up 3x before it got here).
    pub eff: f32,
    /// Drawn-art evidence at the real resolution (classify::Ink).
    pub line: f32,
    pub palette: f32,
}

#[derive(Serialize, Debug, Clone)]
pub struct Recommendation {
    pub model: String,
    pub scale: f32,
    /// "off" / "natural" / "faithful".
    pub fidelity: String,
    pub detail: f32,
    pub denoise: f32,
    pub ensemble: usize,
    /// Photo model texture strength (weight interpolation with Clean).
    pub realism: f32,
    /// Photo + Clean fused per pixel by source texture (2x compute).
    pub adaptive: bool,
    /// Shrink the input to its real resolution first (1 = no): an image
    /// that was blown up before gives the network nothing at pixel scale,
    /// so it runs on the real detail and upscales from there.
    pub prescale: f32,
    /// Expected run time on the budget's device (seconds).
    pub est_seconds: f32,
    /// Short human-readable reasons, most important first.
    pub reasons: Vec<String>,
    /// One-line summary, e.g. "Noisy photo · 6× · Fast + denoise".
    pub summary: String,
}

#[derive(Debug, Clone, Copy)]
pub struct Budget {
    /// Device speed in GFLOP/s (the app learns it from real runs).
    pub gflops: f32,
    /// Longest acceptable run (seconds).
    pub max_seconds: f32,
    /// Output cap in megapixels.
    pub max_output_mp: f32,
    /// Running on the server: the budget is not a phone's.
    pub cloud: bool,
}

impl Default for Budget {
    fn default() -> Self {
        Budget { gflops: 40.0, max_seconds: 180.0, max_output_mp: 120.0, cloud: false }
    }
}

/// Native-resolution centre crop, at most `n` x `n`.
fn centre_crop(img: &Image, n: usize) -> Image {
    let (w, h) = (img.w.min(n), img.h.min(n));
    // Keep the crop on the 8x8 JPEG grid so blocking is measured in phase.
    let x = ((img.w - w) / 2) & !7;
    let y = ((img.h - h) / 2) & !7;
    img.crop(x, y, w, h)
}

fn luma255(img: &Image) -> Vec<f32> {
    img.px.par_iter().map(|p| Image::luma(*p) * 255.0).collect()
}

/// Immerkær (1996) noise sigma, restricted to pixels whose Sobel gradient
/// is below the 60th percentile so edges and texture don't count as noise.
fn noise_sigma(l: &[f32], w: usize, h: usize) -> f32 {
    if w < 8 || h < 8 {
        return 0.0;
    }
    let rows: Vec<(Vec<f32>, Vec<f32>)> = (1..h - 1)
        .into_par_iter()
        .map(|y| {
            let mut g = Vec::with_capacity(w);
            let mut c = Vec::with_capacity(w);
            for x in 1..w - 1 {
                let p = |dx: isize, dy: isize| l[(y as isize + dy) as usize * w + (x as isize + dx) as usize];
                let gx = p(1, -1) + 2.0 * p(1, 0) + p(1, 1) - p(-1, -1) - 2.0 * p(-1, 0) - p(-1, 1);
                let gy = p(-1, 1) + 2.0 * p(0, 1) + p(1, 1) - p(-1, -1) - 2.0 * p(0, -1) - p(1, -1);
                g.push(gx.abs() + gy.abs());
                let lap = p(-1, -1) - 2.0 * p(0, -1) + p(1, -1) - 2.0 * p(-1, 0) + 4.0 * p(0, 0) - 2.0 * p(1, 0) + p(-1, 1) - 2.0 * p(0, 1) + p(1, 1);
                c.push(lap.abs());
            }
            (g, c)
        })
        .collect();
    let mut grads: Vec<f32> = rows.iter().flat_map(|r| r.0.iter().copied()).collect();
    if grads.is_empty() {
        return 0.0;
    }
    let k = grads.len() * 6 / 10;
    let thr = *grads.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1;
    let (mut sum, mut n) = (0.0f64, 0usize);
    for (g, c) in &rows {
        for (gv, cv) in g.iter().zip(c) {
            if *gv <= thr {
                sum += *cv as f64;
                n += 1;
            }
        }
    }
    if n == 0 {
        return 0.0;
    }
    ((std::f64::consts::PI / 2.0).sqrt() * sum / (6.0 * n as f64)) as f32
}

/// JPEG blocking: mean luma step across 8-px block boundaries relative to
/// the steps inside blocks, both directions. Returns 0..1.
fn blocking(l: &[f32], w: usize, h: usize) -> f32 {
    if w < 32 || h < 32 {
        return 0.0;
    }
    let (mut bd, mut bn, mut id, mut inn) = (0.0f64, 0usize, 0.0f64, 0usize);
    for y in 0..h {
        for x in 0..w - 1 {
            let d = (l[y * w + x + 1] - l[y * w + x]).abs() as f64;
            // Ignore real edges: they are not grid artefacts.
            if d > 24.0 {
                continue;
            }
            if x % 8 == 7 {
                bd += d;
                bn += 1;
            } else if x % 8 == 3 || x % 8 == 4 {
                id += d;
                inn += 1;
            }
        }
    }
    for y in 0..h - 1 {
        for x in 0..w {
            let d = (l[(y + 1) * w + x] - l[y * w + x]).abs() as f64;
            if d > 24.0 {
                continue;
            }
            if y % 8 == 7 {
                bd += d;
                bn += 1;
            } else if y % 8 == 3 || y % 8 == 4 {
                id += d;
                inn += 1;
            }
        }
    }
    if bn == 0 || inn == 0 {
        return 0.0;
    }
    let b = bd / bn as f64;
    let i = id / inn as f64 + 0.15;
    // 1.0 = no grid; ~1.6+ = obvious blocks.
    ((b / i - 1.08) / 0.6).clamp(0.0, 1.0) as f32
}

/// Crété et al. (2007) blur metric, reported as sharpness = 1 - blur.
/// The image is re-blurred with a 9-tap box; a sharp image loses much of
/// its neighbour-to-neighbour variation, an already blurry one hardly any.
/// Crisp photos land around 0.65-0.8, soft / upsampled ones ~0.45-0.55
/// (measured on the gallery test set, blurred by 3x down-up).
fn sharpness(l: &[f32], w: usize, h: usize) -> f32 {
    if w < 24 || h < 24 {
        return 0.7;
    }
    let r = 4isize;
    let at = |x: isize, y: isize| l[y.clamp(0, h as isize - 1) as usize * w + x.clamp(0, w as isize - 1) as usize];
    let bh: Vec<f32> = (0..h * w).into_par_iter().map(|i| {
        let (x, y) = ((i % w) as isize, (i / w) as isize);
        (-r..=r).map(|k| at(x + k, y)).sum::<f32>() / 9.0
    }).collect();
    let bv: Vec<f32> = (0..h * w).into_par_iter().map(|i| {
        let (x, y) = ((i % w) as isize, (i / w) as isize);
        (-r..=r).map(|k| at(x, y + k)).sum::<f32>() / 9.0
    }).collect();
    let (mut fh, mut vh, mut fv, mut vv) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for y in 1..h {
        for x in 1..w {
            let i = y * w + x;
            let dfh = (l[i] - l[i - 1]).abs();
            let dbh = (bh[i] - bh[i - 1]).abs();
            let dfv = (l[i] - l[i - w]).abs();
            let dbv = (bv[i] - bv[i - w]).abs();
            fh += dfh as f64;
            vh += (dfh - dbh).max(0.0) as f64;
            fv += dfv as f64;
            vv += (dfv - dbv).max(0.0) as f64;
        }
    }
    if fh < 1.0 || fv < 1.0 {
        return 0.7;
    }
    let blur = ((fh - vh) / fh).max((fv - vv) / fv);
    (1.0 - blur) as f32
}

/// Skin tones (YCbCr box from Chai & Ngan, tightened) in the middle 60%.
fn skin_share(small: &Image) -> f32 {
    let (w, h) = (small.w, small.h);
    let (x0, x1, y0, y1) = (w / 5, w - w / 5, h / 5, h - h / 5);
    let (mut n, mut s) = (0usize, 0usize);
    for y in y0..y1 {
        for x in x0..x1 {
            let p = small.px[y * w + x];
            let (r, g, b) = (p[0] * 255.0, p[1] * 255.0, p[2] * 255.0);
            let yy = 0.299 * r + 0.587 * g + 0.114 * b;
            let cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b;
            let cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b;
            n += 1;
            if yy > 50.0 && (80.0..=125.0).contains(&cb) && (136.0..=170.0).contains(&cr) && r > g && r > b {
                s += 1;
            }
        }
    }
    s as f32 / n.max(1) as f32
}

/// Measures from a native-resolution centre crop (noise, blocking,
/// sharpness) and a small copy of the whole photo (content, skin, light).
pub fn measure_parts(crop: &Image, small: &Image, width: usize, height: usize, has_alpha: bool) -> Measures {
    let l = luma255(crop);
    let st = classify::stats(small);
    let eff = classify::effective_scale(crop);
    let ink = classify::ink(&classify::at_real_resolution(small, eff, width, 512));
    // Sharpness of what the network will actually see (after prescale).
    let sharp_src = if eff >= 1.5 { classify::at_real_resolution(crop, eff, crop.w, usize::MAX) } else { crop.clone() };
    let ls = luma255(&sharp_src);
    // Ink lines are decisive; flat-fill evidence alone needs a grain-free
    // picture (a sharp photo of a wall still has sensor noise).
    let kind = match classify::kind_of_ink(&ink, eff) {
        "anime" if ink.line > 0.0145 || l_noise(&l, crop) < 2.5 => "anime",
        _ => "photo",
    };
    let f = 384.0 / small.w.max(small.h) as f32;
    let tiny = if f < 1.0 { resize(small, ((small.w as f32 * f) as usize).max(8), ((small.h as f32 * f) as usize).max(8), Filter::Box) } else { small.clone() };
    let brightness = tiny.px.iter().map(|p| Image::luma(*p)).sum::<f32>() / tiny.px.len().max(1) as f32;
    Measures {
        width,
        height,
        kind,
        noise: noise_sigma(&l, crop.w, crop.h),
        blocking: blocking(&l, crop.w, crop.h),
        sharpness: sharpness(&ls, sharp_src.w, sharp_src.h),
        skin: skin_share(&tiny),
        brightness,
        has_alpha,
        flat: st.flat,
        eff,
        line: ink.line,
        palette: ink.palette,
    }
}

fn l_noise(l: &[f32], crop: &Image) -> f32 {
    noise_sigma(l, crop.w, crop.h)
}

fn shrink(img: &Image, max_side: usize) -> Image {
    let f = max_side as f32 / img.w.max(img.h) as f32;
    if f < 1.0 {
        resize(img, ((img.w as f32 * f) as usize).max(8), ((img.h as f32 * f) as usize).max(8), Filter::Box)
    } else {
        img.clone()
    }
}

pub fn measure(img: &Image) -> Measures {
    measure_parts(&centre_crop(img, 768), &shrink(img, 1024), img.w, img.h, img.has_alpha())
}

fn region<I: image::GenericImageView<Pixel = image::Rgba<u8>>>(buf: &I, x: u32, y: u32, w: u32, h: u32) -> Image {
    let mut px = Vec::with_capacity((w * h) as usize);
    for yy in y..y + h {
        for xx in x..x + w {
            let p = buf.get_pixel(xx, yy).0;
            px.push([p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0, p[3] as f32 / 255.0]);
        }
    }
    Image { w: w as usize, h: h as usize, px }
}

/// Measure a file without holding it as f32: only the decoded pixels
/// (RGB8 for photos, 3 B/px), a native centre crop and a <=1024 px copy.
/// A 64 MP JPEG peaks around 200 MB instead of ~1 GB as f32.
pub fn measure_file(path: &str) -> Result<Measures, String> {
    use image::GenericImageView;
    let mut r = image::ImageReader::open(path).map_err(|e| format!("{path}: {e}"))?.with_guessed_format().map_err(|e| e.to_string())?;
    r.no_limits();
    let d = r.decode().map_err(|e| format!("decode: {e}"))?;
    let (w, h) = d.dimensions();
    let has_alpha = match &d {
        image::DynamicImage::ImageRgba8(b) => b.pixels().any(|p| p.0[3] < 255),
        image::DynamicImage::ImageLumaA8(b) => b.pixels().any(|p| p.0[1] < 255),
        image::DynamicImage::ImageRgba16(b) => b.pixels().any(|p| p.0[3] < 65535),
        other => other.color().has_alpha(),
    };
    let (cw, ch) = (w.min(768), h.min(768));
    let crop = region(&d, ((w - cw) / 2) & !7, ((h - ch) / 2) & !7, cw, ch);
    let f = 1024.0 / w.max(h) as f32;
    let small = if f < 1.0 {
        let t = image::imageops::thumbnail(&d, ((w as f32 * f) as u32).max(8), ((h as f32 * f) as u32).max(8));
        region(&t, 0, 0, t.width(), t.height())
    } else {
        region(&d, 0, 0, w, h)
    };
    drop(d);
    Ok(measure_parts(&crop, &small, w as usize, h as usize, has_alpha))
}

/// Expected compute (GFLOP) of one upscale.
pub fn gflop(model: &str, w: usize, h: usize, scale: f32, ensemble: usize) -> f64 {
    let m = models::find(model).copied().unwrap_or(models::MODELS[0]);
    let px = (w * h) as f64;
    let first = 2.0 * 18.4e6 * m.cost as f64 * px / 1e9;
    // Above the model's own scale (+50% headroom, as in the pipeline) the
    // model runs again on its output.
    let s = m.scale as f64;
    let second = if (scale as f64) / s > 1.5 { first * s * s } else { 0.0 };
    (first + second) * ensemble.max(1) as f64
}

const SCALES: [f32; 5] = [2.0, 3.0, 4.0, 6.0, 8.0];

fn pct(v: f32) -> u32 {
    (v * 100.0).round() as u32
}

/// Settings the user chose themselves; `None` = decide from the photo.
#[derive(Debug, Clone, Default)]
pub struct Fixed {
    pub model: Option<String>,
    pub scale: Option<f32>,
    pub fidelity: Option<String>,
    pub detail: Option<f32>,
    pub ensemble: Option<usize>,
    pub denoise: Option<f32>,
    pub realism: Option<f32>,
    pub adaptive: Option<bool>,
    pub prescale: Option<f32>,
}

pub fn recommend(m: &Measures, b: Budget) -> Recommendation {
    recommend_fixed(m, b, &Fixed::default())
}

/// Choose every setting the user left on Auto, around the ones they fixed.
pub fn recommend_fixed(m: &Measures, b: Budget, fx: &Fixed) -> Recommendation {
    let mut why: Vec<String> = Vec::new();
    let anime = m.kind == "anime";
    let long = m.width.max(m.height) as f32;
    let mp_in = (m.width * m.height) as f32 / 1e6;
    let noisy = m.noise > 3.2;
    let very_noisy = m.noise > 7.0;
    let blocky = m.blocking > 0.3;
    let soft = m.sharpness < 0.58;
    let portrait = !anime && m.skin > 0.14;

    // ---- size: aim for a 4K-class long side, within the output cap.
    let target = if anime { 3840.0 } else { 4096.0 };
    let max_mp = if b.max_output_mp > 0.0 { b.max_output_mp } else { 120.0 };
    let mut scale = SCALES.iter().copied().find(|s| long * s >= target).unwrap_or(8.0);
    while scale > 2.0 && mp_in * scale * scale > max_mp {
        scale = SCALES.iter().copied().rev().find(|s| *s < scale).unwrap_or(2.0);
    }
    if let Some(s) = fx.scale.filter(|s| *s >= 1.0) {
        scale = s;
    } else if long >= target {
        why.push(format!("Already {}×{} — 2× is plenty", m.width, m.height));
    } else {
        why.push(format!("{}×{} → {}× gives about {} px on the long side", m.width, m.height, scale as u32, (long * scale) as u32));
    }
    // ---- real resolution: run the network on the real detail.
    // A network factor above ~6x means a second model pass over the 4x
    // intermediate (16x the pixels of the first). Worth it for images blown
    // up 3x or more, or while the intermediate stays small; otherwise shrink
    // less so one pass plus a mild resample covers the factor.
    let mut prescale = if m.eff >= 1.5 { 1.0 / m.eff } else { 1.0 };
    let intermediate_mp = (m.width * m.height) as f32 * prescale * prescale * 16.0 / 1e6;
    if m.eff < 3.0 && scale / prescale > 6.0 && intermediate_mp > 2.5 {
        prescale = (scale / 6.0).min(1.0);
    }
    if prescale > 0.85 {
        prescale = 1.0;
    }
    if let Some(p) = fx.prescale.filter(|p| *p > 0.0) {
        prescale = p.min(1.0);
    } else if prescale < 1.0 {
        why.push(format!("Detail is {:.1}× coarser than the pixels (blown up before) → the network works on the real detail, {}×{}", m.eff, (m.width as f32 * prescale) as u32, (m.height as f32 * prescale) as u32));
    }
    // Network input size and factor after the prescale.
    let (ew, eh) = (((m.width as f32 * prescale) as usize).max(1), ((m.height as f32 * prescale) as usize).max(1));
    let net_scale = scale / prescale;

    // ---- model + denoise
    let mut denoise = 0.0f32;
    let mut model: &str = if anime {
        if blocky {
            why.push("Drawn art with compression blocks → Line Art model (trained on compressed frames)".into());
            "lineart"
        } else {
            why.push("Drawn art: flat colour and hard lines → Anime model".into());
            "anime"
        }
    } else if noisy || blocky {
        denoise = ((m.noise - 2.5) / 8.0).max(m.blocking * 0.8).clamp(0.2, 1.0);
        if very_noisy {
            why.push(format!("Strong noise (σ≈{:.1}) → Fast model with {}% denoise", m.noise, pct(denoise)));
        } else if blocky {
            why.push(format!("JPEG blocking ({}%) → Fast model with {}% clean-up", pct(m.blocking), pct(denoise)));
        } else {
            why.push(format!("Some noise (σ≈{:.1}) → Fast model with {}% denoise", m.noise, pct(denoise)));
        }
        "fast"
    } else if portrait {
        "clean"
    } else if scale <= 2.0 {
        why.push("Clean photo at 2× → Native 2× network".into());
        "native2x"
    } else {
        why.push("Clean photo → Photo model for the best texture".into());
        "photo"
    };

    // ---- budget: fall back to a cheaper network when the phone would take
    // too long (the cloud has no such limit).
    let est = |model: &str, ens: usize| -> f32 { (gflop(model, ew, eh, net_scale, ens) / b.gflops.max(1.0) as f64) as f32 };
    // Adaptive fusion (Photo where the source has texture, Clean on skin
    // and flat areas) costs two networks: use it when it fits the budget.
    let mut adaptive = false;
    let mut realism = 1.0f32;
    let fits_fused = |e: usize| b.cloud || est("photo", 2 * e) <= b.max_seconds;
    if portrait && !very_noisy {
        if fits_fused(1) {
            model = "photo";
            adaptive = true;
            realism = 0.8;
            why.push(format!("Skin tones fill {}% of the middle → Adaptive: real texture on hair and clothes, faithful skin", pct(m.skin)));
        } else {
            why.push(format!("Skin tones fill {}% of the middle → Clean model (no invented skin texture)", pct(m.skin)));
        }
    } else if model == "photo" && (b.cloud || m.flat > 0.3) && fits_fused(1) {
        adaptive = true;
        why.push(if m.flat > 0.3 { "Large smooth areas (sky, walls) → Adaptive keeps them clean while texture gets rebuilt".to_string() } else { "Adaptive: texture rebuilt only where the photo has it".to_string() });
    }
    let cost = |model: &str, ens: usize, ad: bool| if ad { est("photo", 2 * ens) } else { est(model, ens) };
    if !b.cloud && cost(model, 1, adaptive) > b.max_seconds {
        let cheaper = if anime { "lineart" } else { "fast" };
        if cheaper != model {
            why.push(format!("{} would take ~{} on this iPhone → {} model instead", models::find(model).map(|i| i.name).unwrap_or(model), fmt_secs(cost(model, 1, adaptive)), models::find(cheaper).map(|i| i.name).unwrap_or(cheaper)));
            model = cheaper;
            adaptive = false;
        }
    }
    if let Some(fm) = fx.model.as_deref().filter(|m| *m != "auto") {
        if fm != model {
            why.push(format!("{} (your choice)", models::find(fm).map(|i| i.name).unwrap_or(fm)));
        }
        model = fm;
        adaptive = model == "photo" && fx.adaptive.unwrap_or(false);
        realism = fx.realism.unwrap_or(1.0);
        denoise = if model == "fast" { fx.denoise.unwrap_or(denoise) } else { 0.0 };
    } else {
        if let Some(a) = fx.adaptive {
            adaptive = a && model == "photo";
        }
        if let Some(r) = fx.realism.filter(|_| model == "photo") {
            realism = r;
        }
    }

    // ---- clarity (fidelity)
    let fixed_fid = fx.fidelity.as_deref().filter(|f| *f != "auto");
    let fidelity = if let Some(f) = fixed_fid {
        f
    } else if noisy || blocky {
        why.push("Natural clarity: Faithful would copy the noise and blocks back in".into());
        "natural"
    } else if portrait {
        why.push("Faithful clarity keeps faces exactly as shot".into());
        "faithful"
    } else if anime && m.flat > 0.45 {
        "faithful"
    } else {
        "natural"
    };

    // ---- detail boost
    let mut detail: f32 = if anime { 0.25 } else { 0.35 };
    if soft {
        detail += 0.25 * ((0.58 - m.sharpness) / 0.1).clamp(0.3, 1.0);
        why.push(format!("Looks soft (sharpness {:.2}) → more detail", m.sharpness));
    }
    if noisy {
        detail -= (0.04 * (m.noise - 3.2)).clamp(0.0, 0.25);
    }
    if blocky {
        detail -= 0.15 * m.blocking;
    }
    if portrait {
        detail = detail.min(0.22);
    }
    let detail = match fx.detail.filter(|d| *d >= 0.0) {
        Some(d) => d.min(1.0),
        None => detail.clamp(0.05, 0.7),
    };

    // ---- quality passes when they fit the budget
    let mut ensemble = 1;
    if let Some(e) = fx.ensemble.filter(|e| *e > 0) {
        ensemble = e;
    } else if b.cloud {
        // The server has time: all 8 passes when the network input is small.
        ensemble = if ew * eh <= 400_000 { 8 } else { 2 };
    } else {
        for e in [8usize, 2] {
            let limit = if e == 8 { 0.25 } else { 0.5 };
            if cost(model, e, adaptive) <= b.max_seconds * limit {
                ensemble = e;
                break;
            }
        }
    }
    if ensemble > 1 && fx.ensemble.is_none() {
        why.push(format!("{} quality fits the time budget", if ensemble >= 8 { "Max" } else { "High" }));
    }
    if m.has_alpha {
        why.push("Transparency is kept".into());
    }
    if m.brightness < 0.18 && !noisy {
        why.push("Dark photo but clean — no denoise needed".into());
    }

    let est_seconds = if b.cloud { 0.0 } else { cost(model, ensemble, adaptive) };
    let what = if anime {
        "Drawn art"
    } else if very_noisy {
        "Noisy photo"
    } else if blocky {
        "Compressed photo"
    } else if noisy {
        "Slightly noisy photo"
    } else if portrait {
        "Portrait"
    } else if soft {
        "Soft photo"
    } else {
        "Clean photo"
    };
    let name = models::find(model).map(|i| i.name).unwrap_or(model);
    let extra = if denoise > 0.0 && model == "fast" {
        " + denoise"
    } else if adaptive {
        " · Adaptive"
    } else {
        ""
    };
    let summary = format!("{what} · {}× · {name}{extra}", scale as u32);
    let r = Recommendation { model: model.into(), scale, fidelity: fidelity.into(), detail, denoise, ensemble, realism, adaptive, prescale, est_seconds, reasons: why, summary };
    report(m, b, fx, &r);
    r
}

/// Telemetry: every measurement next to every decision, the budget and
/// which fields the user had fixed — what the analyzer thresholds get
/// re-fitted from (together with later Auto overrides and ratings).
fn report(m: &Measures, b: Budget, fx: &Fixed, r: &Recommendation) {
    let e = crate::telemetry::event("analyze", "recommend");
    if !e.active() {
        return;
    }
    let fixed: Vec<&str> = [
        ("model", fx.model.is_some()),
        ("scale", fx.scale.is_some()),
        ("fidelity", fx.fidelity.is_some()),
        ("detail", fx.detail.is_some()),
        ("ensemble", fx.ensemble.is_some()),
        ("denoise", fx.denoise.is_some()),
        ("realism", fx.realism.is_some()),
        ("adaptive", fx.adaptive.is_some()),
        ("prescale", fx.prescale.is_some()),
    ]
    .iter()
    .filter(|f| f.1)
    .map(|f| f.0)
    .collect();
    let mut rv = serde_json::to_value(r).unwrap_or_default();
    if let Some(o) = rv.as_object_mut() {
        o.remove("reasons");
        o.remove("summary");
    }
    e.json("m", &serde_json::to_value(m).unwrap_or_default())
        .json("rec", &rv)
        .metric("budget.gflops", b.gflops)
        .metric("budget.seconds", b.max_seconds)
        .metric("budget.max_mp", b.max_output_mp)
        .flag("budget.cloud", b.cloud)
        .metric("fixed.n", fixed.len() as f64)
        .tag("fixed", fixed.join(","))
        .emit();
}

fn fmt_secs(s: f32) -> String {
    if s < 90.0 {
        format!("{}s", s.round() as u32)
    } else if s < 3600.0 {
        format!("{} min", (s / 60.0).round() as u32)
    } else {
        format!("{:.1} h", s / 3600.0)
    }
}

pub fn analyze(img: &Image, b: Budget) -> (Measures, Recommendation) {
    let m = measure(img);
    let r = recommend(&m, b);
    (m, r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::hash_noise;

    /// Smooth synthetic "photo": gradients plus soft texture.
    fn scene(w: usize, h: usize) -> Image {
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f32, y as f32);
                let t = 0.5 + 0.2 * (fx * 0.05).sin() * (fy * 0.07).cos() + 0.1 * ((fx + fy) * 0.21).sin();
                let edge = if (x / 37 + y / 53) % 2 == 0 { 0.15 } else { -0.15 };
                // Fine texture (a sharp photo has pixel-scale detail).
                let tex = (hash_noise(x as u32 / 3, y as u32 / 3, 3) - 0.5) * 0.12;
                img.px[y * w + x] = [t + edge + tex, t * 0.9 + tex, t * 0.8 - edge * 0.5 + tex, 1.0];
            }
        }
        img
    }

    #[test]
    fn noise_is_detected_and_scales() {
        let clean = scene(512, 384);
        let mut noisy = clean.clone();
        for (i, p) in noisy.px.iter_mut().enumerate() {
            let n = (hash_noise((i % 512) as u32, (i / 512) as u32, 7) - 0.5) * 2.0 * 0.06;
            for c in 0..3 {
                p[c] += n;
            }
        }
        let a = measure(&clean);
        let b = measure(&noisy);
        assert_eq!(b.kind, "photo", "grainy pictures are photos");
        assert!(a.noise < b.noise / 3.0, "clean noise {}", a.noise);
        assert!(b.noise > 5.0, "noisy noise {}", b.noise);
        let r = recommend(&b, Budget::default());
        assert_eq!(r.model, "fast");
        assert!(r.denoise > 0.2);
    }

    #[test]
    fn soft_images_read_soft() {
        let img = scene(512, 384);
        let small = resize(&img, 128, 96, Filter::Box);
        let soft = resize(&small, 512, 384, Filter::CatmullRom);
        // Raw pixel-scale sharpness drops...
        let raw = |im: &Image| sharpness(&luma255(im), im.w, im.h);
        let (a, b) = (raw(&img), raw(&soft));
        assert!(b < a - 0.1, "sharp {a} soft {b}");
        // ...and the analyzer finds the real resolution (4x blown up).
        let m = measure(&soft);
        assert!(m.eff >= 2.5, "eff {}", m.eff);
    }

    #[test]
    fn jpeg_blocks_are_detected() {
        let img = scene(512, 384);
        let mut q = image::RgbImage::new(512, 384);
        for (i, p) in q.pixels_mut().enumerate() {
            let s = img.px[i];
            *p = image::Rgb([(s[0].clamp(0.0, 1.0) * 255.0) as u8, (s[1].clamp(0.0, 1.0) * 255.0) as u8, (s[2].clamp(0.0, 1.0) * 255.0) as u8]);
        }
        let mut bytes = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, 15).encode_image(&q).unwrap();
        let j = Image::decode(&bytes).unwrap();
        let mut hi = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut hi, 97).encode_image(&q).unwrap();
        let k = Image::decode(&hi).unwrap();
        let (bj, bk) = (measure(&j).blocking, measure(&k).blocking);
        assert!(bj > 0.25 && bk < 0.15, "q15 {bj} q97 {bk}");
    }

    #[test]
    fn scale_follows_size_and_cap() {
        let mut m = measure(&scene(64, 64));
        m.kind = "photo";
        m.width = 640;
        m.height = 480;
        assert_eq!(recommend(&m, Budget::default()).scale, 8.0);
        m.width = 1920;
        m.height = 1080;
        assert_eq!(recommend(&m, Budget::default()).scale, 3.0);
        m.width = 5000;
        m.height = 4000;
        assert_eq!(recommend(&m, Budget::default()).scale, 2.0);
        m.width = 1500;
        m.height = 1500;
        assert_eq!(recommend(&m, Budget { max_output_mp: 20.0, ..Budget::default() }).scale, 2.0);
    }

    #[test]
    fn slow_phone_gets_cheaper_model() {
        let mut m = measure(&scene(64, 64));
        m.kind = "photo";
        m.noise = 0.5;
        m.blocking = 0.0;
        m.skin = 0.0;
        m.width = 900;
        m.height = 900;
        let fast = recommend(&m, Budget { gflops: 10_000.0, ..Budget::default() });
        let slow = recommend(&m, Budget { gflops: 5.0, ..Budget::default() });
        assert_eq!(fast.model, "photo");
        assert_eq!(slow.model, "fast");
        assert!(fast.ensemble >= 2);
        assert!(!slow.adaptive);
    }

    #[test]
    fn portraits_go_adaptive_when_it_fits() {
        let mut m = measure(&scene(64, 64));
        m.kind = "photo";
        m.noise = 0.5;
        m.blocking = 0.0;
        m.skin = 0.4;
        m.width = 900;
        m.height = 900;
        let gpu = recommend(&m, Budget { gflops: 2_000.0, ..Budget::default() });
        assert_eq!(gpu.model, "photo");
        assert!(gpu.adaptive && gpu.realism < 1.0);
        let slow = recommend(&m, Budget { gflops: 20.0, ..Budget::default() });
        assert!(!slow.adaptive);
        assert_ne!(slow.model, "photo");
        let cloud = recommend(&m, Budget { cloud: true, ..Budget::default() });
        assert!(cloud.adaptive);
    }
}

#[cfg(test)]
mod real_resolution_tests {
    use super::*;
    use crate::resample::{resize, Filter};

    fn detailed(w: usize, h: usize) -> Image {
        let mut img = Image::new(w, h);
        let mut s = 12345u32;
        for (i, p) in img.px.iter_mut().enumerate() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let n = (s >> 24) as f32 / 255.0;
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            let v = 0.5 + 0.25 * (x * 0.31).sin() * (y * 0.23).cos() + 0.2 * (n - 0.5);
            *p = [v, v * 0.9, v * 0.8, 1.0];
        }
        img
    }

    #[test]
    fn blown_up_images_are_found() {
        let sharp = detailed(300, 300);
        assert_eq!(classify::effective_scale(&sharp), 1.0);
        let small = resize(&sharp, 100, 100, Filter::Lanczos3);
        let blown = resize(&small, 300, 300, Filter::CatmullRom);
        let e = classify::effective_scale(&blown);
        assert!(e >= 2.5, "3x blown-up image read as {e}");
        let m = measure(&blown);
        let r = recommend(&m, Budget { gflops: 2000.0, ..Budget::default() });
        assert!(r.prescale < 0.45, "prescale {}", r.prescale);
    }

    #[test]
    fn fixed_settings_are_kept() {
        let m = measure(&detailed(200, 160));
        let fx = Fixed { model: Some("anime".into()), fidelity: Some("off".into()), ensemble: Some(4), detail: Some(0.9), ..Fixed::default() };
        let r = recommend_fixed(&m, Budget::default(), &fx);
        assert_eq!((r.model.as_str(), r.fidelity.as_str(), r.ensemble), ("anime", "off", 4));
        assert!((r.detail - 0.9).abs() < 1e-6);
        // Auto fields still come from the photo.
        let auto = recommend_fixed(&m, Budget::default(), &Fixed { model: Some("photo".into()), ..Fixed::default() });
        assert_eq!(auto.model, "photo");
        assert!(auto.fidelity != "auto" && auto.ensemble >= 1 && auto.detail >= 0.0);
    }
}
