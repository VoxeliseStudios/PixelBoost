//! Picks a model for "Auto" from simple, fast image statistics.
//!
//! Drawn art (anime, cel shading, line art) has large flat fills separated
//! by hard edges and almost no sensor-like noise; photos and rendered 3D
//! scenes have texture almost everywhere.

use crate::image::Image;
use crate::resample::{resize, Filter};
use serde::Serialize;

#[derive(Serialize, Debug, Clone, Copy)]
pub struct Stats {
    /// Share of pixels whose 3x3 neighbourhood is essentially one colour.
    pub flat: f32,
    /// Share of pixels on a hard edge.
    pub edges: f32,
    /// Median absolute Laplacian on non-edge pixels (noise / fine texture).
    pub texture: f32,
}

pub fn stats(img: &Image) -> Stats {
    let f = 512.0 / img.w.max(img.h) as f32;
    let small = if f < 1.0 { resize(img, ((img.w as f32 * f) as usize).max(8), ((img.h as f32 * f) as usize).max(8), Filter::Box) } else { img.clone() };
    let (w, h) = (small.w, small.h);
    let l: Vec<f32> = small.px.iter().map(|p| Image::luma(*p)).collect();
    let (mut flat, mut edges, mut n) = (0usize, 0usize, 0usize);
    let mut tex = Vec::new();
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let c = l[y * w + x];
            let mut lo = c;
            let mut hi = c;
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let v = l[((y as isize + dy) as usize) * w + (x as isize + dx) as usize];
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            let range = hi - lo;
            n += 1;
            if range < 0.02 {
                flat += 1;
            }
            if range > 0.25 {
                edges += 1;
            } else {
                let lap = 4.0 * c - l[y * w + x - 1] - l[y * w + x + 1] - l[(y - 1) * w + x] - l[(y + 1) * w + x];
                tex.push(lap.abs());
            }
        }
    }
    tex.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = n.max(1) as f32;
    Stats { flat: flat as f32 / n, edges: edges as f32 / n, texture: tex.get(tex.len() / 2).copied().unwrap_or(0.0) }
}

/// Content type only: "anime" (drawn) or "photo" (photo / render),
/// judged at the image's real resolution.
pub fn content_kind(img: &Image) -> &'static str {
    let (cw, ch) = (img.w.min(768), img.h.min(768));
    let crop = img.crop((img.w - cw) / 2, (img.h - ch) / 2, cw, ch);
    let eff = effective_scale(&crop);
    kind_of_ink(&ink(&at_real_resolution(img, eff, img.w, 512)), eff)
}

pub fn kind_of(s: &Stats) -> &'static str {
    if (s.flat > 0.35 && s.texture < 0.013) || s.flat > 0.55 {
        "anime"
    } else {
        "photo"
    }
}

/// Model for Auto: the best model for the content whose cost fits a
/// compute budget (phones run these on the CPU). The heavy 23-block photo
/// network is used up to ~1 MP; above that the compact general model,
/// which scores as well on the bench at a fifteenth of the cost.
pub fn pick_model(img: &Image) -> &'static str {
    let mp = (img.w * img.h) as f32 / 1e6;
    match content_kind(img) {
        "anime" if mp <= 4.0 => "anime",
        "anime" => "lineart",
        _ if mp <= 1.0 => "photo",
        _ => "fast",
    }
}

/// How much coarser the image's real detail is than its pixel grid: 1 =
/// full detail, 4 = it was blown up 4x (a meme re-saved from a phone
/// screenshot, a video frame scaled up). Found by shrinking a native crop
/// by growing factors and scaling it back: while that loses less than
/// ~1.2 levels (8-bit RMS), the pixels carry nothing finer.
pub fn effective_scale(crop: &Image) -> f32 {
    if crop.w < 48 || crop.h < 48 {
        return 1.0;
    }
    let base: Vec<f32> = crop.px.iter().map(|p| Image::luma(*p) * 255.0).collect();
    let (w, h) = (crop.w, crop.h);
    let mut best = 1.0;
    // Telemetry: the loss curve the 1.2-level threshold is applied to.
    let mut ev = crate::telemetry::event("classify", "real_resolution");
    for s in [1.25f32, 1.5, 2.0, 2.5, 3.0, 4.0] {
        let (sw, sh) = (((w as f32 / s).round() as usize).max(8), ((h as f32 / s).round() as usize).max(8));
        let back = resize(&resize(crop, sw, sh, Filter::Lanczos3), w, h, Filter::CatmullRom);
        let (mut se, mut n) = (0.0f64, 0usize);
        for y in 4..h - 4 {
            for x in 4..w - 4 {
                let d = (base[y * w + x] - Image::luma(back.px[y * w + x]) * 255.0) as f64;
                se += d * d;
                n += 1;
            }
        }
        let rms = (se / n.max(1) as f64).sqrt();
        ev = ev.metric(&format!("rms@{s}"), rms);
        if rms < 1.2 {
            best = s;
        } else {
            break;
        }
    }
    ev.metric("eff", best).emit();
    best
}

/// Evidence of drawn art, measured at the image's real resolution on a
/// copy of at most 512 px.
#[derive(Serialize, Debug, Clone, Copy, Default)]
pub struct Ink {
    /// Share of pixels on thin dark lines (darker than both neighbours two
    /// pixels away, horizontally or vertically, by > 0.12).
    pub line: f32,
    /// Share of non-edge pixels in the 24 most common colours (5 bits per
    /// channel): fills drawn with a palette concentrate here.
    pub palette: f32,
    /// Share of pixel-flat 3x3 neighbourhoods (range < 1.5/255).
    pub flat0: f32,
}

pub fn ink(img: &Image) -> Ink {
    let (w, h) = (img.w, img.h);
    if w < 8 || h < 8 {
        return Ink::default();
    }
    let l: Vec<f32> = img.px.iter().map(|p| Image::luma(*p)).collect();
    let at = |x: isize, y: isize| l[(y.clamp(0, h as isize - 1) as usize) * w + x.clamp(0, w as isize - 1) as usize];
    let (mut flat0, mut n, mut smooth) = (0usize, 0usize, 0usize);
    let mut hist = vec![0u32; 32768];
    // Darker than both neighbours two pixels away: across a vertical line
    // (vx) or a horizontal one (hy).
    // Grain lifts the bar: a line must stand out of the noise (Immerkær's
    // estimate on this copy) as well as be dark.
    let mut acc = 0.0f64;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let (xi, yi) = (x as isize, y as isize);
            let v = at(xi - 1, yi - 1) + at(xi + 1, yi - 1) + at(xi - 1, yi + 1) + at(xi + 1, yi + 1) - 2.0 * (at(xi, yi - 1) + at(xi, yi + 1) + at(xi - 1, yi) + at(xi + 1, yi)) + 4.0 * at(xi, yi);
            acc += v.abs() as f64;
        }
    }
    let sigma = (acc * (std::f64::consts::PI / 2.0).sqrt() / (6.0 * ((w - 2) * (h - 2)) as f64)) as f32;
    let dark = 0.12 + 3.0 * sigma;
    let mut vx = vec![false; w * h];
    let mut hy = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let (xi, yi) = (x as isize, y as isize);
            let c = at(xi, yi);
            vx[y * w + x] = at(xi - 2, yi).min(at(xi + 2, yi)) - c > dark;
            hy[y * w + x] = at(xi, yi - 2).min(at(xi, yi + 2)) - c > dark;
        }
    }
    let mut line = 0usize;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let (xi, yi) = (x as isize, y as isize);
            let c = at(xi, yi);
            let (mut lo, mut hi) = (c, c);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let v = at(xi + dx, yi + dy);
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            let range = hi - lo;
            n += 1;
            if range < 0.006 {
                flat0 += 1;
            }
            if range < 0.12 {
                let p = img.px[y * w + x];
                let q = |v: f32| (v.clamp(0.0, 1.0) * 31.999) as usize;
                hist[(q(p[0]) << 10) | (q(p[1]) << 5) | q(p[2])] += 1;
                smooth += 1;
            }
            // Ink is continuous: a line pixel has a line neighbour along the
            // line (isolated dark noise pixels don't count).
            let i = y * w + x;
            let v_ok = vx[i] && (vx[i - w] || vx[i + w] || vx[i - w - 1] || vx[i - w + 1] || vx[i + w - 1] || vx[i + w + 1]);
            let h_ok = hy[i] && (hy[i - 1] || hy[i + 1] || hy[i - w - 1] || hy[i + w - 1] || hy[i - w + 1] || hy[i + w + 1]);
            if v_ok || h_ok {
                line += 1;
            }
        }
    }
    hist.sort_unstable_by(|a, b| b.cmp(a));
    let top: u32 = hist[..24].iter().sum();
    let n = n.max(1) as f32;
    Ink { line: line as f32 / n, palette: top as f32 / smooth.max(1) as f32, flat0: flat0 as f32 / n }
}

/// Drawn art needs positive evidence: ink lines over a concentrated
/// palette, or mostly pixel-flat fills. Everything else is treated as a
/// photo / render — the photo networks handle a cartoon well, while the
/// anime network turns a real face into a painting. Thresholds from 38
/// labelled images (gallery art + photo / cartoon memes): no photo is
/// called drawn.
pub fn kind_of_ink(i: &Ink, eff: f32) -> &'static str {
    // At full detail (not blown up), pixel-flat fills over a tight palette
    // are drawn too: real photos at native resolution carry sensor grain.
    let sharp_fills = eff <= 1.25 && i.flat0 > 0.25 && i.palette > 0.6;
    if (i.line > 0.0145 && i.palette > 0.3) || (i.flat0 > 0.55 && i.palette > 0.85) || sharp_fills {
        "anime"
    } else {
        "photo"
    }
}

/// `img` shrunk to its real resolution (`eff`), then to at most `max_side`.
pub fn at_real_resolution(img: &Image, eff: f32, full_w: usize, max_side: usize) -> Image {
    // `img` may already be a reduced copy of a `full_w`-wide original.
    let have = img.w as f32 / full_w.max(1) as f32;
    let want = (1.0 / eff.max(1.0)).min(max_side as f32 / full_w.max(img.h * full_w / img.w.max(1)) as f32);
    let f = want / have;
    if f >= 0.999 {
        return img.clone();
    }
    let (w, h) = (((img.w as f32 * f).round() as usize).max(8), ((img.h as f32 * f).round() as usize).max(8));
    resize(img, w, h, if eff > 1.0 { Filter::Lanczos3 } else { Filter::Box })
}
