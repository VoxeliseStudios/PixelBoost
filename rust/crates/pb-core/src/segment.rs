//! Subject cutout with PixelBoost's own segmentation networks (replaces
//! the system's subject lift).
//!
//! - **Fast** (bundled): U²-Net-p, 1.1 M parameters, 320 px input.
//! - **HQ** (download): IS-Net general use, 44 M parameters,
//!   1024 px input — much finer hair and holes.
//!
//! The network sees a square copy; its saliency map is stretched to 0..1,
//! scaled back up and refined with a guided filter on the full-resolution
//! photo, so the matte edge follows the real edges (hair, fur) instead of
//! the network's coarse grid. A soft contrast curve turns saliency into
//! alpha.

use crate::filters::box_blur_plane;
use crate::image::Image;
use crate::resample::{resize, Filter};
use pb_nn::fmap::Fmap;
use pb_nn::seg::SegNet;
use rayon::prelude::*;
use std::path::Path;
use std::sync::{Arc, Mutex};

pub const FAST_FILE: &str = "cutout-u2netp.pbnn";
pub const HQ_FILE: &str = "cutout-isnet.pbnn";

static CACHE: Mutex<Option<(String, Arc<SegNet>)>> = Mutex::new(None);

/// Load (and keep one warm) segmentation network.
pub fn load(path: &Path) -> Result<Arc<SegNet>, String> {
    let key = path.to_string_lossy().to_string();
    let mut g = CACHE.lock().unwrap();
    if let Some((k, n)) = g.as_ref() {
        if *k == key {
            return Ok(n.clone());
        }
    }
    *g = None;
    let b = std::fs::read(path).map_err(|e| format!("{key}: {e}"))?;
    let n = Arc::new(SegNet::from_weights(&pb_nn::pbm::read(&b)?)?);
    *g = Some((key, n.clone()));
    Ok(n)
}

/// Free the cached network (memory pressure).
pub fn unload() {
    *CACHE.lock().unwrap() = None;
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Network input side (0 = the model's own size).
    pub size: usize,
    /// Guided-filter refinement strength, 0..1 (0 = network edge only).
    pub refine: f32,
}

impl Default for Options {
    fn default() -> Self {
        Options { size: 0, refine: 1.0 }
    }
}

/// Alpha matte (0..1, one value per pixel) of the main subject.
pub fn subject_mask(net: &SegNet, img: &Image, opt: Options) -> Vec<f32> {
    let t0 = std::time::Instant::now();
    let n = if opt.size >= 64 { opt.size / 32 * 32 } else { net.size };
    let small = resize(img, n, n, Filter::Lanczos3);
    // Normalisation the networks were trained with.
    let (mean, std) = if net.variant == "isnet" { ([0.5f32, 0.5, 0.5], [1.0f32, 1.0, 1.0]) } else { ([0.485f32, 0.456, 0.406], [0.229f32, 0.224, 0.225]) };
    let mx = small.px.iter().fold(1e-6f32, |m, p| m.max(p[0]).max(p[1]).max(p[2]));
    let mut x = Fmap::new(n, n, 3);
    for y in 0..n {
        for xx in 0..n {
            let p = small.px[y * n + xx];
            // Transparent pixels count as background.
            let a = p[3];
            let d = x.px_mut(y, xx);
            for c in 0..3 {
                d[c] = ((p[c] * a + (1.0 - a)) / mx - mean[c]) / std[c];
            }
        }
    }
    let t_pre = t0.elapsed().as_secs_f64() * 1000.0;
    let sal = net.forward(&x);
    let t_net = t0.elapsed().as_secs_f64() * 1000.0 - t_pre;
    let (lo, hi) = sal.iter().fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
    let span = (hi - lo).max(1e-6);
    let mut m = Image::new(n, n);
    for (o, v) in m.px.iter_mut().zip(&sal) {
        let s = (v - lo) / span;
        *o = [s, s, s, 1.0];
    }
    let up = resize(&m, img.w, img.h, Filter::CatmullRom);
    let coarse: Vec<f32> = up.px.par_iter().map(|p| p[0].clamp(0.0, 1.0)).collect();
    let mut mask = if opt.refine > 0.0 { refine(img, &coarse, opt.refine) } else { coarse };
    // Saliency -> alpha: keep soft edges, push the inside / outside solid.
    mask.par_iter_mut().for_each(|v| *v = crate::filters::smoothstep(0.08, 0.92, *v));
    report(net, img, n, &mask, t_pre, t_net, t0.elapsed().as_secs_f64() * 1000.0);
    mask
}

/// Guided filter (He et al.) of the coarse mask with the photo's luma as
/// guide, radius scaled to the picture; mixes back by `k`.
fn refine(img: &Image, m: &[f32], k: f32) -> Vec<f32> {
    let (w, h) = (img.w, img.h);
    let r = ((w.min(h) as f32 / 180.0).round() as usize).clamp(2, 24);
    let eps = 2e-3f32;
    let i: Vec<f32> = img.px.par_iter().map(|p| Image::luma(*p)).collect();
    let mean = |v: &[f32]| box_blur_plane(v, w, h, r);
    let mi = mean(&i);
    let mp = mean(m);
    let ii: Vec<f32> = i.par_iter().map(|v| v * v).collect();
    let ip: Vec<f32> = i.par_iter().zip(m).map(|(a, b)| a * b).collect();
    let mii = mean(&ii);
    let mip = mean(&ip);
    let a: Vec<f32> = (0..w * h).into_par_iter().map(|j| (mip[j] - mi[j] * mp[j]) / (mii[j] - mi[j] * mi[j] + eps)).collect();
    let b: Vec<f32> = (0..w * h).into_par_iter().map(|j| mp[j] - a[j] * mi[j]).collect();
    let ma = mean(&a);
    let mb = mean(&b);
    (0..w * h).into_par_iter().map(|j| {
        let q = (ma[j] * i[j] + mb[j]).clamp(0.0, 1.0);
        // Only refine near the edge: far inside / outside keep the net.
        let edge = 1.0 - (2.0 * m[j] - 1.0).abs();
        let t = (k * (0.3 + 0.7 * edge)).clamp(0.0, 1.0);
        m[j] + (q - m[j]) * t
    })
    .collect()
}

/// Telemetry: model, sizes, timings, how much of the picture is subject,
/// how much of the matte is undecided (soft) — the share people end up
/// fixing with the brush shows up as feedback "mask_fix".
fn report(net: &SegNet, img: &Image, n: usize, mask: &[f32], t_pre: f64, t_net: f64, ms: f64) {
    let e = crate::telemetry::event("cutout", "mask");
    if !e.active() {
        return;
    }
    let len = mask.len().max(1) as f64;
    let cover = mask.iter().map(|v| *v as f64).sum::<f64>() / len;
    let soft = mask.iter().filter(|v| **v > 0.1 && **v < 0.9).count() as f64 / len;
    // Fragmentation proxy: subject pixels on the image border.
    let (w, h) = (img.w, img.h);
    let border = (0..w).map(|x| mask[x] + mask[(h - 1) * w + x]).chain((0..h).map(|y| mask[y * w] + mask[y * w + w - 1])).map(|v| v as f64).sum::<f64>() / (2 * (w + h)) as f64;
    let (_, peak) = crate::telemetry::memory_mb();
    e.ms(ms)
        .tag("model", &net.variant)
        .tag("img", crate::telemetry::fingerprint(img))
        .metric("net_size", n as f64)
        .metric("mp", (w * h) as f64 / 1e6)
        .metric("ms_pre", t_pre)
        .metric("ms_net", t_net)
        .metric("coverage", cover)
        .metric("uncertain", soft)
        .metric("border_subject", border)
        .metric("mem_peak_mb", peak)
        .emit();
}

/// Write the mask as an 8-bit grey PNG (what the editor's mask ops read).
pub fn save_mask(mask: &[f32], w: usize, h: usize, path: &str) -> Result<(), String> {
    let mut img = Image::new(w, h);
    img.px.par_iter_mut().zip(mask.par_iter()).for_each(|(p, v)| *p = [*v, *v, *v, 1.0]);
    img.save(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_model_finds_a_subject() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../Models/pbnn").join(FAST_FILE);
        let Ok(net) = load(&path) else {
            eprintln!("{FAST_FILE} not found: skipping");
            return;
        };
        // A bright disc on a dark textured background.
        let (w, h) = (240, 180);
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = (x as f32 - 120.0, y as f32 - 95.0);
                let inside = dx * dx + dy * dy < 55.0 * 55.0;
                let n = crate::filters::hash_noise(x as u32, y as u32, 1) * 0.05;
                img.px[y * w + x] = if inside { [0.95, 0.55 + n, 0.2, 1.0] } else { [0.15 + n, 0.2 + n, 0.25 + n, 1.0] };
            }
        }
        let m = subject_mask(&net, &img, Options::default());
        let centre = m[95 * w + 120];
        let corner = m[5 * w + 5];
        assert!(centre > 0.8 && corner < 0.2, "centre {centre} corner {corner}");
    }
}
