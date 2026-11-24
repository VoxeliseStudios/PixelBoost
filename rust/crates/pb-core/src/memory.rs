//! Memory budget for big jobs, and decoding that never holds a whole big
//! picture as f32.
//!
//! iOS kills an app outright when it crosses its memory limit (about 2 GB
//! on a 4 GB iPhone): no error and no telemetry. A fixed 120 MP output cap
//! was fine on a 12 GB phone and fatal on a 4 GB one, so the cap now comes
//! from what the process may still allocate ([`available_mb`]) and what the
//! job will need ([`UpscaleMem`]).
//!
//! Previews and thumbnails of results ([`open_preview`]) stream PNG rows
//! into a box grid instead of decoding the full image as f32 (16 B/px: a
//! 120 MP result was 1.9 GB just to draw a 2048 px preview).

use crate::image::Image;
use crate::resample::{resize, Filter};
use std::io::BufReader;

/// Memory the process can still allocate before the OS steps in, in MB.
/// iOS: `os_proc_available_memory` (the distance to the jetsam limit).
/// Linux: MemAvailable. Elsewhere: unknown.
pub fn available_mb() -> Option<f64> {
    if let Some(v) = std::env::var("PB_AVAILABLE_MB").ok().and_then(|v| v.parse::<f64>().ok()) {
        return Some(v);
    }
    #[cfg(any(target_os = "ios", target_os = "tvos"))]
    {
        extern "C" {
            fn os_proc_available_memory() -> usize;
        }
        // SAFETY: plain libSystem query (iOS 13+), no arguments.
        let b = unsafe { os_proc_available_memory() };
        if b > 0 {
            return Some(b as f64 / 1_048_576.0);
        }
        None
    }
    #[cfg(target_os = "linux")]
    {
        let s = std::fs::read_to_string("/proc/meminfo").ok()?;
        let kb = s.lines().find(|l| l.starts_with("MemAvailable:"))?.split_whitespace().nth(1)?.parse::<f64>().ok()?;
        Some(kb / 1024.0)
    }
    #[cfg(not(any(target_os = "ios", target_os = "tvos", target_os = "linux")))]
    {
        None
    }
}

/// Share of the available memory one upscale may plan to use. The rest
/// covers GPU buffers, the app's own images and estimate error.
pub const BUDGET_SHARE: f64 = 0.5;

/// Tiles in flight, network activations, GPU staging, preview copies.
const FIXED_MB: f64 = 160.0;

/// What one upscale allocates, by input size and settings (bytes per pixel
/// from `stream::upscale`: f32 input 16, 8-bit output 4).
#[derive(Debug, Clone, Copy)]
pub struct UpscaleMem {
    pub in_w: usize,
    pub in_h: usize,
    /// Requested scale of the original input.
    pub scale: f32,
    /// Network input share of the source side (prescale), 1 = none.
    pub prescale: f32,
    /// Network scale (4 for every model here).
    pub net_scale: usize,
    pub alpha: bool,
    pub dehalo: bool,
}

impl UpscaleMem {
    fn eff_px(&self) -> f64 {
        let p = if self.prescale > 0.0 && self.prescale < 0.97 { self.prescale as f64 } else { 1.0 };
        (self.in_w * self.in_h) as f64 * p * p
    }

    fn eff_scale(&self) -> f64 {
        let p = if self.prescale > 0.0 && self.prescale < 0.97 { self.prescale as f64 } else { 1.0 };
        self.scale.clamp(1.0, 64.0) as f64 / p
    }

    /// Everything except the output buffer, in MB.
    fn base_mb(&self) -> f64 {
        let in_px = (self.in_w * self.in_h) as f64;
        let eff = self.eff_px();
        let mut b = in_px * 16.0; // decoded input
        if self.dehalo {
            b += in_px * 16.0;
        }
        if eff < in_px {
            b += eff * 16.0; // prescaled copy
        }
        if self.alpha {
            b += eff * 16.0; // colour bled under transparency
        }
        let s = self.net_scale.max(1) as f64;
        if self.eff_scale() > s * 1.5 {
            b += eff * s * s * 4.0; // first-pass result of a two-pass run
        }
        b / 1_048_576.0 + FIXED_MB
    }

    /// Output pixels the job would produce uncapped.
    pub fn out_px(&self) -> f64 {
        let s = self.scale.clamp(1.0, 64.0) as f64;
        (self.in_w * self.in_h) as f64 * s * s
    }

    /// Estimated peak for an output of `out_px` pixels, in MB.
    pub fn peak_mb(&self, out_px: f64) -> f64 {
        self.base_mb() + out_px * 4.0 / 1_048_576.0
    }

    /// Largest output (pixels) that fits in `budget_mb`; never below the
    /// input size (1x always runs).
    pub fn max_out_px(&self, budget_mb: f64) -> usize {
        let room = (budget_mb - self.base_mb()).max(0.0) * 1_048_576.0 / 4.0;
        room.max((self.in_w * self.in_h) as f64) as usize
    }
}

/// Output cap (pixels) for this job: the smaller of `requested_px` and what
/// [`BUDGET_SHARE`] of the available memory allows. Returns the cap and the
/// available MB it was based on (0 = unknown).
pub fn output_cap(job: &UpscaleMem, requested_px: usize) -> (usize, f64) {
    match available_mb() {
        Some(avail) => (requested_px.min(job.max_out_px(avail * BUDGET_SHARE)), avail),
        None => (requested_px, 0.0),
    }
}

/// Width, height and colour type from the header; no pixels decoded.
pub fn header(path: &str) -> Result<(usize, usize, bool), String> {
    let r = image::ImageReader::open(path).map_err(|e| format!("{path}: {e}"))?.with_guessed_format().map_err(|e| e.to_string())?;
    let d = r.into_decoder().map_err(|e| e.to_string())?;
    use image::ImageDecoder;
    let (w, h) = d.dimensions();
    let alpha_type = d.color_type().has_alpha();
    Ok((w as usize, h as usize, alpha_type))
}

/// Size and real transparency (any alpha < 1) with 8-bit memory at most:
/// pictures without an alpha channel are never decoded.
pub fn info(path: &str) -> Result<(usize, usize, bool), String> {
    let (w, h, alpha_type) = header(path)?;
    if !alpha_type {
        return Ok((w, h, false));
    }
    let mut any = false;
    rows_rgba8(path, &mut |_, row| {
        if !any && row.chunks_exact(4).any(|p| p[3] < 255) {
            any = true;
        }
    })?;
    Ok((w, h, any))
}

/// Calls `f(y, rgba_row)` for every row, top to bottom. PNG (all results)
/// is streamed one row at a time; other formats are decoded to 8-bit first.
fn rows_rgba8(path: &str, f: &mut dyn FnMut(usize, &[u8])) -> Result<(usize, usize), String> {
    if let Some(r) = png_rows(path, f) {
        return r;
    }
    let img = crate::strips::open_rgba8(path)?;
    for (y, row) in img.data.chunks_exact(img.w * 4).enumerate() {
        f(y, row);
    }
    Ok((img.w, img.h))
}

/// `None` = not a plain (non-interlaced) PNG: use the general path.
fn png_rows(path: &str, f: &mut dyn FnMut(usize, &[u8])) -> Option<Result<(usize, usize), String>> {
    let file = std::fs::File::open(path).ok()?;
    let mut dec = png::Decoder::new(BufReader::with_capacity(1 << 16, file));
    dec.set_transformations(png::Transformations::normalize_to_color8());
    let mut rd = dec.read_info().ok()?;
    let info = rd.info();
    if info.interlaced || info.is_animated() {
        return None;
    }
    let (w, h) = (info.width as usize, info.height as usize);
    let (ct, _) = rd.output_color_type();
    let mut rgba = vec![0u8; w * 4];
    let mut y = 0;
    loop {
        let row = match rd.next_row() {
            Ok(Some(r)) => r,
            Ok(None) => break,
            Err(e) => return Some(Err(format!("{path}: {e}"))),
        };
        let d = row.data();
        match ct {
            png::ColorType::Rgba => rgba.copy_from_slice(&d[..w * 4]),
            png::ColorType::Rgb => {
                for (o, i) in rgba.chunks_exact_mut(4).zip(d.chunks_exact(3)) {
                    o.copy_from_slice(&[i[0], i[1], i[2], 255]);
                }
            }
            png::ColorType::GrayscaleAlpha => {
                for (o, i) in rgba.chunks_exact_mut(4).zip(d.chunks_exact(2)) {
                    o.copy_from_slice(&[i[0], i[0], i[0], i[1]]);
                }
            }
            png::ColorType::Grayscale => {
                for (o, i) in rgba.chunks_exact_mut(4).zip(d.iter()) {
                    o.copy_from_slice(&[*i, *i, *i, 255]);
                }
            }
            png::ColorType::Indexed => return None,
        }
        f(y, &rgba);
        y += 1;
        if y == h {
            break;
        }
    }
    Some(Ok((w, h)))
}

/// The picture at `path` shrunk to at most `max_side` px on the long side.
/// Big pictures go through an integer box grid (1-2x the target, built row
/// by row), then Lanczos to the exact size, so memory stays near
/// `max_side`² instead of the full picture.
pub fn open_preview(path: &str, max_side: usize) -> Result<Image, String> {
    let m = max_side.max(16);
    let (w, h, _) = header(path)?;
    let fit = |img: Image| {
        let f = m as f32 / img.w.max(img.h) as f32;
        if f < 1.0 {
            resize(&img, ((img.w as f32 * f).round() as usize).max(1), ((img.h as f32 * f).round() as usize).max(1), Filter::Lanczos3)
        } else {
            img
        }
    };
    // Grid between 1x and 2x the target (a 2048 px preview of a 10666 px
    // result: 2134² = 73 MB of sums).
    let k = w.max(h) / m;
    if k < 2 {
        return Image::open(path).map(fit);
    }
    let (gw, gh) = (w.div_ceil(k), h.div_ceil(k));
    let mut acc = vec![[0f32; 4]; gw * gh];
    rows_rgba8(path, &mut |y, row| {
        let g = &mut acc[(y / k) * gw..(y / k + 1) * gw];
        for (x, p) in row.chunks_exact(4).enumerate() {
            let a = &mut g[x / k];
            a[0] += p[0] as f32;
            a[1] += p[1] as f32;
            a[2] += p[2] as f32;
            a[3] += p[3] as f32;
        }
    })?;
    let mut img = Image::new(gw, gh);
    for gy in 0..gh {
        let rows = (h - gy * k).min(k) as f32;
        for gx in 0..gw {
            let cols = (w - gx * k).min(k) as f32;
            let n = 255.0 * rows * cols;
            let a = acc[gy * gw + gx];
            img.px[gy * gw + gx] = [a[0] / n, a[1] / n, a[2] / n, a[3] / n];
        }
    }
    drop(acc);
    Ok(fit(img))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> String {
        std::env::temp_dir().join(format!("pb-mem-{}-{name}", std::process::id())).to_string_lossy().into()
    }

    fn gradient(w: usize, h: usize, alpha: bool) -> crate::stream::Rgba8 {
        let mut r = crate::stream::Rgba8::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                r.data[i] = (x * 255 / w) as u8;
                r.data[i + 1] = (y * 255 / h) as u8;
                r.data[i + 2] = 128;
                r.data[i + 3] = if alpha && x < w / 4 { 0 } else { 255 };
            }
        }
        r
    }

    #[test]
    fn preview_matches_full_decode() {
        let p = tmp("prev.png");
        gradient(3000, 1700, false).save(&p, 95).unwrap();
        let a = open_preview(&p, 256).unwrap();
        let full = Image::open(&p).unwrap();
        let b = resize(&full, a.w, a.h, Filter::Lanczos3);
        assert_eq!((a.w, a.h), (256, 145));
        let err = a.px.iter().zip(&b.px).map(|(x, y)| (0..3).map(|c| (x[c] - y[c]).abs()).fold(0f32, f32::max)).fold(0f32, f32::max);
        assert!(err < 0.02, "max diff {err}");
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn info_reads_alpha_without_full_decode() {
        let p = tmp("alpha.png");
        gradient(400, 300, true).save(&p, 95).unwrap();
        assert_eq!(info(&p).unwrap(), (400, 300, true));
        let q = tmp("opaque.png");
        gradient(400, 300, false).save(&q, 95).unwrap();
        assert_eq!(info(&q).unwrap(), (400, 300, false));
        std::fs::remove_file(&p).ok();
        std::fs::remove_file(&q).ok();
    }

    #[test]
    fn cap_follows_available_memory() {
        // The crash case: 4096² at 4x on a 4 GB iPhone (~1.7 GB left).
        let job = UpscaleMem { in_w: 4096, in_h: 4096, scale: 4.0, prescale: 1.0, net_scale: 4, alpha: false, dehalo: false };
        let cap = job.max_out_px(1700.0 * BUDGET_SHARE);
        assert!(cap < 120_000_000 && cap > 40_000_000, "cap {cap}");
        assert!(job.peak_mb(cap as f64) <= 1700.0 * BUDGET_SHARE + 1.0);
        // A 12 GB phone keeps the full 120 MP.
        assert!(job.max_out_px(9000.0 * BUDGET_SHARE) >= 120_000_000);
        // Never below 1x.
        assert_eq!(job.max_out_px(10.0), 4096 * 4096);
    }
}
