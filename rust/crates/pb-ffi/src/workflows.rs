//! Batch edits, model comparison and the pixel loupe.

use crate::{to_rgba, CancelToken, PbError, PixelBoostCore, RgbaImage};
use pb_core::fidelity::{fidelity_db, Fidelity};
use pb_core::pipeline::Settings;
use pb_core::{metrics, ops, Image};
use std::sync::Arc;

// ------------------------------------------------------------ batch edits

#[uniffi::export(with_foreign)]
pub trait BatchListener: Send + Sync {
    /// `index` finished (`error` empty on success).
    fn on_item(&self, index: u32, total: u32, output_path: String, error: String);
}

#[derive(uniffi::Record, Clone)]
pub struct BatchSummary {
    pub done: u32,
    pub failed: u32,
    pub seconds: f64,
    pub run_id: String,
}

// ------------------------------------------------------------ compare

#[derive(uniffi::Record, Clone)]
pub struct CompareCandidate {
    /// e.g. "photo", "photo:r0.5", "photo+adaptive", "clean", "anime".
    pub id: String,
    pub label: String,
    pub image: RgbaImage,
    pub ms: f64,
    /// Agreement with the source when shrunk back (dB; higher = faithful).
    pub fidelity_db: f64,
    /// Fine-detail level relative to a plain resize (1 = same as bicubic).
    pub detail: f64,
}

#[derive(uniffi::Record, Clone)]
pub struct CompareResult {
    pub run_id: String,
    /// The source crop (x, y, w, h in source pixels) that was compared.
    pub crop: Vec<u32>,
    pub candidates: Vec<CompareCandidate>,
    /// What the analyzer would pick for this photo.
    pub auto_pick: String,
}

#[uniffi::export(with_foreign)]
pub trait CompareListener: Send + Sync {
    fn on_candidate(&self, done: u32, total: u32, id: String);
}

#[uniffi::export]
impl PixelBoostCore {
    /// Apply one edit recipe (op JSON: a single op or a chain, e.g. a
    /// preset or copied edits) to many photos, each written to the matching
    /// output path. Files are independent: one failure doesn't stop the rest.
    pub fn batch_edit(&self, inputs: Vec<String>, outputs: Vec<String>, op_json: String, listener: Arc<dyn BatchListener>, cancel: Arc<CancelToken>) -> Result<BatchSummary, PbError> {
        if inputs.len() != outputs.len() {
            return Err(PbError::from("inputs and outputs differ in length".to_string()));
        }
        let op = ops::parse(&op_json)?;
        if op.uses_files() {
            return Err(PbError::from("this edit uses masks or files of one photo; it can't be applied to others".to_string()));
        }
        let run = pb_core::telemetry::new_run_id();
        let t0 = std::time::Instant::now();
        let (mut done, mut failed) = (0u32, 0u32);
        let total = inputs.len() as u32;
        pb_core::telemetry::scope(&run, || {
            for (i, (inp, out)) in inputs.iter().zip(&outputs).enumerate() {
                if cancel.is_cancelled() {
                    break;
                }
                match pb_core::strips::edit_file(inp, out, &op, 95) {
                    Ok(_) => {
                        done += 1;
                        listener.on_item(i as u32, total, out.clone(), String::new());
                    }
                    Err(e) => {
                        failed += 1;
                        listener.on_item(i as u32, total, out.clone(), e);
                    }
                }
            }
        });
        let seconds = t0.elapsed().as_secs_f64();
        pb_core::telemetry::event("batch", "edit")
            .run(&run)
            .ms(seconds * 1000.0)
            .tag("op", ops::op_name(&op))
            .metric("files", total as f64)
            .metric("done", done as f64)
            .metric("failed", failed as f64)
            .flag("cancelled", cancel.is_cancelled())
            .emit();
        Ok(BatchSummary { done, failed, seconds, run_id: run })
    }

    /// Run a crop of the photo through several models / variants so they
    /// can be compared side by side. `crop` = [x, y, w, h] in source pixels
    /// (empty = the most detailed `size` x `size` window). Candidates that
    /// aren't installed are skipped. Results are `scale`x the crop.
    pub fn compare_models(&self, input_path: String, crop: Vec<u32>, size: u32, candidates: Vec<String>, scale: f32, listener: Arc<dyn CompareListener>, cancel: Arc<CancelToken>) -> Result<CompareResult, PbError> {
        let run = pb_core::telemetry::new_run_id();
        let img = Image::open(&input_path)?;
        let n = (size.clamp(48, 384) as usize).min(img.w).min(img.h);
        let (cx, cy, cw, ch) = if crop.len() == 4 {
            let (x, y) = ((crop[0] as usize).min(img.w - 1), (crop[1] as usize).min(img.h - 1));
            (x, y, (crop[2] as usize).clamp(8, img.w - x), (crop[3] as usize).clamp(8, img.h - y))
        } else {
            let (x, y) = most_detailed(&img, n);
            (x, y, n, n)
        };
        let src = img.crop(cx, cy, cw, ch);
        let scale = scale.clamp(2.0, 4.0);
        let auto_pick = pb_core::classify::pick_model(&img).to_string();
        drop(img);
        let base = pb_core::resample::resize(&src, (cw as f32 * scale) as usize, (ch as f32 * scale) as usize, pb_core::resample::Filter::CatmullRom);
        let base_detail = metrics::gradient_energy(&base).max(1e-9);
        let total = candidates.len() as u32;
        let mut out = Vec::new();
        pb_core::telemetry::scope(&run, || -> Result<(), PbError> {
            for (k, id) in candidates.iter().enumerate() {
                if cancel.is_cancelled() {
                    return Err(PbError::Cancelled);
                }
                let (model, realism, adaptive) = parse_candidate(id);
                if !pb_core::model_available(&self.models_dir, model) {
                    continue;
                }
                let t0 = std::time::Instant::now();
                let stack = pb_core::fusion::Stack::load(&self.models_dir, model, None, realism, adaptive)?;
                let st = Settings { scale, fidelity: Fidelity::Natural, detail: 0.35, tile: 192, pad: 16, ensemble: 1, realism, adaptive, prescale: 1.0, dehalo: 0.0 };
                let (o, _rep, _) = pb_core::stream::upscale(&stack, &src, &st, pb_core::stream::Limits::default(), &mut |_| !cancel.is_cancelled()).ok_or(PbError::Cancelled)?;
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                let oi = Image { w: o.w, h: o.h, px: o.data.chunks_exact(4).map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0, 1.0]).collect() };
                let s = (oi.w / src.w).max(1);
                let fid = if oi.w == src.w * s && oi.h == src.h * s { fidelity_db(&src, &oi, s) } else { 0.0 };
                let detail = metrics::gradient_energy(&oi) / base_detail;
                pb_core::telemetry::event("compare", "candidate").ms(ms).tag("candidate", id).metric("fidelity_db", fid).metric("detail", detail).metric("crop_px", (cw * ch) as f64).emit();
                out.push(CompareCandidate { id: id.clone(), label: label_of(id), image: RgbaImage { width: o.w as u32, height: o.h as u32, rgba: o.data }, ms, fidelity_db: fid, detail });
                listener.on_candidate(k as u32 + 1, total, id.clone());
            }
            Ok(())
        })?;
        pb_core::telemetry::event("compare", "sweep").run(&run).tag("auto_pick", &auto_pick).tag("candidates", candidates.join(",")).metric("count", out.len() as f64).emit();
        Ok(CompareResult { run_id: run, crop: vec![cx as u32, cy as u32, cw as u32, ch as u32], candidates: out, auto_pick })
    }

    /// Pixel loupe: `w` x `h` source pixels at (x, y), each drawn as a
    /// `zoom` x `zoom` block (nearest neighbour, so pixels stay pixels).
    pub fn loupe(&self, path: String, x: u32, y: u32, w: u32, h: u32, zoom: u32) -> Result<RgbaImage, PbError> {
        let img = pb_core::strips::open_rgba8(&path)?;
        let z = zoom.clamp(1, 32) as usize;
        let (w, h) = ((w as usize).clamp(1, 256), (h as usize).clamp(1, 256));
        let (x0, y0) = ((x as usize).min(img.w.saturating_sub(1)), (y as usize).min(img.h.saturating_sub(1)));
        let mut out = vec![0u8; w * z * h * z * 4];
        for oy in 0..h * z {
            let sy = y0 + oy / z;
            for ox in 0..w * z {
                let sx = x0 + ox / z;
                let d = (oy * w * z + ox) * 4;
                if sx < img.w && sy < img.h {
                    let s = (sy * img.w + sx) * 4;
                    out[d..d + 4].copy_from_slice(&img.data[s..s + 4]);
                }
            }
        }
        Ok(RgbaImage { width: (w * z) as u32, height: (h * z) as u32, rgba: out })
    }

    /// Small preview of a result for the compare grid (fits `max_side`).
    pub fn preview_rgba(&self, image: RgbaImage, max_side: u32) -> Result<RgbaImage, PbError> {
        let img = crate::from_rgba(&image)?;
        let f = max_side.max(16) as f32 / img.w.max(img.h) as f32;
        if f >= 1.0 {
            return Ok(image);
        }
        Ok(to_rgba(&pb_core::resample::resize(&img, ((img.w as f32 * f) as usize).max(1), ((img.h as f32 * f) as usize).max(1), pb_core::resample::Filter::Lanczos3)))
    }
}

/// "photo", "photo:r0.5", "photo+adaptive" -> (model, realism, adaptive).
fn parse_candidate(id: &str) -> (&str, f32, bool) {
    if let Some(m) = id.strip_suffix("+adaptive") {
        return (m, 1.0, true);
    }
    if let Some((m, r)) = id.split_once(":r") {
        return (m, r.parse().unwrap_or(1.0), false);
    }
    (id, 1.0, false)
}

fn label_of(id: &str) -> String {
    let (m, r, a) = parse_candidate(id);
    let name = pb_core::models::find(m).map(|i| i.name).unwrap_or(m);
    if a {
        format!("{name} · Adaptive")
    } else if r < 0.999 {
        format!("{name} · Realism {}%", (r * 100.0).round())
    } else {
        name.to_string()
    }
}

fn most_detailed(img: &Image, n: usize) -> (usize, usize) {
    let (mut best, mut bx, mut by) = (-1.0, 0, 0);
    let step = (n / 2).max(16);
    let mut y = 0;
    while y + n <= img.h {
        let mut x = 0;
        while x + n <= img.w {
            let g = metrics::gradient_energy(&img.crop(x, y, n, n));
            if g > best {
                (best, bx, by) = (g, x, y);
            }
            x += step;
        }
        y += step;
    }
    (bx, by)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidates_parse() {
        assert_eq!(parse_candidate("photo:r0.5"), ("photo", 0.5, false));
        assert_eq!(parse_candidate("photo+adaptive"), ("photo", 1.0, true));
        assert_eq!(parse_candidate("anime"), ("anime", 1.0, false));
        assert!(label_of("photo:r0.5").contains("50%"));
    }
}
