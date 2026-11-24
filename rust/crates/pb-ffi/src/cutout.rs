//! Subject cutout for the app (Rust networks, see pb_core::segment).

use crate::{PbError, PixelBoostCore};
use pb_core::segment;
use pb_core::Image;

#[derive(uniffi::Record, Clone)]
pub struct SubjectMaskResult {
    /// Grey PNG mask, same size as the photo.
    pub path: String,
    /// "u2netp" (fast, bundled) or "isnet" (HQ download).
    pub model: String,
    /// Share of the picture that is subject, 0..1.
    pub coverage: f32,
    /// Telemetry run (send "mask_fix" feedback with it when the user
    /// repaints the mask).
    pub run_id: String,
    pub ms: f64,
}

/// Keep only the connected subject region under (px, py) (pixels), if any.
fn keep_component(mask: &mut [f32], w: usize, h: usize, px: usize, py: usize) -> bool {
    let at = |x: usize, y: usize| mask[y * w + x] >= 0.5;
    if !at(px, py) {
        return false;
    }
    let mut keep = vec![false; w * h];
    let mut stack = vec![(px, py)];
    keep[py * w + px] = true;
    while let Some((x, y)) = stack.pop() {
        for (dx, dy) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
            let (nx, ny) = (x as isize + dx, y as isize + dy);
            if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                continue;
            }
            let (nx, ny) = (nx as usize, ny as usize);
            if !keep[ny * w + nx] && at(nx, ny) {
                keep[ny * w + nx] = true;
                stack.push((nx, ny));
            }
        }
    }
    // Grow the kept region a little so its soft edge survives.
    let grown = pb_core::filters::dilate_plane(&keep.iter().map(|k| if *k { 1.0 } else { 0.0 }).collect::<Vec<f32>>(), w, h, (w.max(h) / 150).max(2));
    for (m, g) in mask.iter_mut().zip(grown) {
        if g < 0.5 {
            *m = 0.0;
        }
    }
    true
}

#[uniffi::export]
impl PixelBoostCore {
    /// Mask of the main subject. `hq` uses IS-Net when it is downloaded
    /// (else the bundled net). `point_x/y` (0..1, top-left origin; < 0 =
    /// none) keeps only the subject under that point.
    pub fn subject_mask(&self, input_path: String, output_path: String, hq: bool, point_x: f32, point_y: f32) -> Result<SubjectMaskResult, PbError> {
        let run = pb_core::telemetry::new_run_id();
        pb_core::telemetry::scope(&run, || {
            let t0 = std::time::Instant::now();
            let file = if hq && pb_core::locate(&self.models_dir, segment::HQ_FILE).is_some() { segment::HQ_FILE } else { segment::FAST_FILE };
            let path = pb_core::locate(&self.models_dir, file).ok_or_else(|| PbError::from(format!("{file} is missing")))?;
            let net = segment::load(&path)?;
            let img = Image::open(&input_path)?;
            let mut m = segment::subject_mask(&net, &img, segment::Options::default());
            if point_x >= 0.0 && point_y >= 0.0 {
                let px = ((point_x * img.w as f32) as usize).min(img.w - 1);
                let py = ((point_y * img.h as f32) as usize).min(img.h - 1);
                if !keep_component(&mut m, img.w, img.h, px, py) {
                    pb_core::telemetry::event("cutout", "pick_missed").emit();
                    return Err(PbError::from("No subject found there.".to_string()));
                }
            }
            let coverage = m.iter().sum::<f32>() / m.len().max(1) as f32;
            if coverage < 0.002 {
                return Err(PbError::from("No subject found.".to_string()));
            }
            segment::save_mask(&m, img.w, img.h, &output_path)?;
            if hq && file == segment::FAST_FILE {
                pb_core::telemetry::event("cutout", "hq_unavailable").emit();
            }
            Ok(SubjectMaskResult { path: output_path, model: net.variant.clone(), coverage, run_id: run.clone(), ms: t0.elapsed().as_secs_f64() * 1000.0 })
        })
    }

    /// Free the cutout network (memory warning).
    pub fn release_cutout(&self) {
        segment::unload();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_pick() {
        let (w, h) = (20, 10);
        let mut m = vec![0.0f32; w * h];
        for y in 2..8 {
            for x in 1..6 {
                m[y * w + x] = 1.0;
            }
            for x in 12..18 {
                m[y * w + x] = 1.0;
            }
        }
        assert!(!keep_component(&mut m.clone(), w, h, 9, 5));
        assert!(keep_component(&mut m, w, h, 14, 5));
        assert_eq!(m[5 * w + 3], 0.0);
        assert_eq!(m[5 * w + 14], 1.0);
    }
}
