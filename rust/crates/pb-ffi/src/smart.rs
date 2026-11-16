//! Smart settings, gallery replacement and shape crops for the app.

use crate::{PbError, PixelBoostCore, UpscaleSettings};
use pb_core::analyze::{self, Budget};
use pb_core::{meta, shape};

#[derive(uniffi::Record, Clone)]
pub struct PhotoAnalysis {
    /// "anime" or "photo".
    pub kind: String,
    pub width: u32,
    pub height: u32,
    /// Noise sigma in 8-bit levels.
    pub noise: f32,
    /// JPEG blocking 0..1.
    pub blocking: f32,
    /// Pixel-scale sharpness (~0.45 soft .. 0.85+ crisp).
    pub sharpness: f32,
    /// Skin-toned share of the middle of the frame.
    pub skin: f32,
    pub has_alpha: bool,
    /// Real detail as a fraction of the pixel size (1 = sharp; 0.25 = the
    /// image was blown up 4x before).
    pub real_resolution: f32,
    /// The settings chosen for this photo.
    pub settings: UpscaleSettings,
    pub est_seconds: f32,
    pub summary: String,
    pub reasons: Vec<String>,
}

#[derive(uniffi::Record, Clone)]
pub struct ShapeEntry {
    pub id: String,
    pub name: String,
    /// Stretches to the crop box (else keeps its proportions: lock 1:1).
    pub stretches: bool,
}

#[derive(uniffi::Record, Clone)]
pub struct Replacement {
    pub path: String,
    pub width: u32,
    pub height: u32,
    /// "jpg" or "png".
    pub format: String,
    pub had_exif: bool,
    pub had_profile: bool,
}

#[uniffi::export]
impl PixelBoostCore {
    /// Look at the photo and choose every upscale setting for it.
    /// `device_gflops`: measured phone speed; `max_seconds`: longest run the
    /// user accepts on the phone; `cloud`: plan for the server instead.
    pub fn analyze(&self, input_path: String, device_gflops: f32, max_seconds: f32, max_output_mp: f32, cloud: bool) -> Result<PhotoAnalysis, PbError> {
        let m = analyze::measure_file(&input_path)?;
        Ok(analysis(&m, budget(device_gflops, max_seconds, max_output_mp, cloud), &analyze::Fixed::default(), max_output_mp))
    }

    /// Fill in every setting left on Auto (model "auto", scale <= 0,
    /// clarity "auto", detail < 0, quality 0, prescale 0) from the photo,
    /// keeping the ones the user set. The analysis looks at the pixels:
    /// real resolution, drawn vs photo, noise, JPEG blocks, softness, skin.
    pub fn resolve(&self, input_path: String, settings: UpscaleSettings, device_gflops: f32, max_seconds: f32, cloud: bool) -> Result<PhotoAnalysis, PbError> {
        let m = analyze::measure_file(&input_path)?;
        let mo = if settings.max_output_mp > 0.0 { settings.max_output_mp } else { 120.0 };
        let mut a = analysis(&m, budget(device_gflops, max_seconds, mo, cloud), &fixed_of(&settings), mo);
        a.settings.tile = settings.tile.max(32);
        Ok(a)
    }
    /// Build the file that replaces a photo in the library: the upscaled
    /// pixels with the original's EXIF (date, camera, place; orientation
    /// reset, size updated) and colour profile. JPEG unless the result has
    /// transparency and `allow_png`. `output_base` gets the extension.
    pub fn prepare_replacement(&self, original_path: String, result_path: String, output_base: String, quality: u8, allow_png: bool) -> Result<Replacement, PbError> {
        let bytes = std::fs::read(&original_path).map_err(|e| PbError::from(format!("{original_path}: {e}")))?;
        let m = meta::read(&bytes);
        drop(bytes);
        // Write to a temp name, then rename with the right extension.
        let tmp = format!("{output_base}.tmp");
        let (w, h, ext) = meta::write_with_meta(&result_path, &tmp, &m, quality, allow_png)?;
        let path = format!("{output_base}.{ext}");
        std::fs::rename(&tmp, &path).map_err(|e| PbError::from(e.to_string()))?;
        Ok(Replacement { path, width: w, height: h, format: ext.into(), had_exif: m.exif.is_some(), had_profile: m.icc.is_some() })
    }

    /// Adjustment data stored with an in-place edit, so Photos (and this
    /// app) can tell what was done. JSON, small.
    pub fn replacement_note(&self, model: String, scale: f32, in_width: u32, in_height: u32) -> String {
        serde_json::json!({ "app": "PixelBoost", "op": "upscale", "model": model, "scale": scale, "from": [in_width, in_height] }).to_string()
    }

    pub fn shapes(&self) -> Vec<ShapeEntry> {
        shape::SHAPES.iter().map(|(id, name)| ShapeEntry { id: (*id).into(), name: (*name).into(), stretches: shape::stretches(id) }).collect()
    }

    pub fn ring_styles(&self) -> Vec<ShapeEntry> {
        shape::RING_STYLES.iter().map(|(id, name)| ShapeEntry { id: (*id).into(), name: (*name).into(), stretches: false }).collect()
    }

    /// Outline of `shape` in a `width` x `height` (pixel) box, `offset`
    /// pixels outside the edge (negative = inside), as x,y pairs normalised
    /// to the box. For drawing the crop overlay and ring previews.
    #[allow(clippy::too_many_arguments)]
    pub fn shape_outline(&self, shape: String, width: f32, height: f32, rotation: f32, param: f32, offset: f32, points: u32) -> Result<Vec<f32>, PbError> {
        let pts = shape::outline(&shape, width.max(4.0), height.max(4.0), rotation, param, offset, points as usize)?;
        Ok(pts.into_iter().flatten().collect())
    }
}

fn budget(gflops: f32, max_seconds: f32, max_output_mp: f32, cloud: bool) -> Budget {
    Budget { gflops: gflops.max(1.0), max_seconds: max_seconds.max(5.0), max_output_mp, cloud }
}

/// The user's fixed choices in `s` (Auto sentinels become `None`).
pub(crate) fn fixed_of(s: &UpscaleSettings) -> analyze::Fixed {
    let photo = s.model == "photo";
    analyze::Fixed {
        model: (s.model != "auto").then(|| s.model.clone()),
        scale: (s.scale >= 1.0).then_some(s.scale),
        fidelity: (s.fidelity != "auto").then(|| s.fidelity.clone()),
        detail: (s.detail >= 0.0).then_some(s.detail),
        ensemble: (s.ensemble > 0).then_some(s.ensemble as usize),
        denoise: (s.model == "fast").then_some(s.denoise),
        realism: photo.then_some(if s.realism > 0.0 { s.realism } else { 1.0 }),
        adaptive: photo.then_some(s.adaptive),
        prescale: (s.prescale > 0.0).then_some(s.prescale),
    }
}

/// Whether any field is left on Auto.
pub(crate) fn has_auto(s: &UpscaleSettings) -> bool {
    s.model == "auto" || s.scale < 1.0 || s.fidelity == "auto" || s.detail < 0.0 || s.ensemble == 0 || s.prescale <= 0.0
}

fn analysis(m: &analyze::Measures, b: Budget, fx: &analyze::Fixed, max_output_mp: f32) -> PhotoAnalysis {
    let r = analyze::recommend_fixed(m, b, fx);
    PhotoAnalysis {
        kind: m.kind.into(),
        width: m.width as u32,
        height: m.height as u32,
        noise: m.noise,
        blocking: m.blocking,
        sharpness: m.sharpness,
        skin: m.skin,
        has_alpha: m.has_alpha,
        real_resolution: 1.0 / m.eff.max(1.0),
        settings: UpscaleSettings {
            model: r.model,
            scale: r.scale,
            fidelity: r.fidelity,
            detail: r.detail,
            denoise: r.denoise,
            ensemble: r.ensemble as u32,
            tile: 128,
            max_output_mp,
            realism: r.realism,
            adaptive: r.adaptive,
            prescale: r.prescale,
        },
        est_seconds: r.est_seconds,
        summary: r.summary,
        reasons: r.reasons,
    }
}
