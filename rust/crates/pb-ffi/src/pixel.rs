//! Pixel Art tool API: styles, palettes and the op JSON are all built here,
//! so the Swift panel only shows controls bound to [`PixelArtSettings`].

use crate::PixelBoostCore;
use crate::{PbError, RgbaImage};
use pb_core::pixelart::{self, PixelArt};

/// Every control of the Pixel Art panel.
#[derive(uniffi::Record, Clone, Debug, PartialEq)]
pub struct PixelArtSettings {
    /// Art pixels across (32..320).
    pub width: u32,
    /// Palette size for the adaptive palettes ("auto", "ramps").
    pub colors: u32,
    pub palette: String,
    pub dither: String,
    pub dither_amount: f32,
    pub outline: String,
    /// Line detail 0..1.
    pub lines: f32,
    pub rim: f32,
    /// Light direction in degrees (0 = right, 90 = top).
    pub light: f32,
    pub hue_shift: f32,
    pub punch: f32,
    pub smooth: f32,
    pub sparkle: f32,
    pub cleanup: bool,
    /// Output 1 px per art pixel instead of the photo's size.
    pub native: bool,
    /// A custom palette (0xRRGGBB); when set it replaces `palette`.
    pub custom: Vec<u32>,
}

impl From<PixelArt> for PixelArtSettings {
    fn from(p: PixelArt) -> Self {
        PixelArtSettings {
            width: p.width as u32,
            colors: p.colors as u32,
            palette: p.palette,
            dither: p.dither,
            dither_amount: p.dither_amount,
            outline: p.outline,
            lines: p.lines,
            rim: p.rim,
            light: p.light,
            hue_shift: p.hue_shift,
            punch: p.punch,
            smooth: p.smooth,
            sparkle: p.sparkle,
            cleanup: p.cleanup,
            native: p.native,
            custom: p.custom.iter().map(|c| hex(*c)).collect(),
        }
    }
}

impl From<&PixelArtSettings> for PixelArt {
    fn from(s: &PixelArtSettings) -> Self {
        PixelArt {
            width: (s.width as usize).clamp(8, 1024),
            colors: (s.colors as usize).clamp(2, 256),
            palette: s.palette.clone(),
            dither: s.dither.clone(),
            dither_amount: s.dither_amount,
            outline: s.outline.clone(),
            lines: s.lines,
            rim: s.rim,
            light: s.light,
            hue_shift: s.hue_shift,
            punch: s.punch,
            smooth: s.smooth,
            sparkle: s.sparkle,
            cleanup: s.cleanup,
            native: s.native,
            custom: s.custom.iter().take(256).map(|c| unhex(*c)).collect(),
            ..PixelArt::default()
        }
    }
}

#[derive(uniffi::Record, Clone)]
pub struct PixelStyle {
    pub id: String,
    pub name: String,
    pub blurb: String,
    pub settings: PixelArtSettings,
}

#[derive(uniffi::Record, Clone)]
pub struct PixelPalette {
    pub id: String,
    pub name: String,
    /// 0xRRGGBB swatches (empty for the adaptive palettes).
    pub swatches: Vec<u32>,
    /// Adaptive palettes use the Colours slider.
    pub adaptive: bool,
}

#[derive(uniffi::Record, Clone)]
pub struct PixelOption {
    pub id: String,
    pub name: String,
}

fn unhex(c: u32) -> [f32; 3] {
    [((c >> 16) & 255) as f32 / 255.0, ((c >> 8) & 255) as f32 / 255.0, (c & 255) as f32 / 255.0]
}

fn hex(c: [f32; 3]) -> u32 {
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    (b(c[0]) << 16) | (b(c[1]) << 8) | b(c[2])
}

#[uniffi::export]
impl PixelBoostCore {
    /// Built-in looks; the first is the default.
    pub fn pixel_styles(&self) -> Vec<PixelStyle> {
        pixelart::STYLES
            .iter()
            .map(|s| PixelStyle { id: s.id.into(), name: s.name.into(), blurb: s.blurb.into(), settings: (s.make)().into() })
            .collect()
    }

    pub fn pixel_palettes(&self) -> Vec<PixelPalette> {
        let mut out = vec![
            PixelPalette { id: "auto".into(), name: "From photo".into(), swatches: Vec::new(), adaptive: true },
            PixelPalette { id: "ramps".into(), name: "Ramps".into(), swatches: Vec::new(), adaptive: true },
        ];
        out.extend(pixelart::PALETTES.iter().map(|p| PixelPalette {
            id: p.id.into(),
            name: p.name.into(),
            swatches: pixelart::palette_colors(p.id).into_iter().map(hex).collect(),
            adaptive: false,
        }));
        out
    }

    pub fn pixel_dithers(&self) -> Vec<PixelOption> {
        pixelart::DITHERS.iter().map(|(id, n)| PixelOption { id: (*id).into(), name: (*n).into() }).collect()
    }

    pub fn pixel_outlines(&self) -> Vec<PixelOption> {
        pixelart::OUTLINES.iter().map(|(id, n)| PixelOption { id: (*id).into(), name: (*n).into() }).collect()
    }

    /// The `pixel_art` op for these settings (what the panel previews and
    /// applies).
    pub fn pixel_art_op(&self, settings: PixelArtSettings) -> String {
        let p: PixelArt = (&settings).into();
        let mut v = serde_json::to_value(&p).unwrap_or_default();
        v["op"] = "pixel_art".into();
        if let Some(o) = v.as_object_mut() {
            if settings.custom.is_empty() {
                o.remove("custom");
            }
            o.remove("cell");
        }
        v.to_string()
    }

    /// The main colours of a picture (seed for a custom palette).
    pub fn extract_palette(&self, image: RgbaImage, colors: u32) -> Result<Vec<u32>, PbError> {
        let img = crate::from_rgba(&image)?;
        Ok(pixelart::extract_palette(&img, colors as usize).into_iter().map(hex).collect())
    }
}

/// The user's own palettes (JSON in the app's documents).
#[derive(uniffi::Object)]
pub struct PaletteStore {
    file: std::path::PathBuf,
    list: std::sync::Mutex<Vec<PixelPalette>>,
}

#[uniffi::export]
impl PaletteStore {
    #[uniffi::constructor]
    pub fn new(dir: String) -> std::sync::Arc<Self> {
        let _ = std::fs::create_dir_all(&dir);
        let file = std::path::Path::new(&dir).join("palettes.json");
        let list: Vec<PixelPalette> = std::fs::read(&file).ok().and_then(|b| serde_json::from_slice::<Vec<(String, String, Vec<u32>)>>(&b).ok()).unwrap_or_default().into_iter().map(|(id, name, swatches)| PixelPalette { id, name, swatches, adaptive: false }).collect();
        std::sync::Arc::new(PaletteStore { file, list: std::sync::Mutex::new(list) })
    }

    pub fn list(&self) -> Vec<PixelPalette> {
        self.list.lock().unwrap().clone()
    }

    /// Create or update (same id) a palette; returns it.
    pub fn save(&self, id: String, name: String, swatches: Vec<u32>) -> Result<PixelPalette, PbError> {
        if swatches.len() < 2 {
            return Err(PbError::from("a palette needs at least 2 colours".to_string()));
        }
        let id = if id.is_empty() { format!("custom:{}", pb_core::telemetry::new_run_id()) } else { id };
        let p = PixelPalette { id: id.clone(), name: if name.trim().is_empty() { "My palette".into() } else { name.chars().take(40).collect() }, swatches: swatches.into_iter().take(256).collect(), adaptive: false };
        let mut l = self.list.lock().unwrap();
        match l.iter_mut().find(|x| x.id == id) {
            Some(x) => *x = p.clone(),
            None => l.push(p.clone()),
        }
        self.persist(&l)?;
        pb_core::telemetry::event("pixelart", "palette_saved").metric("colours", p.swatches.len() as f64).emit();
        Ok(p)
    }

    pub fn delete(&self, id: String) -> Result<(), PbError> {
        let mut l = self.list.lock().unwrap();
        l.retain(|x| x.id != id);
        self.persist(&l)
    }
}

impl PaletteStore {
    fn persist(&self, l: &[PixelPalette]) -> Result<(), PbError> {
        let v: Vec<(String, String, Vec<u32>)> = l.iter().map(|p| (p.id.clone(), p.name.clone(), p.swatches.clone())).collect();
        let tmp = self.file.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&v).unwrap_or_default()).map_err(|e| PbError::from(e.to_string()))?;
        std::fs::rename(&tmp, &self.file).map_err(|e| PbError::from(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_round_trips_through_core() {
        let core = PixelBoostCore::new("/nonexistent".into());
        for s in core.pixel_styles() {
            let json = core.pixel_art_op(s.settings.clone());
            let op = pb_core::ops::parse(&json).unwrap();
            match op {
                pb_core::ops::Op::PixelArt(p) => {
                    assert_eq!(PixelArtSettings::from(p), s.settings, "{}", s.id);
                }
                _ => panic!("not a pixel_art op"),
            }
        }
        let pals = core.pixel_palettes();
        assert!(pals.iter().filter(|p| !p.adaptive).all(|p| !p.swatches.is_empty()));
        assert_eq!(core.pixel_dithers().len(), 4);
    }
}
