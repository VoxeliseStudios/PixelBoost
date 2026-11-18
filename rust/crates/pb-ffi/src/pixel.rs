//! Pixel Art tool API: styles, palettes and the op JSON are all built here,
//! so the Swift panel only shows controls bound to [`PixelArtSettings`].

use crate::PixelBoostCore;
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
            o.remove("custom");
            o.remove("cell");
        }
        v.to_string()
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
