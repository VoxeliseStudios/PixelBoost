//! Animated pictures (GIF / APNG / WebP) and video frames for the app.
//! Video files are decoded and encoded by the platform (AVFoundation on
//! iOS: Apple's codecs, no Rust equivalent on device); every frame goes
//! through [`VideoUpscaler`] here, which upscales and stabilises it.

use crate::{from_rgba, CancelToken, PbError, PixelBoostCore, RgbaImage, UpscaleSettings};
use pb_core::animate::{self, Temporal};
use pb_core::stream::Rgba8;
use std::sync::{Arc, Mutex};

#[derive(uniffi::Record, Clone)]
pub struct AnimationInfo {
    pub frames: u32,
    pub width: u32,
    pub height: u32,
    /// Total duration in ms.
    pub duration_ms: u32,
}

#[derive(uniffi::Record, Clone)]
pub struct AnimationOutcome {
    pub output_path: String,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub seconds: f64,
    pub run_id: String,
}

#[uniffi::export(with_foreign)]
pub trait FrameListener: Send + Sync {
    fn on_frame(&self, done: u32, total: u32);
}

#[uniffi::export]
impl PixelBoostCore {
    /// Frame count etc. of a GIF / APNG / WebP (`frames` = 1 for stills).
    pub fn animation_info(&self, path: String) -> Result<AnimationInfo, PbError> {
        let b = std::fs::read(&path).map_err(|e| PbError::from(e.to_string()))?;
        let f = animate::decode(&b)?;
        let first = f.first().ok_or_else(|| PbError::from("no frames".to_string()))?;
        Ok(AnimationInfo { frames: f.len() as u32, width: first.img.w as u32, height: first.img.h as u32, duration_ms: f.iter().map(|x| x.delay_ms).sum() })
    }

    /// Upscale every frame of an animation (output .png = APNG, .gif = GIF).
    /// Settings must be resolved (no Auto); 4x photo models on long clips
    /// are slow: the app suggests Fast / Line Art.
    pub fn upscale_animation(&self, input_path: String, output_path: String, settings: UpscaleSettings, listener: Arc<dyn FrameListener>, cancel: Arc<CancelToken>) -> Result<AnimationOutcome, PbError> {
        let run = pb_core::telemetry::new_run_id();
        pb_core::telemetry::scope(&run, || {
            let t0 = std::time::Instant::now();
            let (_model, stack, st, lim, _, _) = self.prepare(&settings)?;
            let r = animate::upscale_file(&stack, &input_path, &output_path, &st, lim, &mut |d, n| {
                listener.on_frame(d as u32, n as u32);
                !cancel.is_cancelled()
            });
            let (w, h, n) = r.map_err(|e| if cancel.is_cancelled() { PbError::Cancelled } else { PbError::from(e) })?;
            Ok(AnimationOutcome { output_path, width: w as u32, height: h as u32, frames: n as u32, seconds: t0.elapsed().as_secs_f64(), run_id: run.clone() })
        })
    }
}

/// One video's upscaler: frames in display order, one at a time.
#[derive(uniffi::Object)]
pub struct VideoUpscaler {
    core: Arc<PixelBoostCore>,
    settings: UpscaleSettings,
    state: Mutex<Option<(pb_core::fusion::Stack, pb_core::pipeline::Settings, pb_core::stream::Limits, Temporal, String)>>,
    run: String,
}

#[uniffi::export]
impl VideoUpscaler {
    #[uniffi::constructor]
    pub fn new(core: Arc<PixelBoostCore>, settings: UpscaleSettings) -> Arc<Self> {
        Arc::new(VideoUpscaler { core, settings, state: Mutex::new(None), run: pb_core::telemetry::new_run_id() })
    }

    pub fn run_id(&self) -> String {
        self.run.clone()
    }

    /// Upscale the next frame (straight RGBA8).
    pub fn push(&self, frame: RgbaImage) -> Result<RgbaImage, PbError> {
        let mut g = self.state.lock().unwrap();
        if g.is_none() {
            let (model, stack, st, lim, _, _) = self.core.prepare(&self.settings)?;
            *g = Some((stack, st, lim, Temporal::default(), model));
        }
        let (stack, st, lim, tp, _) = g.as_mut().unwrap();
        let _ = from_rgba(&frame)?; // size check
        let src = Rgba8 { w: frame.width as usize, h: frame.height as usize, data: frame.rgba };
        let out = pb_core::telemetry::scope(&self.run, || tp.push(stack, &src, st, *lim, &mut |_| true)).ok_or(PbError::Cancelled)?;
        Ok(RgbaImage { width: out.w as u32, height: out.h as u32, rgba: out.data })
    }

    /// Output size for a source frame size.
    pub fn output_size(&self, width: u32, height: u32) -> Vec<u32> {
        let s = self.settings.scale.clamp(1.0, 8.0);
        vec![(width as f32 * s).round() as u32, (height as f32 * s).round() as u32]
    }

    /// Done (telemetry summary).
    pub fn finish(&self) {
        if let Some((_, _, _, tp, model)) = self.state.lock().unwrap().as_ref() {
            pb_core::telemetry::scope(&self.run, || tp.report("video", model));
        }
    }
}
