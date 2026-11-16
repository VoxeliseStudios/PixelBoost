//! PixelBoostCore — the Rust engine as a Swift module (UniFFI).
//!
//! Images cross the boundary as files (full-size work) or raw RGBA8 buffers
//! (previews), never as UIImage/CGImage, so Swift does no pixel work.

use pb_core::fidelity::Fidelity;
use pb_core::image::Format;
use pb_core::pipeline::Settings;
use pb_core::{classify, models, ops, Engine, Image};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

pub mod editor;
pub mod history;
pub mod server;
pub mod smart;

uniffi::setup_scaffolding!();

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum PbError {
    #[error("{message}")]
    Failed { message: String },
    #[error("cancelled")]
    Cancelled,
}

impl From<String> for PbError {
    fn from(message: String) -> Self {
        PbError::Failed { message }
    }
}

#[derive(uniffi::Record, Clone)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub best_for: String,
    pub scale: u32,
    pub cost: f32,
}

#[derive(uniffi::Record, Clone)]
pub struct UpscaleSettings {
    /// A model id, or "auto".
    pub model: String,
    pub scale: f32,
    /// "off", "natural" or "faithful".
    pub fidelity: String,
    pub detail: f32,
    /// 0..1, only used by the "fast" model.
    pub denoise: f32,
    /// 1, 2, 4 or 8.
    pub ensemble: u32,
    pub tile: u32,
    /// Output cap in megapixels (0 = default 120).
    pub max_output_mp: f32,
    /// Photo model texture strength, 0..1 (1 = full GAN texture). Below 1
    /// the colours are always locked to the original (Natural clarity).
    pub realism: f32,
    /// Photo model: also run Clean and mix per pixel by source texture.
    pub adaptive: bool,
}

#[derive(uniffi::Record, Clone)]
pub struct UpscaleOutcome {
    pub output_path: String,
    pub width: u32,
    pub height: u32,
    pub model: String,
    pub seconds: f64,
    pub fidelity_db: f64,
    pub tiles: u32,
    pub passes: u32,
    /// The output was made smaller than asked to fit the size cap.
    pub capped: bool,
    /// Where the networks ran ("gpu: Apple A17 Pro GPU (Metal)").
    pub backend: String,
    /// Photo + Clean fusion ran (twice the network compute).
    pub adaptive: bool,
}

#[derive(uniffi::Record, Clone)]
pub struct ComputeInfo {
    /// "auto", "cpu" or "gpu".
    pub mode: String,
    /// The GPU PixelBoost can use, if any.
    pub gpu: Option<String>,
    /// SIMD kernel of the CPU path.
    pub cpu_kernel: String,
    /// Measured on the tuning conv (0 when not tuned yet).
    pub gpu_gflops: f64,
    pub cpu_gflops: f64,
    /// Whether Auto runs networks on the GPU.
    pub auto_uses_gpu: bool,
    /// Why the GPU is not used (no adapter, failed the self-check).
    pub gpu_note: Option<String>,
}

#[derive(uniffi::Record, Clone)]
pub struct RunPlan {
    pub model: String,
    pub out_width: u32,
    pub out_height: u32,
    pub tiles: u32,
    pub gflop: f64,
}

#[derive(uniffi::Record, Clone)]
pub struct ImageInfo {
    pub path: String,
    pub width: u32,
    pub height: u32,
    pub has_alpha: bool,
}

#[derive(uniffi::Record, Clone)]
pub struct RgbaImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Receives live progress. `rgba` is the finished tile's pixels (straight
/// RGBA8, `pixel_w` x `pixel_h`) to draw into the rectangle (`x`,`y`,`w`,`h`)
/// of the final output; empty when there is nothing to draw.
#[uniffi::export(with_foreign)]
pub trait UpscaleListener: Send + Sync {
    #[allow(clippy::too_many_arguments)]
    fn on_tile(&self, done: u32, total: u32, x: u32, y: u32, w: u32, h: u32, pixel_w: u32, pixel_h: u32, rgba: Vec<u8>);
}

/// Cancellation handle shared between Swift and a running job.
#[derive(uniffi::Object, Default)]
pub struct CancelToken {
    flag: AtomicBool,
}

#[uniffi::export]
impl CancelToken {
    #[uniffi::constructor]
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
}

#[derive(uniffi::Object)]
pub struct PixelBoostCore {
    models_dir: PathBuf,
    /// Loaded networks, keyed by model id (+ denoise level), kept warm
    /// between runs.
    cache: Mutex<HashMap<String, Arc<Engine>>>,
}

fn fmt_for(path: &str) -> Format {
    Format::from_path(path)
}

fn to_rgba(img: &Image) -> RgbaImage {
    RgbaImage { width: img.w as u32, height: img.h as u32, rgba: img.to_rgba8() }
}

pub(crate) fn from_rgba(r: &RgbaImage) -> Result<Image, PbError> {
    let (w, h) = (r.width as usize, r.height as usize);
    if r.rgba.len() != w * h * 4 {
        return Err(PbError::from(format!("rgba buffer is {} bytes, expected {}", r.rgba.len(), w * h * 4)));
    }
    let px = r.rgba.chunks_exact(4).map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0, c[3] as f32 / 255.0]).collect();
    Ok(Image { w, h, px })
}

#[uniffi::export]
impl PixelBoostCore {
    /// `models_dir` holds the bundled `.pbnn` files.
    #[uniffi::constructor]
    pub fn new(models_dir: String) -> Arc<Self> {
        Arc::new(PixelBoostCore { models_dir: PathBuf::from(models_dir), cache: Mutex::new(HashMap::new()) })
    }

    pub fn models(&self) -> Vec<ModelEntry> {
        models::MODELS
            .iter()
            .filter(|m| m.id != "fast-denoise")
            .map(|m| ModelEntry { id: m.id.into(), name: m.name.into(), best_for: m.best_for.into(), scale: m.scale, cost: m.cost })
            .collect()
    }

    pub fn kernel(&self) -> String {
        match pb_core::gpu_name() {
            Some(g) if pb_core::backend() != pb_core::Backend::Cpu => format!("GPU: {g} · CPU: {}", pb_nn::conv::kernel_name()),
            _ => pb_nn::conv::kernel_name().into(),
        }
    }

    /// "auto" (GPU when it is faster than the CPU), "gpu" or "cpu".
    pub fn set_compute(&self, mode: String) {
        pb_core::set_backend(match mode.as_str() {
            "cpu" => pb_core::Backend::Cpu,
            "gpu" => pb_core::Backend::Gpu,
            _ => pb_core::Backend::Auto,
        });
        self.cache.lock().unwrap().clear();
    }

    /// GPU + CPU details. The first call on a device tunes the GPU kernels
    /// (a second or two) and caches the result in `tune_cache_dir`.
    pub fn compute_info(&self) -> ComputeInfo {
        let gpu = pb_core::gpu_name();
        let (g, c) = pb_core::gpu_speed().unwrap_or((0.0, 0.0));
        ComputeInfo {
            mode: match pb_core::backend() {
                pb_core::Backend::Auto => "auto",
                pb_core::Backend::Cpu => "cpu",
                pb_core::Backend::Gpu => "gpu",
            }
            .into(),
            auto_uses_gpu: gpu.is_some() && (g == 0.0 || g > c * 1.1),
            gpu,
            cpu_kernel: pb_nn::conv::kernel_name().into(),
            gpu_gflops: g,
            cpu_gflops: c,
            gpu_note: pb_core::gpu_note(),
        }
    }

    /// Where to cache the per-device GPU tuning (call once at launch).
    pub fn set_cache_dir(&self, dir: String) {
        pb_core::set_gpu_tune_cache(&PathBuf::from(dir).join("pixelboost-gpu-tune.json"));
    }

    /// iOS does not allow GPU work in the background: pause it when the
    /// app leaves the foreground (running jobs continue on the CPU) and
    /// resume when it returns. Resuming also recovers from a lost GPU.
    pub fn set_gpu_paused(&self, paused: bool) {
        #[cfg(feature = "gpu")]
        {
            pb_gpu::set_paused(paused);
            if !paused && pb_gpu::Gpu::reset() {
                self.cache.lock().unwrap().clear();
            }
        }
        #[cfg(not(feature = "gpu"))]
        let _ = paused;
    }

    /// Which model Auto would use for this image.
    pub fn auto_model(&self, input_path: String) -> Result<String, PbError> {
        let img = Image::open(&input_path)?;
        Ok(classify::pick_model(&img).into())
    }

    pub fn image_info(&self, path: String) -> Result<ImageInfo, PbError> {
        let img = Image::open(&path)?;
        Ok(ImageInfo { width: img.w as u32, height: img.h as u32, has_alpha: img.has_alpha(), path })
    }

    /// Output size, tile count and compute cost before running.
    pub fn plan(&self, width: u32, height: u32, model: String, scale: f32, tile: u32) -> RunPlan {
        let m = models::find(&model).copied().unwrap_or(models::MODELS[0]);
        let t = tile.max(32);
        let tiles = width.div_ceil(t) * height.div_ceil(t);
        // RRDB-23 is ~18.4M MACs per input pixel at full cost.
        let gflop = 2.0 * 18.4e6 * m.cost as f64 * (width as f64) * (height as f64) / 1e9;
        RunPlan {
            model: m.id.into(),
            out_width: (width as f32 * scale).round() as u32,
            out_height: (height as f32 * scale).round() as u32,
            tiles,
            gflop,
        }
    }

    pub fn upscale(&self, input_path: String, output_path: String, settings: UpscaleSettings, listener: Arc<dyn UpscaleListener>, cancel: Arc<CancelToken>) -> Result<UpscaleOutcome, PbError> {
        let img = Image::open(&input_path)?;
        let model = if settings.model == "auto" { classify::pick_model(&img).to_string() } else { settings.model.clone() };
        let denoise = if model == "fast" && settings.denoise > 0.0 { Some(settings.denoise) } else { None };
        let realism = if settings.realism > 0.0 { settings.realism.min(1.0) } else { 1.0 };
        let stack = pb_core::fusion::Stack::build(&model, denoise, realism, settings.adaptive, &mut |id, t| {
            let key = format!("{id}:{}", t.key());
            let mut c = self.cache.lock().unwrap();
            if let Some(e) = c.get(&key) {
                return Ok(e.clone());
            }
            // Keep at most three networks resident (they are large;
            // adaptive needs two at once).
            if c.len() >= 3 {
                c.clear();
            }
            let e = Arc::new(Engine::load_tuned(&self.models_dir, id, t)?);
            c.insert(key, e.clone());
            Ok(e)
        })?;
        let mut fidelity = Fidelity::parse(&settings.fidelity).unwrap_or(Fidelity::Natural);
        if model == "photo" && realism < 0.999 && !settings.adaptive && fidelity == Fidelity::Off {
            // Interpolated weights can drift in colour; clarity locks it.
            fidelity = Fidelity::Natural;
        }
        let st = Settings {
            scale: settings.scale,
            fidelity,
            detail: settings.detail.clamp(0.0, 1.0),
            tile: settings.tile.clamp(32, 512) as usize,
            pad: 16,
            ensemble: settings.ensemble.max(1) as usize,
            realism,
            adaptive: settings.adaptive,
        };
        let lim = pb_core::stream::Limits { max_output_px: if settings.max_output_mp > 0.0 { (settings.max_output_mp as f64 * 1e6) as usize } else { 120_000_000 } };
        let res = pb_core::stream::upscale(&stack, &img, &st, lim, &mut |e| {
            let [x, y, w, h] = e.rect;
            let rgba = if w > 0 { e.pixels.data.clone() } else { Vec::new() };
            let (pw, ph) = (e.pixels.w as u32, e.pixels.h as u32);
            // `rect` is where the tile lands in the final output; the pixels
            // may be at a different (first-pass) resolution.
            listener.on_tile(e.done as u32, e.total as u32, x as u32, y as u32, w as u32, h as u32, pw, ph, rgba);
            !cancel.is_cancelled()
        });
        drop(img);
        let (out, rep, capped) = res.ok_or(PbError::Cancelled)?;
        out.save(&output_path, 95)?;
        Ok(UpscaleOutcome {
            output_path,
            width: out.w as u32,
            height: out.h as u32,
            model: model.clone(),
            seconds: rep.seconds,
            fidelity_db: rep.fidelity_db,
            tiles: rep.tiles as u32,
            passes: rep.passes as u32,
            capped,
            backend: stack.backend(),
            adaptive: matches!(stack, pb_core::fusion::Stack::Fused(_)),
        })
    }

    /// Run one editing operation (JSON, see pb_core::ops::Op) file to file.
    pub fn edit_file(&self, input_path: String, output_path: String, op_json: String) -> Result<ImageInfo, PbError> {
        let img = Image::open(&input_path)?;
        let op = ops::parse(&op_json)?;
        let out = ops::apply(&img, &op)?;
        let bytes = out.encode(fmt_for(&output_path), 95)?;
        std::fs::write(&output_path, bytes).map_err(|e| PbError::from(e.to_string()))?;
        Ok(ImageInfo { path: output_path, width: out.w as u32, height: out.h as u32, has_alpha: out.has_alpha() })
    }

    /// Same, on an in-memory RGBA8 buffer (live previews).
    pub fn edit_rgba(&self, image: RgbaImage, op_json: String) -> Result<RgbaImage, PbError> {
        let img = from_rgba(&image)?;
        let op = ops::parse(&op_json)?;
        Ok(to_rgba(&ops::apply(&img, &op)?))
    }

    /// Decode any supported file to RGBA8, shrunk to fit `max_side`.
    pub fn load_preview(&self, path: String, max_side: u32) -> Result<RgbaImage, PbError> {
        let img = Image::open(&path)?;
        let m = max_side.max(16) as usize;
        let f = m as f32 / img.w.max(img.h) as f32;
        let img = if f < 1.0 { pb_core::resample::resize(&img, ((img.w as f32 * f).round() as usize).max(1), ((img.h as f32 * f).round() as usize).max(1), pb_core::resample::Filter::Lanczos3) } else { img };
        Ok(to_rgba(&img))
    }

    pub fn save_rgba(&self, image: RgbaImage, path: String, quality: u8) -> Result<(), PbError> {
        let img = from_rgba(&image)?;
        let bytes = img.encode(fmt_for(&path), quality)?;
        std::fs::write(&path, bytes).map_err(|e| PbError::from(e.to_string()))
    }

    /// Source-vs-result agreement (dB) for any two files.
    pub fn fidelity(&self, source_path: String, result_path: String) -> Result<f64, PbError> {
        let a = Image::open(&source_path)?;
        let b = Image::open(&result_path)?;
        let s = (b.w / a.w).max(1);
        if b.w != a.w * s || b.h != a.h * s {
            return Err(PbError::from("result is not an integer multiple of the source".to_string()));
        }
        Ok(pb_core::fidelity::fidelity_db(&a, &b, s))
    }

    /// Names of the filter looks.
    pub fn looks(&self) -> Vec<String> {
        pb_core::edit::LOOKS.iter().map(|s| s.to_string()).collect()
    }
}
