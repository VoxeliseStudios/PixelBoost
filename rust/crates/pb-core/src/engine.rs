//! Tiled model execution: every source pixel goes through the network.
//!
//! The image is cut into tiles with a context margin; each tile runs through
//! the network with the margin, and only its inner part is kept, so seams
//! see the same neighbourhood a whole-image pass would. Identical tiles
//! (flat backgrounds, letterboxing, UI) run once.

use crate::image::Image;
use crate::models::{self, ModelInfo};
use pb_nn::pbm::{self, Weights};
use pb_nn::{Fmap, SrNetwork};
use std::collections::HashMap;
use std::path::Path;

pub struct Engine {
    net: Box<dyn SrNetwork>,
    pub info: ModelInfo,
}

/// Weight-space adjustments applied when a model loads.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Tuning {
    /// `fast` only: blend in the denoising twin (0..1).
    pub denoise: Option<f32>,
    /// `photo` only: GAN texture strength (1 = Real-ESRGAN, 0 = RealESRNet).
    pub realism: Option<f32>,
}

impl Tuning {
    /// Cache key part ("d0.40 r1.00").
    pub fn key(&self) -> String {
        format!("d{:.2} r{:.2}", self.denoise.unwrap_or(0.0), self.realism.unwrap_or(1.0))
    }
}

/// Where networks run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// The GPU when there is one, else the CPU.
    Auto,
    Cpu,
    Gpu,
}

static BACKEND: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Choose the backend for engines loaded from now on (the app's setting;
/// `PB_BACKEND=cpu|gpu` overrides it for the CLI and servers).
pub fn set_backend(b: Backend) {
    BACKEND.store(b as u8, std::sync::atomic::Ordering::Relaxed);
}

pub fn backend() -> Backend {
    match std::env::var("PB_BACKEND").as_deref() {
        Ok("cpu") => return Backend::Cpu,
        Ok("gpu") => return Backend::Gpu,
        _ => {}
    }
    match BACKEND.load(std::sync::atomic::Ordering::Relaxed) {
        1 => Backend::Cpu,
        2 => Backend::Gpu,
        _ => Backend::Auto,
    }
}

/// Where the GPU tuning result is cached (first GPU use tunes the kernels
/// for this device, ~1-2 s, then reuses the file).
pub fn set_gpu_tune_cache(path: &Path) {
    #[cfg(feature = "gpu")]
    pb_gpu::tune::set_cache_path(path);
    #[cfg(not(feature = "gpu"))]
    let _ = path;
}

/// Measured GPU and CPU speed (GFLOP/s) after tuning, when there is a GPU.
pub fn gpu_speed() -> Option<(f64, f64)> {
    #[cfg(feature = "gpu")]
    {
        let g = pb_gpu::Gpu::get()?;
        pb_gpu::tune::ensure(&g).map(|r| (r.gpu_gflops, r.cpu_gflops))
    }
    #[cfg(not(feature = "gpu"))]
    {
        None
    }
}

/// The GPU's name when GPU inference is available ("AMD Radeon 610M (Vulkan)").
pub fn gpu_name() -> Option<String> {
    #[cfg(feature = "gpu")]
    {
        pb_gpu::Gpu::get().map(|g| g.describe())
    }
    #[cfg(not(feature = "gpu"))]
    {
        None
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TileOptions {
    /// Inner tile size in source pixels.
    pub tile: usize,
    /// Context margin in source pixels on every side.
    pub pad: usize,
}

impl Default for TileOptions {
    fn default() -> Self {
        TileOptions { tile: 128, pad: 16 }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TileEvent<'a> {
    pub done: usize,
    pub total: usize,
    /// Finished tile in output pixels: x, y, w, h.
    pub rect: [usize; 4],
    /// The tile's model output (for live previews).
    pub tile: &'a Image,
}

fn read(dir: &Path, file: &str) -> Result<Weights, String> {
    let p = dir.join(file);
    let bytes = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    pbm::read(&bytes)
}

impl Engine {
    /// Load a bundled model by id. `denoise` (0..1) only applies to `fast`:
    /// it blends in the denoising twin weights.
    pub fn load(dir: &Path, id: &str, denoise: Option<f32>) -> Result<Engine, String> {
        Self::load_on(dir, id, Tuning { denoise, realism: None }, backend())
    }

    /// Load with weight-space tuning (denoise for `fast`, realism for `photo`).
    pub fn load_tuned(dir: &Path, id: &str, t: Tuning) -> Result<Engine, String> {
        Self::load_on(dir, id, t, backend())
    }

    fn weights(dir: &Path, id: &str, t: Tuning) -> Result<Weights, String> {
        let info = models::find(id).ok_or_else(|| format!("unknown model {id}"))?;
        let mut w = read(dir, info.file)?;
        if let (Some(d), "fast") = (t.denoise, id) {
            if d > 0.0 {
                let wdn = read(dir, models::find("fast-denoise").unwrap().file)?;
                w = pbm::blend(&wdn, &w, d.clamp(0.0, 1.0))?;
            }
        }
        if let (Some(r), "photo") = (t.realism, id) {
            // Real-ESRGAN x4plus was fine-tuned from RealESRNet x4plus, so
            // interpolating their weights walks the perception-distortion
            // curve: 1 = full GAN texture, 0 = the faithful network.
            if r < 0.999 {
                let clean = read(dir, models::find("clean").unwrap().file)?;
                w = pbm::blend(&w, &clean, r.clamp(0.0, 1.0))?;
            }
        }
        Ok(w)
    }

    /// Load on a specific backend (`Gpu` falls back to the CPU when there
    /// is no usable GPU).
    pub fn load_on(dir: &Path, id: &str, tune: Tuning, be: Backend) -> Result<Engine, String> {
        let info = *models::find(id).ok_or_else(|| format!("unknown model {id}"))?;
        let w = Self::weights(dir, id, tune)?;
        #[cfg(feature = "gpu")]
        if be != Backend::Cpu {
            if let Some(gpu) = pb_gpu::Gpu::get().filter(|g| be == Backend::Gpu || pb_gpu::tune::ensure(g).map_or(true, |r| r.gpu_wins())) {
                let (d, i) = (dir.to_path_buf(), id.to_string());
                let fallback: pb_gpu::net_fallback::CpuFallback = Box::new(move || pb_nn::build(&Engine::weights(&d, &i, tune)?));
                match pb_gpu::GpuNet::new(gpu, &w, fallback) {
                    Ok(net) => return Ok(Engine { net: Box::new(net), info }),
                    Err(e) => eprintln!("pb-core: {id} on the GPU failed ({e}); using the CPU"),
                }
            }
        }
        let _ = be;
        let net = pb_nn::build(&w)?;
        Ok(Engine { net, info })
    }

    pub fn from_weights(w: &Weights, info: ModelInfo) -> Result<Engine, String> {
        Ok(Engine { net: pb_nn::build(w)?, info })
    }

    /// "cpu: avx2-fma 6x16" or "gpu: Apple A17 Pro GPU (Metal)".
    pub fn backend(&self) -> String {
        self.net.backend()
    }

    /// Whether this engine's network runs on the GPU.
    pub fn on_gpu(&self) -> bool {
        self.net.backend().starts_with("gpu")
    }

    pub fn scale(&self) -> usize {
        self.net.scale()
    }

    pub fn macs_per_px(&self) -> f64 {
        self.net.macs_per_px()
    }

    /// Run the network once over `region` (no tiling): the region is padded
    /// by edge replication to the network's alignment and the result cropped
    /// back to exactly `scale` times the region.
    pub fn forward(&self, region: &Image) -> Image {
        let s = self.net.scale();
        let a = self.net.align();
        let (w, h) = (region.w.div_ceil(a) * a, region.h.div_ceil(a) * a);
        let mut fm = Fmap::new(h, w, 3);
        for y in 0..h {
            for x in 0..w {
                let p = region.at_clamped(x as isize, y as isize);
                fm.px_mut(y, x).copy_from_slice(&p[..3]);
            }
        }
        let o = self.net.forward(&fm);
        let (ow, oh) = (region.w * s, region.h * s);
        let mut out = Image::new(ow, oh);
        for y in 0..oh {
            for x in 0..ow {
                let v = o.px(y, x);
                out.px[y * ow + x] = [v[0], v[1], v[2], 1.0];
            }
        }
        out
    }

    pub fn align(&self) -> usize {
        self.net.align()
    }

    /// Run the network over the whole image (RGB only; alpha is ignored and
    /// returned as 1). Returns `None` if `progress` asked to stop.
    pub fn run(&self, img: &Image, opt: TileOptions, progress: &mut dyn FnMut(TileEvent<'_>) -> bool) -> Option<Image> {
        let s = self.net.scale();
        let a = self.net.align();
        let round = |v: usize| v.div_ceil(a) * a;
        let (w, h) = (round(img.w), round(img.h));
        let tile = round(opt.tile.max(a * 8));
        let pad = round(opt.pad);
        let mut out = Image::new(w * s, h * s);
        let cols = w.div_ceil(tile);
        let rows = h.div_ceil(tile);
        let total = cols * rows;
        let mut cache: HashMap<u64, Image> = HashMap::new();
        let mut done = 0;
        for ty in 0..rows {
            for tx in 0..cols {
                let (ix, iy) = (tx * tile, ty * tile);
                let (iw, ih) = (tile.min(w - ix), tile.min(h - iy));
                let rx = ix.saturating_sub(pad);
                let ry = iy.saturating_sub(pad);
                let rw = (ix + iw + pad).min(w) - rx;
                let rh = (iy + ih + pad).min(h) - ry;
                let mut fm = Fmap::new(rh, rw, 3);
                let mut hasher = Fnv::new();
                hasher.add(&[rx as u64 % a as u64, ry as u64 % a as u64, rw as u64, rh as u64, (ix - rx) as u64, (iy - ry) as u64]);
                for y in 0..rh {
                    for x in 0..rw {
                        let p = img.at_clamped((rx + x) as isize, (ry + y) as isize);
                        let d = fm.px_mut(y, x);
                        d.copy_from_slice(&p[..3]);
                        hasher.add_f(&p[..3]);
                    }
                }
                let key = hasher.0;
                let part = match cache.get(&key) {
                    Some(p) => p.clone(),
                    None => {
                        let o = self.net.forward(&fm);
                        let mut part = Image::new(iw * s, ih * s);
                        let (ox, oy) = ((ix - rx) * s, (iy - ry) * s);
                        for y in 0..ih * s {
                            for x in 0..iw * s {
                                let v = o.px(oy + y, ox + x);
                                part.px[y * iw * s + x] = [v[0], v[1], v[2], 1.0];
                            }
                        }
                        cache.insert(key, part.clone());
                        part
                    }
                };
                out.paste(&part, ix * s, iy * s);
                done += 1;
                if !progress(TileEvent { done, total, rect: [ix * s, iy * s, iw * s, ih * s], tile: &part }) {
                    return None;
                }
            }
        }
        if out.w != img.w * s || out.h != img.h * s {
            out = out.crop(0, 0, img.w * s, img.h * s);
        }
        Some(out)
    }
}

/// FNV-1a over tile contents (exact float bits) — identical tiles only.
struct Fnv(u64);
impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn add(&mut self, v: &[u64]) {
        for x in v {
            for b in x.to_le_bytes() {
                self.0 ^= b as u64;
                self.0 = self.0.wrapping_mul(0x100_0000_01b3);
            }
        }
    }
    fn add_f(&mut self, v: &[f32]) {
        for x in v {
            self.0 ^= x.to_bits() as u64;
            self.0 = self.0.wrapping_mul(0x100_0000_01b3);
        }
    }
}
