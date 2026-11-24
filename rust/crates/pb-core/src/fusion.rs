//! Adaptive fusion: two networks, one picture.
//!
//! GAN upscalers (Real-ESRGAN) rebuild texture — foliage, hair, fabric,
//! stone — but they also invent grain in flat areas (sky, walls, bokeh)
//! and waxy or crawling detail on skin. Faithful networks (RealESRNet) do
//! the opposite. Fusion runs both on every tile and mixes them per pixel
//! with a *texture map* measured on the source: where the original has
//! fine structure the detail network leads, where it is flat or skin the
//! faithful one does.
//!
//! The map depends only on source pixels inside a small window (radius
//! 8 px), well inside the streaming tile margin (24 px), so tiles fuse
//! without seams.

use crate::engine::Engine;
use crate::filters::{box_blur_plane, sample_plane, smoothstep};
use crate::image::Image;
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;

/// Anything that turns a tile into a `scale()` x bigger tile.
pub trait Upscaler: Sync {
    fn scale(&self) -> usize;
    fn forward(&self, region: &Image) -> Image;
    /// Report name ("photo", "photo+clean fusion").
    fn label(&self) -> String;
    /// Network MACs per input pixel (all networks involved).
    fn macs_per_px(&self) -> f64;
    /// The networks run on the GPU (the CPU is free for post-processing).
    fn on_gpu(&self) -> bool {
        false
    }
}

impl Upscaler for Engine {
    fn scale(&self) -> usize {
        Engine::scale(self)
    }
    fn forward(&self, region: &Image) -> Image {
        Engine::forward(self, region)
    }
    fn label(&self) -> String {
        self.info.id.to_string()
    }
    fn macs_per_px(&self) -> f64 {
        Engine::macs_per_px(self)
    }
    fn on_gpu(&self) -> bool {
        Engine::on_gpu(self)
    }
}

pub struct Fusion {
    /// Texture-rebuilding network (GAN).
    pub detail: Arc<Engine>,
    /// Faithful network (same scale).
    pub faithful: Arc<Engine>,
    /// 0..1: how far textured areas go towards the detail network.
    pub strength: f32,
}

impl Fusion {
    pub fn new(detail: Arc<Engine>, faithful: Arc<Engine>, strength: f32) -> Result<Self, String> {
        if detail.scale() != faithful.scale() {
            return Err("fusion needs two networks of the same scale".into());
        }
        Ok(Fusion { detail, faithful, strength: strength.clamp(0.0, 1.0) })
    }
}

impl<T: Upscaler + Send> Upscaler for Arc<T> {
    fn scale(&self) -> usize {
        (**self).scale()
    }
    fn forward(&self, region: &Image) -> Image {
        (**self).forward(region)
    }
    fn label(&self) -> String {
        (**self).label()
    }
    fn macs_per_px(&self) -> f64 {
        (**self).macs_per_px()
    }
    fn on_gpu(&self) -> bool {
        (**self).on_gpu()
    }
}

/// The networks one upscale needs, resolved from the user's settings:
/// the model (with denoise / realism tuning) and, for adaptive photo
/// upscales, the faithful partner.
pub enum Stack {
    One(Arc<Engine>),
    Fused(Fusion),
}

impl Stack {
    /// `load` gets (model id, tuning) and may serve engines from a cache.
    pub fn build(model: &str, denoise: Option<f32>, realism: f32, adaptive: bool, load: &mut dyn FnMut(&str, crate::engine::Tuning) -> Result<Arc<Engine>, String>) -> Result<Stack, String> {
        use crate::engine::Tuning;
        let realism = realism.clamp(0.0, 1.0);
        if model == "photo" && adaptive {
            let detail = load("photo", Tuning { denoise: None, realism: Some(1.0) })?;
            let faithful = load("clean", Tuning::default())?;
            // Realism scales how far textured areas go towards the GAN.
            return Ok(Stack::Fused(Fusion::new(detail, faithful, realism)?));
        }
        let t = Tuning { denoise: if model == "fast" { denoise.filter(|d| *d > 0.0) } else { None }, realism: if model == "photo" && realism < 0.999 { Some(realism) } else { None } };
        Ok(Stack::One(load(model, t)?))
    }

    /// Load straight from the models folder (no cache).
    pub fn load(dir: &Path, model: &str, denoise: Option<f32>, realism: f32, adaptive: bool) -> Result<Stack, String> {
        Self::build(model, denoise, realism, adaptive, &mut |id, t| Engine::load_tuned(dir, id, t).map(Arc::new))
    }

    pub fn backend(&self) -> String {
        match self {
            Stack::One(e) => e.backend(),
            Stack::Fused(f) => f.detail.backend(),
        }
    }
}

impl Upscaler for Stack {
    fn scale(&self) -> usize {
        match self {
            Stack::One(e) => e.scale(),
            Stack::Fused(f) => f.scale(),
        }
    }
    fn forward(&self, region: &Image) -> Image {
        match self {
            Stack::One(e) => e.forward(region),
            Stack::Fused(f) => f.forward(region),
        }
    }
    fn label(&self) -> String {
        match self {
            Stack::One(e) => Upscaler::label(&**e),
            Stack::Fused(f) => f.label(),
        }
    }
    fn macs_per_px(&self) -> f64 {
        match self {
            Stack::One(e) => e.macs_per_px(),
            Stack::Fused(f) => Upscaler::macs_per_px(f),
        }
    }
    fn on_gpu(&self) -> bool {
        match self {
            Stack::One(e) => e.on_gpu(),
            Stack::Fused(f) => f.detail.on_gpu(),
        }
    }
}

/// Skin likelihood of one pixel (soft YCbCr box, 0..1).
fn skin(p: [f32; 4]) -> f32 {
    let (r, g, b) = (p[0] * 255.0, p[1] * 255.0, p[2] * 255.0);
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b;
    let cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b;
    if r <= g || r <= b {
        return 0.0;
    }
    let inside = |v: f32, lo: f32, hi: f32, soft: f32| smoothstep(lo - soft, lo, v) * (1.0 - smoothstep(hi, hi + soft, v));
    inside(cb, 80.0, 125.0, 6.0) * inside(cr, 136.0, 170.0, 6.0) * smoothstep(40.0, 60.0, y)
}

/// Per-source-pixel weight of the detail network (0..1).
pub fn detail_map(img: &Image) -> Vec<f32> {
    let (w, h) = (img.w, img.h);
    let l: Vec<f32> = img.px.par_iter().map(|p| (0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]) * 255.0).collect();
    // Fine structure: |luma - 3x3 mean|, pooled over 5x5.
    let m = box_blur_plane(&l, w, h, 1);
    let hf: Vec<f32> = l.par_iter().zip(m.par_iter()).map(|(a, b)| (a - b).abs()).collect();
    let tex = box_blur_plane(&hf, w, h, 2);
    let sk: Vec<f32> = img.px.par_iter().map(|p| skin(*p)).collect();
    let sk = box_blur_plane(&sk, w, h, 3);
    // Texture level (8-bit units): < 1 is sensor-flat, > 4.5 clearly textured.
    let raw: Vec<f32> = tex.par_iter().zip(sk.par_iter()).map(|(t, s)| smoothstep(1.0, 4.5, *t) * (1.0 - 0.75 * s.clamp(0.0, 1.0))).collect();
    box_blur_plane(&raw, w, h, 2)
}

impl Upscaler for Fusion {
    fn scale(&self) -> usize {
        self.detail.scale()
    }

    fn label(&self) -> String {
        format!("{}+{} fusion", self.detail.info.id, self.faithful.info.id)
    }

    fn macs_per_px(&self) -> f64 {
        self.detail.macs_per_px() + self.faithful.macs_per_px()
    }

    fn on_gpu(&self) -> bool {
        self.detail.on_gpu()
    }

    fn forward(&self, region: &Image) -> Image {
        let s = self.scale();
        let mut a = self.detail.forward(region);
        let b = self.faithful.forward(region);
        let map = detail_map(region);
        let (w, h) = (region.w, region.h);
        let k = self.strength;
        let inv = 1.0 / s as f32;
        a.px.par_chunks_mut(a.w).zip(b.px.par_chunks(b.w)).enumerate().for_each(|(y, (ra, rb))| {
            let sy = (y as f32 + 0.5) * inv - 0.5;
            for (x, (pa, pb)) in ra.iter_mut().zip(rb).enumerate() {
                let t = sample_plane(&map, w, h, (x as f32 + 0.5) * inv - 0.5, sy) * k;
                for c in 0..3 {
                    pa[c] = pb[c] + (pa[c] - pb[c]) * t;
                }
            }
        });
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_is_faithful_texture_is_detail() {
        let (w, h) = (64, 32);
        let mut img = Image::new(w, h);
        let mut seed = 1u32;
        for y in 0..h {
            for x in 0..w {
                // Left half: flat blue sky. Right half: high-contrast noise.
                let v = if x < w / 2 {
                    [0.35, 0.55, 0.85, 1.0]
                } else {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let n = (seed >> 24) as f32 / 255.0;
                    [n, n, n, 1.0]
                };
                img.px[y * w + x] = v;
            }
        }
        let m = detail_map(&img);
        assert!(m[16 * w + 8] < 0.05, "flat area should use the faithful net: {}", m[16 * w + 8]);
        assert!(m[16 * w + 56] > 0.9, "texture should use the detail net: {}", m[16 * w + 56]);
    }

    #[test]
    fn skin_lowers_the_detail_weight() {
        let (w, h) = (32, 32);
        let mut img = Image::new(w, h);
        let mut seed = 7u32;
        for p in img.px.iter_mut() {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let n = ((seed >> 24) as f32 / 255.0 - 0.5) * 0.08;
            *p = [0.85 + n, 0.62 + n, 0.50 + n, 1.0];
        }
        let m = detail_map(&img);
        assert!(m[16 * w + 16] < 0.4, "skin should lean faithful: {}", m[16 * w + 16]);
    }
}
