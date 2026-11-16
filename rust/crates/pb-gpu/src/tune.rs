//! On-device autotuning.
//!
//! The best workgroup tile differs a lot between GPUs (AMD wave64 vs Apple
//! SIMD32, threadgroup memory size, register file). The first time a device
//! runs PixelBoost, each block width's candidate tiles are timed on a
//! representative conv and the fastest is kept — together with the GPU's
//! and the CPU's measured speed, so "Auto" can stay on the CPU of devices
//! whose GPU is slower than their CPU. Results are cached per device.

use crate::net::GpuNet;
use crate::shaders::{set_tiles, tiles, Tile, WIDTHS};
use crate::Gpu;
use pb_nn::conv::{conv3x3, Conv3x3, Epi};
use pb_nn::fmap::Fmap;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TuneResult {
    /// Adapter name + driver: the cache is only valid for the same device.
    pub device: String,
    /// bm, bk, tm, tn per block width 64 / 32 / 16.
    pub tiles: [[u32; 4]; 3],
    /// Measured on the 64-channel conv with the chosen tile.
    pub gpu_gflops: f64,
    pub cpu_gflops: f64,
    /// Tuning format version.
    pub v: u32,
}

impl TuneResult {
    /// The GPU is worth using over this CPU.
    pub fn gpu_wins(&self) -> bool {
        self.gpu_gflops > self.cpu_gflops * 1.1
    }
}

const VERSION: u32 = 1;

fn candidates(bn: u32) -> Vec<Tile> {
    let all = [
        Tile::new(128, 16, 8, 8),
        Tile::new(64, 16, 8, 8),
        Tile::new(128, 8, 8, 8),
        Tile::new(64, 16, 4, 8),
        Tile::new(128, 16, 8, 4),
        Tile::new(64, 16, 8, 4),
        Tile::new(128, 8, 8, 4),
        Tile::new(64, 16, 4, 4),
        Tile::new(128, 16, 4, 4),
        Tile::new(64, 32, 8, 4),
        Tile::new(32, 16, 4, 4),
    ];
    let mut v: Vec<Tile> = all.into_iter().filter(|t| t.valid(bn)).collect();
    v.dedup();
    v
}

fn input(h: usize, w: usize, c: usize) -> Fmap {
    let mut x = Fmap::new(h, w, c);
    for y in 0..h {
        for xx in 0..w {
            for (i, v) in x.px_mut(y, xx).iter_mut().enumerate() {
                *v = ((y * 31 + xx * 17 + i * 7) % 97) as f32 / 97.0;
            }
        }
    }
    x
}

/// GFLOP/s of one tile on a (cin -> cout) conv over an `n` x `n` map.
fn time_gpu(net: &GpuNet, x: &Fmap, flop: f64) -> Option<f64> {
    net.try_forward(x).ok()?; // warm-up: compiles, allocates
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        net.try_forward(x).ok()?;
        best = best.min(t.elapsed().as_secs_f64());
    }
    Some(flop / best / 1e9)
}

fn time_cpu(cin: usize, cout: usize, n: usize) -> f64 {
    let w: Vec<f32> = (0..cin * cout * 9).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
    let conv = Conv3x3::from_torch(&w, None, cout, cin);
    let x = input(n, n, cin);
    let mut out = Fmap::new(n, n, cout);
    conv3x3(&x, 0, &conv, &mut out, 0, Epi::PLAIN);
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        conv3x3(&x, 0, &conv, &mut out, 0, Epi::PLAIN);
        best = best.min(t.elapsed().as_secs_f64());
    }
    (n * n * cin * cout * 9 * 2) as f64 / best / 1e9
}

/// Time every candidate tile and install the fastest per block width.
pub fn autotune(gpu: &Arc<Gpu>) -> Result<TuneResult, String> {
    let n = 96;
    let reps = 4;
    let mut chosen = tiles();
    let mut gpu_gflops = 0.0;
    for (i, &bn) in WIDTHS.iter().enumerate() {
        let (cin, cout): (usize, usize) = match bn {
            64 => (192, 64),
            32 => (128, 32),
            _ => (64, 48),
        };
        let x = input(n, n, cin);
        let flop = (n * n * cin * cout.div_ceil(bn as usize) * bn as usize * 9 * 2 * reps) as f64;
        let mut best: Option<(Tile, f64)> = None;
        for t in candidates(bn) {
            let mut table = chosen;
            table[i] = t;
            set_tiles(table);
            // A fresh network per candidate: dispatch sizes depend on the tile.
            let net = GpuNet::synthetic(gpu.clone(), cin, cout, reps)?;
            if let Some(g) = time_gpu(&net, &x, flop) {
                if best.map_or(true, |(_, b)| g > b) {
                    best = Some((t, g));
                }
            }
        }
        let (t, g) = best.ok_or("no tile ran on this GPU")?;
        chosen[i] = t;
        set_tiles(chosen);
        if bn == 64 {
            gpu_gflops = g;
        }
    }
    let cpu_gflops = time_cpu(192, 64, n);
    Ok(TuneResult {
        device: device_key(gpu),
        tiles: chosen.map(|t| [t.bm, t.bk, t.tm, t.tn]),
        gpu_gflops,
        cpu_gflops,
        v: VERSION,
    })
}

fn device_key(gpu: &Gpu) -> String {
    format!("{} | {} {} | {:?}", gpu.info.name, gpu.info.driver, gpu.info.driver_info, gpu.info.backend)
}

static CACHE_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);
static RESULT: OnceLock<Option<TuneResult>> = OnceLock::new();

/// Where to keep the tuning result (the app's caches folder). Without it,
/// `PB_GPU_TUNE` names the file, else the defaults are used untuned.
pub fn set_cache_path(p: &Path) {
    *CACHE_PATH.lock().unwrap() = Some(p.to_path_buf());
}

/// Tune once per process (from the cache when it matches this device).
/// `None` when there is no GPU or tuning failed (defaults stay in place).
pub fn ensure(gpu: &Arc<Gpu>) -> Option<TuneResult> {
    RESULT
        .get_or_init(|| {
            let path = CACHE_PATH.lock().unwrap().clone().or_else(|| std::env::var_os("PB_GPU_TUNE").map(PathBuf::from));
            let path = path?;
            let key = device_key(gpu);
            if let Some(r) = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice::<TuneResult>(&b).ok()) {
                if r.device == key && r.v == VERSION {
                    install(&r);
                    return Some(r);
                }
            }
            match autotune(gpu) {
                Ok(r) => {
                    if let Ok(j) = serde_json::to_vec_pretty(&r) {
                        let _ = std::fs::write(&path, j);
                    }
                    Some(r)
                }
                Err(e) => {
                    eprintln!("pb-gpu: tuning failed ({e}); keeping default tiles");
                    None
                }
            }
        })
        .clone()
}

fn install(r: &TuneResult) {
    let mut t = tiles();
    for (i, v) in r.tiles.iter().enumerate() {
        let c = Tile::new(v[0], v[1], v[2], v[3]);
        if c.valid(WIDTHS[i]) {
            t[i] = c;
        }
    }
    set_tiles(t);
}
