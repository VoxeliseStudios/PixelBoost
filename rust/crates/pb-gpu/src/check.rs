//! GPU self-check and diagnostics.
//!
//! A GPU is only used after it reproduces the CPU networks on a small
//! random RRDB (dense blocks, residuals, upsample-in-conv, scalar and vec4
//! loaders) and SRVGG (PReLU, pixel-shuffle tail). A driver or compiler bug
//! then costs speed (CPU fallback), never a wrong picture.

use crate::net::GpuNet;
use crate::Gpu;
use pb_nn::conv::{conv3x3, Conv3x3, Epi};
use pb_nn::fmap::{upsample_nearest2x, Fmap};
use pb_nn::pbm::{Header, Weights};
use pb_nn::pth::TensorF32;
use std::collections::HashMap;
use std::sync::Arc;

pub struct Rng(pub u64);
impl Rng {
    pub fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
}

fn conv(t: &mut HashMap<String, TensorF32>, r: &mut Rng, name: &str, cout: usize, cin: usize) {
    let k = 1.6 / ((cin * 9) as f32).sqrt();
    t.insert(format!("{name}.weight"), TensorF32 { shape: vec![cout, cin, 3, 3], data: (0..cout * cin * 9).map(|_| r.f() * k).collect() });
    t.insert(format!("{name}.bias"), TensorF32 { shape: vec![cout], data: (0..cout).map(|_| r.f() * 0.1).collect() });
}

fn weights(arch: &str, config: serde_json::Value, tensors: HashMap<String, TensorF32>) -> Weights {
    Weights { header: Header { arch: arch.into(), config, meta: serde_json::json!({}), tensors: vec![] }, tensors }
}

/// Random RRDBNet weights (`blocks` RRDBs, 64 features, 32 growth).
pub fn rrdb(scale: usize, blocks: usize, seed: u64) -> Weights {
    let mut r = Rng(0x9e37_79b9_7f4a_7c15 ^ seed ^ scale as u64);
    let mut t = HashMap::new();
    let (f, g) = (64, 32);
    let inc = 3 * (4 / scale) * (4 / scale);
    conv(&mut t, &mut r, "conv_first", f, inc);
    for i in 0..blocks {
        for rd in 1..=3 {
            for k in 0..4 {
                conv(&mut t, &mut r, &format!("body.{i}.rdb{rd}.conv{}", k + 1), g, f + k * g);
            }
            conv(&mut t, &mut r, &format!("body.{i}.rdb{rd}.conv5"), f, f + 4 * g);
        }
    }
    for n in ["conv_body", "conv_up1", "conv_up2", "conv_hr"] {
        conv(&mut t, &mut r, n, f, f);
    }
    conv(&mut t, &mut r, "conv_last", 3, f);
    weights("rrdb", serde_json::json!({"num_block": blocks, "scale": scale, "num_feat": f, "num_grow": g}), t)
}

/// Random SRVGGNetCompact weights (`n` body convs).
pub fn srvgg(n: usize, seed: u64) -> Weights {
    let mut r = Rng(42 ^ seed);
    let mut t = HashMap::new();
    let f = 64;
    conv(&mut t, &mut r, "body.0", f, 3);
    t.insert("body.1.weight".into(), TensorF32 { shape: vec![f], data: (0..f).map(|_| r.f() * 0.5).collect() });
    for i in 1..=n {
        conv(&mut t, &mut r, &format!("body.{}", i * 2), f, f);
        t.insert(format!("body.{}.weight", i * 2 + 1), TensorF32 { shape: vec![f], data: (0..f).map(|_| r.f() * 0.5).collect() });
    }
    conv(&mut t, &mut r, &format!("body.{}", (n + 1) * 2), 48, f);
    weights("srvgg", serde_json::json!({"num_conv": n, "scale": 4, "num_feat": f}), t)
}

pub fn image(h: usize, w: usize, c: usize, seed: u64) -> Fmap {
    let mut r = Rng(seed | 1);
    let mut x = Fmap::new(h, w, c);
    for y in 0..h {
        for xx in 0..w {
            for v in x.px_mut(y, xx) {
                *v = r.f() + 0.5;
            }
        }
    }
    x
}

/// Largest |a - b| over the pixels (NaN counts as infinite).
pub fn max_diff(a: &Fmap, b: &Fmap) -> f32 {
    if (a.h, a.w, a.c) != (b.h, b.w, b.c) {
        return f32::INFINITY;
    }
    let mut worst = 0.0f32;
    for y in 0..a.h {
        for x in 0..a.w {
            for (p, q) in a.px(y, x).iter().zip(b.px(y, x)) {
                let d = (p - q).abs();
                worst = if d.is_nan() { f32::INFINITY } else { worst.max(d) };
            }
        }
    }
    worst
}

fn net_case(gpu: &Arc<Gpu>, w: &Weights, h: usize, wd: usize) -> Result<f32, String> {
    let cpu = pb_nn::build(w)?;
    let net = GpuNet::new(gpu.clone(), w, Box::new(|| Err("self-check".into())))?;
    let x = image(h, wd, 3, 11);
    let x2 = if w.header.arch == "rrdb" && w.cfg("scale") == Some(2) { pb_nn::fmap::pixel_unshuffle(&x, 2) } else { x.clone() };
    let g = net.try_forward(&x2)?;
    Ok(max_diff(&cpu.forward(&x), &g))
}

/// Run the self-check: `Ok(())` when the GPU reproduces the CPU.
pub fn self_check(gpu: &Arc<Gpu>) -> Result<(), String> {
    for (name, w, h, wd) in [("rrdb x4", rrdb(4, 1, 1), 10, 13), ("rrdb x2", rrdb(2, 1, 2), 10, 12), ("srvgg", srvgg(2, 3), 9, 11)] {
        let d = net_case(gpu, &w, h, wd).map_err(|e| format!("{name}: {e}"))?;
        // f16 storage rounds every feature map to ~3 significant digits.
        let tol = if crate::shaders::half() { 1e-2 } else { 2e-3 };
        if !(d < tol) {
            return Err(format!("{name}: GPU differs from CPU by {d:e}"));
        }
    }
    Ok(())
}

/// Per-kernel diagnosis (single convs in every configuration class), for
/// finding which part breaks on a new GPU. One line per case.
pub fn diagnose(gpu: &Arc<Gpu>) -> Vec<String> {
    let mut out = vec![format!("device: {} | {} {}", gpu.describe(), gpu.info.driver, gpu.info.driver_info)];
    let mut r = Rng(5);
    // (cin, cout, up, zero weights / unit bias)
    let cases = [(4, 16, false, true), (3, 64, false, false), (4, 64, false, false), (64, 64, false, false), (64, 32, false, false), (64, 48, false, false), (64, 3, false, false), (64, 64, true, false), (12, 64, false, false)];
    for (cin, cout, up, unit) in cases {
        let w: Vec<f32> = (0..cout * cin * 9).map(|_| if unit { 0.0 } else { r.f() * 0.2 }).collect();
        let b: Vec<f32> = (0..cout).map(|_| if unit { 1.0 } else { r.f() * 0.1 }).collect();
        let x = image(7, 9, cin, 3);
        let src = if up { upsample_nearest2x(&x, cin) } else { x.clone() };
        let mut want = Fmap::new(src.h, src.w, cout);
        conv3x3(&src, 0, &Conv3x3::from_torch(&w, Some(&b), cout, cin), &mut want, 0, Epi::PLAIN);
        let line = match GpuNet::single_conv(gpu.clone(), &w, &b, cout, cin, up).and_then(|n| n.try_forward(&x)) {
            Ok(g) => {
                let sample: Vec<f32> = g.px(0, 0).iter().take(4).copied().collect();
                format!("conv {cin:>3}->{cout:<3} up={up:<5} unit={unit:<5} max diff {:e}  gpu px0 {:?} cpu px0 {:?}", max_diff(&want, &g), sample, &want.px(0, 0)[..4.min(cout)])
            }
            Err(e) => format!("conv {cin}->{cout} up={up}: error {e}"),
        };
        out.push(line);
    }
    for (name, w, h, wd) in [("rrdb x4", rrdb(4, 1, 1), 10, 13), ("srvgg", srvgg(2, 3), 9, 11)] {
        out.push(match net_case(gpu, &w, h, wd) {
            Ok(d) => format!("{name}: max diff {d:e}"),
            Err(e) => format!("{name}: error {e}"),
        });
    }
    out
}
