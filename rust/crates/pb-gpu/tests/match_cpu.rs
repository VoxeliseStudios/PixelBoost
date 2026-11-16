//! The GPU kernels must reproduce pb-nn's CPU networks.

use pb_gpu::{Gpu, GpuNet};
use pb_nn::fmap::Fmap;
use pb_nn::pbm::{Header, Weights};
use pb_nn::pth::TensorF32;
use std::collections::HashMap;

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
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

fn rrdb(scale: usize, blocks: usize) -> Weights {
    let mut r = Rng(0x9e37_79b9_7f4a_7c15 ^ scale as u64);
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

fn srvgg(n: usize) -> Weights {
    let mut r = Rng(42);
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

fn image(h: usize, w: usize, seed: u64) -> Fmap {
    let mut r = Rng(seed);
    let mut x = Fmap::new(h, w, 3);
    for y in 0..h {
        for xx in 0..w {
            for v in x.px_mut(y, xx) {
                *v = r.f() + 0.5;
            }
        }
    }
    x
}

fn compare(w: &Weights, shapes: &[(usize, usize)]) {
    let Some(gpu) = Gpu::get() else {
        eprintln!("no GPU adapter: skipping");
        return;
    };
    let cpu = pb_nn::build(w).unwrap();
    let w2 = w.tensors.clone();
    let hdr = w.header.clone();
    let net = GpuNet::new(gpu, w, Box::new(move || pb_nn::build(&Weights { header: hdr.clone(), tensors: w2.clone() }))).unwrap();
    // Several shapes in a row exercise buffer reuse, growth and re-zeroing.
    for (i, &(h, wd)) in shapes.iter().enumerate() {
        let x = image(h, wd, 7 + i as u64);
        let a = cpu.forward(&x);
        let b = pb_nn::SrNetwork::forward(&net, &x);
        assert_eq!((a.h, a.w, a.c), (b.h, b.w, b.c));
        let mut worst = 0.0f32;
        for y in 0..a.h {
            for xx in 0..a.w {
                for (p, q) in a.px(y, xx).iter().zip(b.px(y, xx)) {
                    if (p - q).abs() > 1e-2 && std::env::var("PB_DBG").is_ok() {
                        eprintln!("({y},{xx}) cpu {:?} gpu {:?}", a.px(y, xx), b.px(y, xx));
                    }
                    worst = worst.max((p - q).abs());
                }
            }
        }
        eprintln!("{} {h}x{wd}: max |cpu - gpu| = {worst:e} ({})", w.header.arch, pb_nn::SrNetwork::backend(&net));
        assert!(worst < 2e-3, "{h}x{wd}: GPU differs from CPU by {worst}");
    }
}

#[test]
fn rrdb_x4_matches_cpu() {
    compare(&rrdb(4, 2), &[(13, 17), (24, 24), (9, 30), (24, 24)]);
}

#[test]
fn rrdb_x2_matches_cpu() {
    compare(&rrdb(2, 1), &[(14, 18), (20, 10)]);
}

#[test]
fn srvgg_matches_cpu() {
    compare(&srvgg(4), &[(11, 19), (32, 32), (11, 19)]);
}

#[test]
fn real_models_match_cpu() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../Models/pbnn");
    for f in ["fast-x4.pbnn", "anime-x4.pbnn", "photo-x4.pbnn"] {
        let Ok(bytes) = std::fs::read(dir.join(f)) else {
            eprintln!("{f} not found: skipping");
            continue;
        };
        let w = pb_nn::pbm::read(&bytes).unwrap();
        compare(&w, &[(40, 36)]);
    }
}

#[test]
fn srvgg_minimal_matches_cpu() {
    compare(&srvgg(0), &[(4, 5)]);
}
