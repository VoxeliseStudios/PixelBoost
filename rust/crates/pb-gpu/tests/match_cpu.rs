//! The GPU kernels must reproduce pb-nn's CPU networks.

use pb_gpu::check::{image, max_diff, rrdb, srvgg};
use pb_gpu::{Gpu, GpuNet};
use pb_nn::pbm::Weights;

fn gpu() -> Option<std::sync::Arc<Gpu>> {
    let g = Gpu::get();
    if g.is_none() {
        eprintln!("no usable GPU ({}): skipping", pb_gpu::unavailable_reason().unwrap_or_default());
    }
    g
}

fn compare(w: &Weights, shapes: &[(usize, usize)]) {
    let Some(gpu) = gpu() else { return };
    let cpu = pb_nn::build(w).unwrap();
    let w2 = w.tensors.clone();
    let hdr = w.header.clone();
    let net = GpuNet::new(gpu, w, Box::new(move || pb_nn::build(&Weights { header: hdr.clone(), tensors: w2.clone() }))).unwrap();
    // Several shapes in a row exercise buffer reuse, growth and re-zeroing.
    for (i, &(h, wd)) in shapes.iter().enumerate() {
        let x = image(h, wd, 3, 7 + i as u64);
        let a = cpu.forward(&x);
        let b = pb_nn::SrNetwork::forward(&net, &x);
        let worst = max_diff(&a, &b);
        eprintln!("{} {h}x{wd}: max |cpu - gpu| = {worst:e} ({})", w.header.arch, pb_nn::SrNetwork::backend(&net));
        assert!(worst < 2e-3, "{h}x{wd}: GPU differs from CPU by {worst}");
    }
}

#[test]
fn rrdb_x4_matches_cpu() {
    compare(&rrdb(4, 2, 0), &[(13, 17), (24, 24), (9, 30), (24, 24)]);
}

#[test]
fn rrdb_x2_matches_cpu() {
    compare(&rrdb(2, 1, 0), &[(14, 18), (20, 10)]);
}

#[test]
fn srvgg_matches_cpu() {
    compare(&srvgg(4, 0), &[(11, 19), (32, 32), (11, 19)]);
}

#[test]
fn srvgg_minimal_matches_cpu() {
    compare(&srvgg(0, 0), &[(4, 5)]);
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

/// `cargo test -p pb-gpu --test match_cpu diagnose -- --ignored --nocapture`
/// with PB_GPU_NOCHECK=1: per-kernel report for a GPU that fails.
#[test]
#[ignore]
fn diagnose() {
    let Some(gpu) = gpu() else { return };
    for l in pb_gpu::check::diagnose(&gpu) {
        println!("{l}");
    }
}
