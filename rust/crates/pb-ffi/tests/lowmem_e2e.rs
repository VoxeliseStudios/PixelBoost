//! The 2026-11-24 crash: 4096² at 4x on a 4 GB iPhone. Runs the app's
//! whole path (upscale, live preview, history thumbnail) with the memory a
//! 4 GB phone has left (`PB_AVAILABLE_MB`) and checks the process peak
//! stays inside it. Slow (a real 4096² upscale): run on demand with
//! `cargo test --release -p pb-ffi --test lowmem_e2e -- --ignored --nocapture`
//! and `PB_LOWMEM_INPUT=<4096² image>` (a gradient is made otherwise).
use pixelboost_core::history::HistoryStore;
use pixelboost_core::telemetry::Telemetry;
use pixelboost_core::{CancelToken, PixelBoostCore, UpscaleListener, UpscaleSettings};
use std::sync::Arc;

struct L;
impl UpscaleListener for L {
    fn on_tile(&self, _d: u32, _t: u32, _x: u32, _y: u32, _w: u32, _h: u32, _pw: u32, _ph: u32, _rgba: Vec<u8>) {}
}

fn hwm_mb() -> f64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    s.lines().find(|l| l.starts_with("VmHWM:")).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0) / 1024.0
}

#[test]
#[ignore]
fn upscale_4096_x4_fits_a_4gb_phone() {
    // What a 4 GB iPhone has left for the app at the start of a run.
    let avail = 1700.0;
    std::env::set_var("PB_AVAILABLE_MB", avail.to_string());
    let dir = std::env::temp_dir().join(format!("pb-lowmem-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = std::env::var("PB_LOWMEM_INPUT").unwrap_or_else(|_| {
        let p = dir.join("in.png");
        let mut r = pb_core::stream::Rgba8::new(4096, 4096);
        for (i, px) in r.data.chunks_exact_mut(4).enumerate() {
            let (x, y) = (i % 4096, i / 4096);
            px.copy_from_slice(&[(x / 16) as u8, (y / 16) as u8, ((x ^ y) & 255) as u8, 255]);
        }
        r.save(p.to_str().unwrap(), 95).unwrap();
        p.to_string_lossy().into()
    });
    let tm = Telemetry::new(dir.join("tm").to_string_lossy().into(), "lowmem-test".into(), "5.8.1".into(), "26.1".into(), "iPhone14,5".into(), true);
    let models = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../Models/pbnn");
    let core = PixelBoostCore::new(models.into());
    let base = hwm_mb();
    let s = UpscaleSettings {
        model: "lineart".into(),
        scale: 4.0,
        fidelity: "natural".into(),
        detail: 0.2,
        denoise: 0.0,
        ensemble: 1,
        tile: 128,
        max_output_mp: 120.0,
        realism: 1.0,
        adaptive: false,
        prescale: 1.0,
        dehalo: 0.0,
    };
    let out = dir.join("out.png").to_string_lossy().to_string();
    assert!(!dir.join("tm").join("inflight.json").exists());
    let o = core.upscale(input.clone(), out.clone(), s, Arc::new(L), CancelToken::new()).unwrap();
    let after_run = hwm_mb();
    let prev = core.load_preview(o.output_path.clone(), 2048).unwrap();
    let hist = HistoryStore::new(dir.join("hist").to_string_lossy().into());
    let _ = hist.add(
        pixelboost_core::history::HistoryEntry { id: String::new(), created_at: 0.0, model: o.model.clone(), scale: 4.0, in_width: 4096, in_height: 4096, out_width: o.width, out_height: o.height, seconds: o.seconds, fidelity_db: o.fidelity_db, result_path: String::new(), thumb_path: String::new() },
        o.output_path.clone(),
    ).unwrap();
    let after_all = hwm_mb();
    let mp = (o.width as f64 * o.height as f64) / 1e6;
    eprintln!("out {}x{} ({mp:.0} MP, capped {}), preview {}x{}, peak: start {base:.0} MB, run {after_run:.0} MB, after preview+history {after_all:.0} MB", o.width, o.height, o.capped, prev.width, prev.height);
    assert!(o.capped && mp < 120.0, "memory cap should apply below 120 MP");
    assert!(after_all - base < avail * 0.75, "job used {:.0} MB of {avail} MB", after_all - base);
    assert!(!dir.join("tm").join("inflight.json").exists(), "marker left behind");
    drop(tm);
    std::fs::remove_dir_all(&dir).ok();
}
