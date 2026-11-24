//! Telemetry + model downloads against a live bridge (scratch DB):
//! set PB_TEST_SERVER (and PB_TEST_KEY); the bridge needs MODEL_FILES_DIR
//! pointing at Models/downloads.
use pixelboost_core::downloads::{ModelProgress, ModelStore};
use pixelboost_core::server::ServerClient;
use pixelboost_core::telemetry::Telemetry;
use std::sync::Arc;

struct P;
impl ModelProgress for P {
    fn on_progress(&self, _r: u64, _t: u64) {}
}

#[test]
fn telemetry_and_downloads() {
    let Ok(base) = std::env::var("PB_TEST_SERVER") else { return };
    let key = std::env::var("PB_TEST_KEY").unwrap_or_default();
    let dev = format!("pb-tm-test-{}", std::process::id());
    let c = ServerClient::new(base.clone(), key.clone(), dev.clone());
    let dir = std::env::temp_dir().join(format!("pbtm-e2e-{}", std::process::id()));
    let t = Telemetry::new(dir.join("tm").to_string_lossy().into(), dev.clone(), "5.7.0".into(), "26.1".into(), "iPhone17,1".into(), true);
    t.set_context("nominal".into(), false, 0.8, 8192);
    let run = t.new_run_id();
    pb_core::telemetry::event("upscale", "run").run(&run).ms(1234.0).tag("model", "photo").tag("outcome", "ok").metric("net_ms", 1000.0).metric("gflop", 50.0).emit();
    t.feedback(run.clone(), "kept".into(), 1.0, "{}".into());
    t.feedback(run.clone(), "rating".into(), 1.0, "{}".into());
    for i in 0..70 {
        pb_core::telemetry::sample("editor", "preview", "adjust", i as f64);
    }
    let sent = t.flush(c.clone()).unwrap();
    assert!(sent >= 4, "sent {sent}");
    assert_eq!(t.pending(), 0);
    let agent = ureq::agent();
    let get = |path: &str| -> serde_json::Value {
        let mut r = agent.get(&format!("{base}{path}"));
        if !key.is_empty() {
            r = r.set("Authorization", &format!("Bearer {key}"));
        }
        r.call().unwrap().into_json().unwrap()
    };
    let ev = get(&format!("/log/telemetry?device_id={dev}&limit=100"));
    let evs = ev["events"].as_array().unwrap();
    let up = evs.iter().find(|e| e["system"] == "upscale").expect("upscale event stored");
    assert_eq!(up["run_id"], run.as_str());
    assert_eq!(up["metrics"]["dev.memory_mb"], 8192.0);
    assert_eq!(up["device_model"], "iPhone17,1");
    assert!(evs.iter().any(|e| e["system"] == "editor" && e["name"] == "preview" && e["metrics"]["ms.n"] == 64.0));
    let fb = get("/log/telemetry/feedback?by=model&days=1");
    assert!(fb["groups"].as_array().unwrap().iter().any(|g| g["grp"] == "photo" && g["feedback"] == "kept"), "{fb}");
    let sm = get("/log/telemetry/summary?system=upscale&days=1");
    assert!(sm["metrics"].as_array().unwrap().iter().any(|m| m["key"] == "net_ms"));
    // Opt-in example upload.
    let img = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testimages/src/g4249.jpg");
    let sid = t.send_sample(c.clone(), run.clone(), "great_result".into(), img.into(), img.into(), "test".into()).unwrap();
    assert!(!sid.is_empty());
    let list = get("/log/telemetry/samples?limit=5");
    assert!(list["samples"].as_array().unwrap().iter().any(|s| s["id"] == sid.as_str()));
    // Model download: interrupted part file + resume + checksum.
    let store = ModelStore::new(dir.join("models").to_string_lossy().into());
    assert!(!store.installed("native2x".into()));
    let src = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../Models/downloads/native-x2.pbnn");
    let bytes = std::fs::read(src).unwrap();
    std::fs::write(dir.join("models").join("native-x2.pbnn.part"), &bytes[..1_000_000]).unwrap();
    assert_eq!(store.list().iter().find(|m| m.id == "native2x").unwrap().partial, 1_000_000);
    store.download(c.clone(), "native2x".into(), Arc::new(P)).unwrap();
    assert!(store.installed("native2x".into()));
    assert_eq!(std::fs::read(dir.join("models").join("native-x2.pbnn")).unwrap(), bytes);
    assert!(pb_core::locate(std::path::Path::new("/nonexistent"), "native-x2.pbnn").is_some());
    store.remove("native2x".into()).unwrap();
    let _ = std::fs::remove_dir_all(dir);
    println!("telemetry + downloads ok ({sent} events)");
}
