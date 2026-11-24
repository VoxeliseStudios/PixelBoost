//! The app's telemetry pipe: every engine event (pb_nn::telemetry) plus the
//! app's own UI feedback goes into an on-disk queue (Caches/telemetry,
//! JSON lines) and is uploaded to the PixelBoost server in batches.
//!
//! - Measurements and settings only. No pixels, file names or text.
//! - The user can turn it off (Settings › Telemetry); the queue is then
//!   dropped and nothing is recorded.
//! - Image samples (small crops of a source + result) are a separate,
//!   explicit "Send this example" action per result; never automatic.
//! - Ambient device state (thermal, low power, battery, memory) is attached
//!   to every event because speed numbers mean little without it.

use crate::server::ServerClient;
use crate::PbError;
use pb_core::telemetry::{self as tm, Event, Sink};
use pb_core::Image;
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const INFLIGHT_FILE: &str = "inflight.json";
static INFLIGHT: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// Marks a run as in progress (numbers become metrics, strings tags of the
/// `upscale/killed` event the next launch sends if it never finished).
pub(crate) fn inflight_begin(v: &Value) {
    if let Some(p) = INFLIGHT.get() {
        let _ = std::fs::write(p, v.to_string());
    }
}

/// The run ended (any outcome).
pub(crate) fn inflight_end() {
    if let Some(p) = INFLIGHT.get() {
        let _ = std::fs::remove_file(p);
    }
}

/// Queue file larger than this drops its oldest half (offline for weeks).
const MAX_QUEUE_BYTES: u64 = 6 << 20;
const BATCH: usize = 400;

#[derive(Default, Clone)]
struct Ctx {
    thermal: String,
    low_power: bool,
    battery: f32,
    memory_mb: u32,
}

struct Queue {
    dir: PathBuf,
    file: Mutex<Option<std::fs::File>>,
    on: AtomicBool,
    ctx: Mutex<Ctx>,
    n: AtomicU64,
}

impl Queue {
    fn path(&self) -> PathBuf {
        self.dir.join("queue.jsonl")
    }

    fn append(&self, e: &Event) {
        let Ok(mut line) = serde_json::to_vec(e) else { return };
        line.push(b'\n');
        let mut g = self.file.lock().unwrap();
        if g.is_none() {
            *g = std::fs::OpenOptions::new().create(true).append(true).open(self.path()).ok();
        }
        if let Some(f) = g.as_mut() {
            if f.write_all(&line).is_ok() {
                self.n.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

impl Sink for Queue {
    fn record(&self, mut e: Event) {
        if !self.on.load(Ordering::Relaxed) {
            return;
        }
        let c = self.ctx.lock().unwrap().clone();
        if !c.thermal.is_empty() {
            e.tags.insert("dev.thermal".into(), c.thermal);
        }
        e.metrics.insert("dev.low_power".into(), if c.low_power { 1.0 } else { 0.0 });
        if c.battery >= 0.0 {
            e.metrics.insert("dev.battery".into(), (c.battery * 100.0).round() as f64 / 100.0);
        }
        if c.memory_mb > 0 {
            e.metrics.insert("dev.memory_mb".into(), c.memory_mb as f64);
        }
        self.append(&e);
    }
}

#[derive(uniffi::Object)]
pub struct Telemetry {
    q: Arc<Queue>,
    device_id: String,
    session_id: String,
    app_version: String,
    os_version: String,
    device_model: String,
    sending: Mutex<()>,
}

#[uniffi::export]
impl Telemetry {
    /// Opens the queue in `dir` and installs it as the engine's sink.
    #[uniffi::constructor]
    pub fn new(dir: String, device_id: String, app_version: String, os_version: String, device_model: String, enabled: bool) -> Arc<Self> {
        let dir = PathBuf::from(dir);
        let _ = std::fs::create_dir_all(&dir);
        let q = Arc::new(Queue { dir, file: Mutex::new(None), on: AtomicBool::new(enabled), ctx: Mutex::new(Ctx { battery: -1.0, ..Default::default() }), n: AtomicU64::new(0) });
        let t = Arc::new(Telemetry { q: q.clone(), device_id, session_id: tm::new_run_id(), app_version, os_version, device_model, sending: Mutex::new(()) });
        if enabled {
            tm::set_sink(Some(q));
        }
        tm::event("app", "session").tag("kernel", pb_nn::conv::kernel_name()).emit();
        let marker = t.q.dir.join(INFLIGHT_FILE);
        let _ = INFLIGHT.set(marker.clone());
        // A run that was still going when the process died: iOS killed it
        // (memory, most likely) or it crashed. Report what it was doing.
        if let Ok(b) = std::fs::read(&marker) {
            if enabled {
                if let Ok(Value::Object(m)) = serde_json::from_slice::<Value>(&b) {
                    let run = m.get("run").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let mut e = tm::event("upscale", "killed").run(&run);
                    for (k, v) in &m {
                        match v {
                            Value::Number(n) => e = e.metric(k, n.as_f64().unwrap_or(0.0)),
                            Value::String(s) if k != "run" => e = e.tag(k, s),
                            _ => {}
                        }
                    }
                    e.emit();
                }
            }
            let _ = std::fs::remove_file(&marker);
        }
        t
    }

    /// Settings › Telemetry. Off also deletes everything still queued.
    pub fn set_enabled(&self, on: bool) {
        self.q.on.store(on, Ordering::Relaxed);
        if on {
            tm::set_sink(Some(self.q.clone()));
        } else {
            tm::set_sink(None);
            *self.q.file.lock().unwrap() = None;
            let _ = std::fs::remove_file(self.q.path());
        }
    }

    pub fn enabled(&self) -> bool {
        self.q.on.load(Ordering::Relaxed)
    }

    /// Ambient device state from the platform (ProcessInfo / UIDevice):
    /// "nominal" | "fair" | "serious" | "critical", Low Power Mode, battery
    /// 0..1 (-1 unknown), physical memory.
    pub fn set_context(&self, thermal: String, low_power: bool, battery: f32, memory_mb: u32) {
        *self.q.ctx.lock().unwrap() = Ctx { thermal, low_power, battery, memory_mb };
    }

    /// A new run id (Swift keeps it to send feedback about that run).
    pub fn new_run_id(&self) -> String {
        tm::new_run_id()
    }

    /// An app-side event: `metrics_json` = {"name": number}, `tags_json` =
    /// {"name": "short label"} (either may be "").
    pub fn track(&self, system: String, name: String, run: String, metrics_json: String, tags_json: String) {
        let mut b = tm::event(&system, &name).run(&run);
        if let Ok(v) = serde_json::from_str::<Value>(&metrics_json) {
            b = b.json("", &v);
        }
        if let Ok(Value::Object(m)) = serde_json::from_str::<Value>(&tags_json) {
            for (k, v) in m {
                if let Some(s) = v.as_str() {
                    b = b.tag(&k, s);
                }
            }
        }
        b.emit();
    }

    /// What the user did with a result — the ground truth every other
    /// number gets judged against. `kind`: "kept" (saved / exported /
    /// replaced), "discarded", "rating" (value -1 / 1), "rerun" (ran again
    /// with other settings), "undo", "override" (changed an Auto choice;
    /// detail {"field", "auto", "user"}), "compare_pick" (detail
    /// {"picked", "candidates"}), "mask_fix" (cutout brush corrections;
    /// value = share of the mask repainted).
    pub fn feedback(&self, run: String, kind: String, value: f64, detail_json: String) {
        let mut b = tm::event("feedback", &kind).run(&run).metric("value", value);
        if let Ok(v) = serde_json::from_str::<Value>(&detail_json) {
            b = b.json("d", &v);
        }
        b.emit();
    }

    /// Events waiting for upload (this session's count + what's on disk).
    pub fn pending(&self) -> u64 {
        std::fs::read(self.q.path()).map(|b| b.iter().filter(|c| **c == b'\n').count() as u64).unwrap_or(0)
    }

    /// Upload everything queued. Returns the number of events sent; events
    /// stay queued if the server can't be reached.
    pub fn flush(&self, server: Arc<ServerClient>) -> Result<u32, PbError> {
        tm::flush_aggregates();
        let _one = self.sending.lock().unwrap();
        let path = self.q.path();
        let sending = self.q.dir.join("sending.jsonl");
        // A previous failed upload goes first.
        if !sending.exists() {
            let mut g = self.q.file.lock().unwrap();
            *g = None;
            if !path.exists() {
                return Ok(0);
            }
            std::fs::rename(&path, &sending).map_err(|e| PbError::from(e.to_string()))?;
        }
        let f = std::fs::File::open(&sending).map_err(|e| PbError::from(e.to_string()))?;
        let events: Vec<Value> = std::io::BufReader::new(f).lines().map_while(Result::ok).filter_map(|l| serde_json::from_str(&l).ok()).collect();
        let mut sent = 0u32;
        for chunk in events.chunks(BATCH) {
            let body = json!({
                "device_id": self.device_id,
                "session_id": self.session_id,
                "app_version": self.app_version,
                "os_version": self.os_version,
                "device_model": self.device_model,
                "events": chunk,
            });
            match server.post_json_now("/log/telemetry", &body) {
                Ok(_) => sent += chunk.len() as u32,
                Err(e) => {
                    // Keep what wasn't sent.
                    let rest: Vec<String> = events[sent as usize..].iter().map(|v| v.to_string()).collect();
                    let _ = std::fs::write(&sending, rest.join("\n") + "\n");
                    trim(&sending);
                    return if sent > 0 { Ok(sent) } else { Err(e) };
                }
            }
        }
        let _ = std::fs::remove_file(&sending);
        Ok(sent)
    }

    /// "Send this example": the user explicitly shares one result to help
    /// tune PixelBoost. Rust cuts a matching 256 px crop from the source and
    /// the result (the most detailed spot), encodes JPEG q90 and uploads
    /// them with the run id and the reason ("bad_result", "great_result",
    /// "bad_cutout", ...). Returns the sample id.
    pub fn send_sample(&self, server: Arc<ServerClient>, run: String, reason: String, source_path: String, result_path: String, note: String) -> Result<String, PbError> {
        let src = Image::open(&source_path)?;
        // 8-bit (4 B/px) instead of f32: results can be 100+ MP.
        let res = pb_core::strips::open_rgba8(&result_path)?;
        let (sc, rc) = sample_crops(&src, &res);
        let sj = sc.encode(pb_core::image::Format::Jpeg, 90)?;
        let rj = rc.encode(pb_core::image::Format::Jpeg, 90)?;
        // One body: [u32 len][source jpeg][result jpeg].
        let mut body = Vec::with_capacity(sj.len() + rj.len() + 4);
        body.extend_from_slice(&(sj.len() as u32).to_le_bytes());
        body.extend_from_slice(&sj);
        body.extend_from_slice(&rj);
        let note: String = note.chars().take(500).collect();
        let v = server.post_bytes_now(
            "/log/telemetry/sample",
            &[("device_id", &self.device_id), ("run", &run), ("reason", &reason), ("app_version", &self.app_version), ("device_model", &self.device_model), ("note", &note)],
            "application/octet-stream",
            &body,
        )?;
        tm::event("feedback", "sample").run(&run).tag("reason", &reason).emit();
        Ok(v.get("id").and_then(|i| i.as_str()).unwrap_or("").to_string())
    }
}

/// Keep a queue file under the size cap by dropping its oldest lines.
fn trim(p: &std::path::Path) {
    let Ok(m) = std::fs::metadata(p) else { return };
    if m.len() <= MAX_QUEUE_BYTES {
        return;
    }
    if let Ok(s) = std::fs::read_to_string(p) {
        let lines: Vec<&str> = s.lines().collect();
        let keep = &lines[lines.len() / 2..];
        let _ = std::fs::write(p, keep.join("\n") + "\n");
    }
}

/// Matching crops: the 256 px source window with the most fine detail and
/// the same area of the result (scaled), result crop capped at 1024 px.
pub(crate) fn sample_crops(src: &Image, res: &dyn pb_core::stream::Source) -> (Image, Image) {
    let (res_w, res_h) = res.size();
    let n = 256usize.min(src.w).min(src.h).max(1);
    let (mut best, mut bx, mut by) = (-1.0f32, 0usize, 0usize);
    let step = (n / 2).max(1);
    let mut y = 0;
    while y + n <= src.h {
        let mut x = 0;
        while x + n <= src.w {
            let mut e = 0.0f32;
            for yy in (y + 1..y + n).step_by(4) {
                for xx in (x + 1..x + n).step_by(4) {
                    e += (Image::luma(src.at(xx, yy)) - Image::luma(src.at(xx - 1, yy))).abs() + (Image::luma(src.at(xx, yy)) - Image::luma(src.at(xx, yy - 1))).abs();
                }
            }
            if e > best {
                (best, bx, by) = (e, x, y);
            }
            x += step;
        }
        y += step;
    }
    let sc = src.crop(bx, by, n, n);
    let fx = res_w as f32 / src.w.max(1) as f32;
    let fy = res_h as f32 / src.h.max(1) as f32;
    let (rx, ry) = ((bx as f32 * fx) as usize, (by as f32 * fy) as usize);
    let (rw, rh) = (((n as f32 * fx) as usize).min(res_w - rx.min(res_w)).max(1), ((n as f32 * fy) as usize).min(res_h - ry.min(res_h)).max(1));
    let mut rc = res.region(rx.min(res_w - 1), ry.min(res_h - 1), rw, rh);
    if rc.w > 1024 {
        let h = (rc.h * 1024 / rc.w).max(1);
        rc = pb_core::resample::resize(&rc, 1024, h, pb_core::resample::Filter::Lanczos3);
    }
    (sc, rc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_records_and_context_is_attached() {
        let dir = std::env::temp_dir().join(format!("pbtm-{}", tm::new_run_id()));
        let t = Telemetry::new(dir.to_string_lossy().into(), "dev".into(), "5.7.0".into(), "26.1".into(), "iPhone".into(), true);
        t.set_context("fair".into(), true, 0.5, 6000);
        t.feedback("r1".into(), "rating".into(), 1.0, r#"{"field":"model"}"#.into());
        t.track("ui".into(), "open".into(), "".into(), r#"{"x":2}"#.into(), r#"{"screen":"studio"}"#.into());
        // Other tests may record too; ours are in there with the context.
        let s = std::fs::read_to_string(dir.join("queue.jsonl")).unwrap();
        let evs: Vec<Event> = s.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        let fb = evs.iter().find(|e| e.system == "feedback" && e.run == "r1").unwrap();
        assert_eq!(fb.tags["dev.thermal"], "fair");
        assert_eq!(fb.tags["d.field"], "model");
        assert_eq!(fb.metrics["dev.memory_mb"], 6000.0);
        assert!(evs.iter().any(|e| e.name == "open" && e.tags["screen"] == "studio" && e.metrics["x"] == 2.0));
        assert!(t.pending() >= 2);
        t.set_enabled(false);
        assert_eq!(t.pending(), 0);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn sample_crops_match() {
        let mut a = Image::new(600, 400);
        for (i, p) in a.px.iter_mut().enumerate() {
            let (x, y) = (i % 600, i / 600);
            let v = if x > 400 && y > 200 { ((x * 7 + y * 13) % 5) as f32 / 5.0 } else { 0.5 };
            *p = [v, v, v, 1.0];
        }
        let b = pb_core::resample::resize(&a, 2400, 1600, pb_core::resample::Filter::CatmullRom);
        let (sc, rc) = sample_crops(&a, &b);
        assert_eq!((sc.w, sc.h), (256, 256));
        assert_eq!((rc.w, rc.h), (1024, 1024));
    }
}
