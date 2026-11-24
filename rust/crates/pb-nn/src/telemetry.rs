//! Telemetry for every custom system (upscaler, analyzer, GPU engine,
//! editor, pixel art, cutout, healing, animation ...).
//!
//! The engine only *describes* what happened: numbers (`metrics`), short
//! labels (`tags`) and a duration, grouped by `system` + `name`. Where the
//! events go is decided by whoever installed a [`Sink`]: the app queues
//! them on disk and uploads batches to its server (pb-ffi `Telemetry`), the
//! CLI prints or saves them. Without a sink an
//! event costs one atomic load.
//!
//! Events never contain pixels, file names or anything typed by the user;
//! only measurements and settings. (Image samples are a separate, opt-in,
//! per-report upload — see pb-ffi.)
//!
//! Every run that the user can react to gets a `run` id, so feedback
//! (kept / discarded / rating / Auto overridden / compare pick) joins back
//! to the measurements that produced the result.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Event {
    /// Unix time (seconds, ms precision).
    pub ts: f64,
    /// "upscale", "analyze", "gpu", "editor", "pixelart", "cutout", ...
    pub system: String,
    pub name: String,
    /// Run this event belongs to (joins feedback to measurements).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub run: String,
    /// Wall time of the measured work, milliseconds (0 = not timed).
    #[serde(default)]
    pub ms: f64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metrics: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tags: BTreeMap<String, String>,
}

pub trait Sink: Send + Sync {
    fn record(&self, e: Event);
}

static ON: AtomicBool = AtomicBool::new(false);
static SINK: RwLock<Option<Arc<dyn Sink>>> = RwLock::new(None);

/// Install (or remove) the process-wide sink.
pub fn set_sink(s: Option<Arc<dyn Sink>>) {
    let on = s.is_some();
    *SINK.write().unwrap() = s;
    ON.store(on, Ordering::Release);
}

#[inline]
pub fn enabled() -> bool {
    ON.load(Ordering::Acquire)
}

pub fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as f64 / 1000.0).unwrap_or(0.0)
}

/// Short random id for runs (not a UUID: 16 hex chars is plenty to join on).
pub fn new_run_id() -> String {
    static N: AtomicU64 = AtomicU64::new(0);
    let t = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
    let mut x = t ^ N.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed) ^ (std::process::id() as u64).rotate_left(32);
    // splitmix64 finaliser
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    format!("{x:016x}")
}

thread_local! {
    static RUN: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

/// Run `f` with `run` as the current run id of this thread: events built
/// inside (on this thread) carry it unless they set their own.
pub fn scope<T>(run: &str, f: impl FnOnce() -> T) -> T {
    let prev = RUN.with(|r| std::mem::replace(&mut *r.borrow_mut(), run.to_string()));
    struct Restore(Option<String>);
    impl Drop for Restore {
        fn drop(&mut self) {
            if let Some(p) = self.0.take() {
                RUN.with(|r| *r.borrow_mut() = p);
            }
        }
    }
    let _g = Restore(Some(prev));
    f()
}

pub fn current_run() -> String {
    RUN.with(|r| r.borrow().clone())
}

pub fn emit(e: Event) {
    if !enabled() {
        return;
    }
    if let Some(s) = SINK.read().unwrap().as_ref() {
        s.record(e);
    }
}

/// Builder: `telemetry::event("upscale", "run").run(&id).ms(t).metric("tiles", 12.0).tag("backend", "gpu").emit()`.
pub struct Builder(Option<Event>);

pub fn event(system: &str, name: &str) -> Builder {
    if !enabled() {
        return Builder(None);
    }
    Builder(Some(Event { ts: now(), system: system.into(), name: name.into(), run: current_run(), ms: 0.0, metrics: BTreeMap::new(), tags: BTreeMap::new() }))
}

impl Builder {
    pub fn active(&self) -> bool {
        self.0.is_some()
    }
    pub fn run(mut self, id: &str) -> Self {
        if let Some(e) = &mut self.0 {
            e.run = id.into();
        }
        self
    }
    pub fn ms(mut self, ms: f64) -> Self {
        if let Some(e) = &mut self.0 {
            e.ms = ms;
        }
        self
    }
    pub fn since(self, t: Instant) -> Self {
        self.ms(t.elapsed().as_secs_f64() * 1000.0)
    }
    pub fn metric(mut self, k: &str, v: impl Into<f64>) -> Self {
        if let Some(e) = &mut self.0 {
            let v = v.into();
            if v.is_finite() {
                e.metrics.insert(k.into(), v);
            }
        }
        self
    }
    pub fn metric_opt(self, k: &str, v: Option<f64>) -> Self {
        match v {
            Some(v) => self.metric(k, v),
            None => self,
        }
    }
    pub fn flag(self, k: &str, v: bool) -> Self {
        self.metric(k, if v { 1.0 } else { 0.0 })
    }
    pub fn tag(mut self, k: &str, v: impl AsRef<str>) -> Self {
        if let Some(e) = &mut self.0 {
            let v = v.as_ref();
            // Tags are labels, not payloads.
            let v: String = v.chars().take(80).collect();
            e.tags.insert(k.into(), v);
        }
        self
    }
    /// Add every numeric / bool field of a JSON object as `prefix.key` metrics
    /// and short strings as tags (settings structs, analyzer reports).
    pub fn json(mut self, prefix: &str, v: &serde_json::Value) -> Self {
        if self.0.is_none() {
            return self;
        }
        if let serde_json::Value::Object(m) = v {
            for (k, x) in m {
                let key = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                self = match x {
                    serde_json::Value::Number(n) => match n.as_f64() {
                        Some(f) => self.metric(&key, f),
                        None => self,
                    },
                    serde_json::Value::Bool(b) => self.flag(&key, *b),
                    serde_json::Value::String(s) if s.len() <= 40 => self.tag(&key, s),
                    _ => self,
                };
            }
        }
        self
    }
    pub fn stats(self, k: &str, s: &Stats) -> Self {
        if s.n == 0 {
            return self;
        }
        self.metric(&format!("{k}.n"), s.n as f64)
            .metric(&format!("{k}.mean"), s.mean())
            .metric(&format!("{k}.min"), s.min)
            .metric(&format!("{k}.max"), s.max)
            .metric(&format!("{k}.p50"), s.quantile(0.5))
            .metric(&format!("{k}.p90"), s.quantile(0.9))
    }
    pub fn emit(self) {
        if let Some(e) = self.0 {
            emit(e);
        }
    }
    pub fn take(self) -> Option<Event> {
        self.0
    }
}

/// Times a block; emits on `finish` (or on drop if never finished, tagged
/// `aborted`, so panics / early returns still show up).
pub struct Span {
    b: Option<Builder>,
    t: Instant,
}

pub fn span(system: &str, name: &str) -> Span {
    Span { b: Some(event(system, name)), t: Instant::now() }
}

impl Span {
    pub fn active(&self) -> bool {
        self.b.as_ref().is_some_and(|b| b.active())
    }
    pub fn with(&mut self, f: impl FnOnce(Builder) -> Builder) {
        if let Some(b) = self.b.take() {
            self.b = Some(f(b));
        }
    }
    pub fn elapsed_ms(&self) -> f64 {
        self.t.elapsed().as_secs_f64() * 1000.0
    }
    pub fn finish(mut self) {
        if let Some(b) = self.b.take() {
            b.since(self.t).emit();
        }
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        if let Some(b) = self.b.take() {
            b.since(self.t).tag("outcome", "aborted").emit();
        }
    }
}

/// Small sample store for timing distributions (keeps every value; used for
/// per-tile timings, at most a few thousand per run).
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub n: usize,
    pub sum: f64,
    pub min: f64,
    pub max: f64,
    v: Vec<f64>,
}

impl Stats {
    pub fn add(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        if self.n == 0 {
            self.min = x;
            self.max = x;
        } else {
            self.min = self.min.min(x);
            self.max = self.max.max(x);
        }
        self.n += 1;
        self.sum += x;
        if self.v.len() < 8192 {
            self.v.push(x);
        }
    }
    pub fn mean(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.sum / self.n as f64
        }
    }
    pub fn quantile(&self, q: f64) -> f64 {
        if self.v.is_empty() {
            return 0.0;
        }
        let mut s = self.v.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[((s.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize]
    }
}

static AGG: Mutex<Option<std::collections::HashMap<(String, String, String), Stats>>> = Mutex::new(None);

/// High-frequency timings (live previews while a slider moves): kept as a
/// distribution per (system, name, key) and emitted as one summary event
/// every 64 samples or on [`flush_aggregates`].
pub fn sample(system: &str, name: &str, key: &str, ms: f64) {
    if !enabled() {
        return;
    }
    let full = {
        let mut g = AGG.lock().unwrap();
        let m = g.get_or_insert_with(Default::default);
        let k = (system.to_string(), name.to_string(), key.to_string());
        let st = m.entry(k.clone()).or_default();
        st.add(ms);
        if st.n >= 64 {
            m.remove(&k).map(|s| (k, s))
        } else {
            None
        }
    };
    if let Some(((sys, nm, key), st)) = full {
        event(&sys, &nm).tag("key", key).stats("ms", &st).emit();
    }
}

/// Emit every partly filled [`sample`] distribution (app going to the
/// background, CLI exit).
pub fn flush_aggregates() {
    let all = AGG.lock().unwrap().take().unwrap_or_default();
    for ((sys, nm, key), st) in all {
        event(&sys, &nm).tag("key", key).stats("ms", &st).emit();
    }
}

/// Collects events in memory (CLI, tests).
#[derive(Default)]
pub struct Collector {
    pub events: Mutex<Vec<Event>>,
}

impl Sink for Collector {
    fn record(&self, e: Event) {
        let mut v = self.events.lock().unwrap();
        if v.len() < 100_000 {
            v.push(e);
        }
    }
}

impl Collector {
    pub fn drain(&self) -> Vec<Event> {
        std::mem::take(&mut *self.events.lock().unwrap())
    }
}

/// Resident memory now and the process peak, in MB (phys footprint on Apple
/// platforms — the number iOS kills apps by; RSS / HWM on Linux).
pub fn memory_mb() -> (f64, f64) {
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
            let field = |k: &str| s.lines().find(|l| l.starts_with(k)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0) / 1024.0;
            return (field("VmRSS:"), field("VmHWM:"));
        }
        (0.0, 0.0)
    }
    #[cfg(any(target_os = "ios", target_os = "macos", target_os = "tvos"))]
    {
        apple::footprint()
    }
    #[cfg(not(any(target_os = "linux", target_os = "ios", target_os = "macos", target_os = "tvos")))]
    {
        (0.0, 0.0)
    }
}

#[cfg(any(target_os = "ios", target_os = "macos", target_os = "tvos"))]
mod apple {
    // task_info(TASK_VM_INFO): phys_footprint at byte 144, rev2's
    // ledger_phys_footprint_peak at byte 168.
    extern "C" {
        static mach_task_self_: u32;
        fn task_info(task: u32, flavor: i32, info: *mut i32, count: *mut u32) -> i32;
    }
    const TASK_VM_INFO: i32 = 22;
    pub fn footprint() -> (f64, f64) {
        let mut buf = [0u64; 96];
        let mut count = (std::mem::size_of_val(&buf) / 4) as u32;
        // SAFETY: the buffer is larger than any task_vm_info revision and
        // `count` tells the kernel its size in natural_t units.
        let kr = unsafe { task_info(mach_task_self_, TASK_VM_INFO, buf.as_mut_ptr() as *mut i32, &mut count) };
        if kr != 0 {
            return (0.0, 0.0);
        }
        let mb = |b: u64| b as f64 / (1024.0 * 1024.0);
        let now = mb(buf[144 / 8]);
        // count is in 4-byte units: rev2 reaches byte 176.
        let peak = if count as usize * 4 >= 176 { mb(buf[168 / 8]) } else { now };
        (now, peak.max(now))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_and_sink() {
        let c = Arc::new(Collector::default());
        set_sink(Some(c.clone()));
        event("t", "a").metric("x", 1.5).metric("nan", f64::NAN).tag("k", "v").flag("b", true).emit();
        {
            let mut s = span("t", "span");
            s.with(|b| b.metric("y", 2.0));
            s.finish();
        }
        {
            let _s = span("t", "dropped");
        }
        let mut st = Stats::default();
        for i in 0..11 {
            st.add(i as f64);
        }
        event("t", "stats").stats("tile", &st).json("set", &serde_json::json!({"a": 1, "b": "x", "c": true, "d": [1]})).emit();
        set_sink(None);
        event("t", "off").emit();
        let ev = c.drain();
        let names: Vec<_> = ev.iter().filter(|e| e.system == "t").map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["a", "span", "dropped", "stats"]);
        let a = &ev.iter().find(|e| e.name == "a").unwrap();
        assert_eq!(a.metrics["x"], 1.5);
        assert!(!a.metrics.contains_key("nan"));
        assert_eq!(a.metrics["b"], 1.0);
        assert_eq!(ev.iter().find(|e| e.name == "dropped").unwrap().tags["outcome"], "aborted");
        let s = ev.iter().find(|e| e.name == "stats").unwrap();
        assert_eq!(s.metrics["tile.p50"], 5.0);
        assert_eq!(s.metrics["tile.n"], 11.0);
        assert_eq!(s.metrics["set.a"], 1.0);
        assert_eq!(s.tags["set.b"], "x");
        assert!(!s.metrics.contains_key("set.d"));
        assert_ne!(new_run_id(), new_run_id());
        set_sink(Some(c.clone()));
        scope("r1", || event("t", "scoped").emit());
        event("t", "after").emit();
        set_sink(None);
        let ev = c.drain();
        assert_eq!(ev.iter().find(|e| e.name == "scoped").unwrap().run, "r1");
        assert_eq!(ev.iter().find(|e| e.name == "after").unwrap().run, "");
        let (now, peak) = memory_mb();
        assert!(peak >= now);
    }
}
