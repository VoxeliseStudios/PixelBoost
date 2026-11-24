//! `pixelboost telemetry ...` — fetch the server's telemetry and turn it
//! into tuning reports (speed per backend / device, Auto overrides vs the
//! analyzer's choices, feedback per model / style / op, cutout fixes ...).
//!
//! Any CLI command also records its own engine events when
//! `PB_TELEMETRY=<file.jsonl>` is set (bench / eval runs feed the same
//! reports).

use pb_core::telemetry::{self as tm, Collector, Event};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, Write};
use std::sync::Arc;

static LOCAL: std::sync::OnceLock<Arc<Collector>> = std::sync::OnceLock::new();

/// `PB_TELEMETRY=file.jsonl`: collect this process's events.
pub fn init_from_env() {
    if std::env::var_os("PB_TELEMETRY").is_some() {
        let c = Arc::new(Collector::default());
        let _ = LOCAL.set(c.clone());
        tm::set_sink(Some(c));
    }
}

/// Append collected events to the `PB_TELEMETRY` file.
pub fn finish() {
    let (Some(c), Some(p)) = (LOCAL.get(), std::env::var_os("PB_TELEMETRY")) else { return };
    tm::flush_aggregates();
    let ev = c.drain();
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
        for e in &ev {
            let _ = writeln!(f, "{}", serde_json::to_string(e).unwrap_or_default());
        }
        eprintln!("telemetry: {} events -> {}", ev.len(), p.to_string_lossy());
    }
}

pub fn cmd(args: &[String]) -> Result<(), String> {
    match args.get(1).map(String::as_str) {
        Some("pull") if args.len() >= 4 => pull(args),
        Some("report") if args.len() >= 3 => report(&load(&args[2])?, args),
        _ => Err("telemetry pull <server url> <out.jsonl> [--key k] [--system s] [--hours h]\n          telemetry report <events.jsonl> [--top n]".into()),
    }
}

/// Page through `GET /log/telemetry` (after_id) into a JSON-lines file.
fn pull(args: &[String]) -> Result<(), String> {
    let base = args[2].trim_end_matches('/');
    let out = &args[3];
    let key = crate::flag(args, "--key").map(String::from).or_else(|| std::env::var("PB_SERVER_KEY").ok()).unwrap_or_default();
    let hours = crate::flag(args, "--hours").unwrap_or("720");
    let agent = ureq::AgentBuilder::new().timeout(std::time::Duration::from_secs(120)).build();
    let mut f = std::fs::File::create(out).map_err(|e| e.to_string())?;
    let (mut after, mut total) = (0i64, 0usize);
    loop {
        let mut r = agent.get(&format!("{base}/log/telemetry")).query("limit", "20000").query("hours", hours).query("after_id", &after.to_string());
        if let Some(s) = crate::flag(args, "--system") {
            r = r.query("system", s);
        }
        if !key.is_empty() {
            r = r.set("Authorization", &format!("Bearer {key}"));
        }
        let v: Value = r.call().map_err(|e| e.to_string())?.into_json().map_err(|e| e.to_string())?;
        let evs = v["events"].as_array().cloned().unwrap_or_default();
        if evs.is_empty() {
            break;
        }
        for e in &evs {
            after = after.max(e["id"].as_i64().unwrap_or(after));
            writeln!(f, "{e}").map_err(|e| e.to_string())?;
        }
        total += evs.len();
        eprintln!("  {total} events");
        if evs.len() < 20000 {
            break;
        }
    }
    eprintln!("{total} events -> {out}");
    Ok(())
}

/// One event as the reports see it (from a local file or the server).
#[derive(Clone, Debug, Default)]
pub struct Ev {
    pub system: String,
    pub name: String,
    pub run: String,
    pub ms: f64,
    pub metrics: BTreeMap<String, f64>,
    pub tags: BTreeMap<String, String>,
    pub device_model: String,
}

pub fn load(path: &str) -> Result<Vec<Ev>, String> {
    let f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut v = Vec::new();
    for l in std::io::BufReader::new(f).lines().map_while(Result::ok) {
        let Ok(j) = serde_json::from_str::<Value>(&l) else { continue };
        if let Ok(e) = serde_json::from_value::<Event>(j.clone()) {
            v.push(Ev { system: e.system, name: e.name, run: e.run, ms: e.ms, metrics: e.metrics, tags: e.tags, device_model: String::new() });
            continue;
        }
        // Server rows: run_id, ms, device_model.
        let m = |k: &str| j.get(k).and_then(|o| o.as_object()).map(|o| o.iter().filter_map(|(a, b)| b.as_f64().map(|f| (a.clone(), f))).collect()).unwrap_or_default();
        let t = |k: &str| j.get(k).and_then(|o| o.as_object()).map(|o| o.iter().filter_map(|(a, b)| b.as_str().map(|s| (a.clone(), s.to_string()))).collect()).unwrap_or_default();
        let s = |k: &str| j.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();
        v.push(Ev { system: s("system"), name: s("name"), run: s("run_id"), ms: j["ms"].as_f64().unwrap_or(0.0), metrics: m("metrics"), tags: t("tags"), device_model: s("device_model") });
    }
    Ok(v)
}

fn q(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

pub fn report(ev: &[Ev], args: &[String]) -> Result<(), String> {
    let top: usize = crate::flag(args, "--top").and_then(|v| v.parse().ok()).unwrap_or(12);
    println!("== {} events", ev.len());
    // 1. Event counts and durations.
    let mut by: BTreeMap<(String, String), Vec<f64>> = BTreeMap::new();
    for e in ev {
        by.entry((e.system.clone(), e.name.clone())).or_default().push(e.ms);
    }
    println!("\n-- events                              n      ms p50     ms p90");
    for ((s, n), mut v) in by {
        let c = v.len();
        println!("  {:<34} {:>6} {:>11.1} {:>11.1}", format!("{s}/{n}"), c, q(&mut v, 0.5), q(&mut v, 0.9));
    }
    let runs: HashMap<&str, &Ev> = ev.iter().filter(|e| e.system == "upscale" && e.name == "run" && !e.run.is_empty()).map(|e| (e.run.as_str(), e)).collect();
    let fb: Vec<&Ev> = ev.iter().filter(|e| e.system == "feedback").collect();

    // 2. Upscale speed per backend / device (effective GFLOP/s of the network).
    let mut speed: BTreeMap<String, (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for e in ev.iter().filter(|e| e.system == "upscale" && e.name == "run" && e.tags.get("outcome").map(|o| o == "ok").unwrap_or(false)) {
        let be = e.tags.get("backend").cloned().unwrap_or_default();
        let be = be.split(':').next().unwrap_or("").to_string();
        let k = format!("{} | {} | {}", if e.device_model.is_empty() { "-" } else { &e.device_model }, be, e.tags.get("where").map(|s| s.as_str()).unwrap_or("phone"));
        let s = speed.entry(k).or_default();
        let net = e.metrics.get("net_ms").copied().unwrap_or(0.0);
        if let (Some(g), true) = (e.metrics.get("gflop"), net > 0.0) {
            s.0.push(g / (net / 1000.0));
        }
        let total = e.ms.max(1.0);
        s.1.push(e.metrics.get("post_ms").copied().unwrap_or(0.0) / total);
        s.2.push(e.metrics.get("stall_ms").copied().unwrap_or(0.0) / total);
        s.3.push(e.metrics.get("mem_peak_mb").copied().unwrap_or(0.0));
    }
    if !speed.is_empty() {
        println!("\n-- upscale speed (device | backend | where)        runs  GFLOP/s p50  post share  GPU stall  peak MB p90");
        for (k, (mut g, p, st, mut m)) in speed {
            println!("  {:<48} {:>5} {:>12.1} {:>10.0}% {:>9.0}% {:>12.0}", k, p.len(), q(&mut g, 0.5), mean(&p) * 100.0, mean(&st) * 100.0, q(&mut m, 0.9));
        }
    }

    // 3. Feedback per model (kept / discarded / rating / rerun).
    if !fb.is_empty() {
        let mut per: BTreeMap<String, BTreeMap<String, (usize, f64)>> = BTreeMap::new();
        for f in &fb {
            let grp = runs.get(f.run.as_str()).and_then(|r| r.tags.get("model").cloned()).unwrap_or_else(|| "(other runs)".into());
            let x = per.entry(grp).or_default().entry(f.name.clone()).or_insert((0, 0.0));
            x.0 += 1;
            x.1 += f.metrics.get("value").copied().unwrap_or(0.0);
        }
        println!("\n-- feedback by upscale model");
        for (g, m) in per {
            let s: Vec<String> = m.iter().map(|(k, (n, sum))| if k == "rating" { format!("rating {n}x avg {:+.2}", sum / *n as f64) } else { format!("{k} {n}") }).collect();
            println!("  {:<14} {}", g, s.join(" · "));
        }
        // 4. Auto overrides: what the analyzer chose vs what people set.
        let mut ov: BTreeMap<String, BTreeMap<(String, String), usize>> = BTreeMap::new();
        for f in fb.iter().filter(|f| f.name == "override") {
            let field = f.tags.get("d.field").cloned().unwrap_or_default();
            let a = f.tags.get("d.auto").cloned().or_else(|| f.metrics.get("d.auto").map(|v| format!("{v}"))).unwrap_or_default();
            let u = f.tags.get("d.user").cloned().or_else(|| f.metrics.get("d.user").map(|v| format!("{v}"))).unwrap_or_default();
            *ov.entry(field).or_default().entry((a, u)).or_default() += 1;
        }
        if !ov.is_empty() {
            println!("\n-- Auto overridden (field: auto -> user, count) — candidates for re-fitting analyze.rs thresholds");
            for (field, m) in ov {
                let mut v: Vec<_> = m.into_iter().collect();
                v.sort_by(|a, b| b.1.cmp(&a.1));
                let s: Vec<String> = v.iter().take(top).map(|((a, u), n)| format!("{a}->{u} {n}")).collect();
                println!("  {:<10} {}", field, s.join(" · "));
            }
        }
    }

    // 5. Analyzer: measured features by chosen model (threshold check).
    let rec: Vec<&Ev> = ev.iter().filter(|e| e.system == "analyze" && e.name == "recommend").collect();
    if !rec.is_empty() {
        let mut per: BTreeMap<String, BTreeMap<&str, Vec<f64>>> = BTreeMap::new();
        for r in &rec {
            let m = r.tags.get("rec.model").cloned().unwrap_or_default();
            for k in ["m.noise", "m.blocking", "m.sharpness", "m.skin", "m.eff", "m.line", "m.palette"] {
                if let Some(v) = r.metrics.get(k) {
                    per.entry(m.clone()).or_default().entry(k).or_default().push(*v);
                }
            }
        }
        println!("\n-- analyzer: feature medians by chosen model");
        for (m, f) in per {
            let s: Vec<String> = f.into_iter().map(|(k, mut v)| format!("{} {:.2}", &k[2..], q(&mut v, 0.5))).collect();
            println!("  {:<8} {}", m, s.join("  "));
        }
    }

    // 6. Editor ops, pixel art styles, cutout corrections.
    let mut ops: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for e in ev.iter().filter(|e| e.system == "editor") {
        let k = e.tags.get("op").or_else(|| e.tags.get("key")).cloned().unwrap_or_default();
        let ms = if e.name == "preview" { e.metrics.get("ms.p50").copied().unwrap_or(e.ms) } else { e.ms };
        ops.entry(format!("{} {}", e.name, k)).or_default().push(ms);
    }
    if !ops.is_empty() {
        println!("\n-- editor (preview = per-64-sample p50)          n    ms p50    ms p90");
        let mut v: Vec<_> = ops.into_iter().collect();
        v.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
        for (k, mut t) in v.into_iter().take(top * 2) {
            println!("  {:<44} {:>4} {:>9.1} {:>9.1}", k, t.len(), q(&mut t, 0.5), q(&mut t, 0.9));
        }
    }
    let px: Vec<&Ev> = ev.iter().filter(|e| e.system == "pixelart").collect();
    if !px.is_empty() {
        let mut st: BTreeMap<String, Vec<&Ev>> = BTreeMap::new();
        for e in &px {
            st.entry(e.tags.get("p.palette").cloned().unwrap_or_default()).or_default().push(e);
        }
        println!("\n-- pixel art by palette: n, width p50, colours used, outline / dither share, ms p50");
        for (k, v) in st {
            let f = |key: &str| v.iter().filter_map(|e| e.metrics.get(key).copied()).collect::<Vec<f64>>();
            let (mut w, c, o, d, mut t) = (f("p.width"), f("colours_used"), f("outline_share"), f("dither_share"), v.iter().map(|e| e.ms).collect::<Vec<_>>());
            println!("  {:<12} {:>4} {:>6.0} {:>6.1} {:>6.1}% {:>6.1}% {:>8.0}", k, v.len(), q(&mut w, 0.5), mean(&c), mean(&o) * 100.0, mean(&d) * 100.0, q(&mut t, 0.5));
        }
    }
    let cut: Vec<&Ev> = ev.iter().filter(|e| e.system == "cutout" && e.name == "mask").collect();
    if !cut.is_empty() {
        let fixes: HashMap<&str, f64> = fb.iter().filter(|f| f.name == "mask_fix").map(|f| (f.run.as_str(), f.metrics.get("value").copied().unwrap_or(0.0))).collect();
        let mut per: BTreeMap<String, (usize, Vec<f64>, Vec<f64>, Vec<f64>)> = BTreeMap::new();
        for c in cut {
            let k = c.tags.get("model").cloned().unwrap_or_default();
            let x = per.entry(k).or_default();
            x.0 += 1;
            x.1.push(c.ms);
            x.2.push(c.metrics.get("uncertain").copied().unwrap_or(0.0));
            if let Some(f) = fixes.get(c.run.as_str()) {
                x.3.push(*f);
            }
        }
        println!("\n-- cutout: runs, ms p50, uncertain edge share, corrected runs, repainted share");
        for (k, (n, mut t, u, f)) in per {
            println!("  {:<10} {:>4} {:>8.0} {:>8.1}% {:>5} {:>8.1}%", k, n, q(&mut t, 0.5), mean(&u) * 100.0, f.len(), mean(&f) * 100.0);
        }
    }
    Ok(())
}
