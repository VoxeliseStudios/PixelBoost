//! `.pbnn` — PixelBoost's own model file: a JSON header describing the
//! architecture and tensors, followed by the tensor data.
//!
//! ```text
//! b"PBNN" | u32 version (1 or 2) | u32 header_len | header JSON | pad to 16 | data
//! ```
//!
//! Version 1 stores every tensor as fp16. Version 2 adds a per-tensor
//! `dtype`: "f16", or "q8" = int8 with one f32 scale per output channel
//! (`[cout x f32 scales][numel x i8]`), half the size of fp16 for conv
//! weights. Weights are expanded back to float at load, so q8 only changes
//! the file (download / app size), not the arithmetic.

use crate::half::{f16_to_f32, f32_to_f16};
use crate::pth::{StateDict, TensorF32};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    /// Byte offset into the data section.
    pub offset: usize,
    /// "f16" (default) or "q8".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Header {
    pub arch: String,
    pub config: serde_json::Value,
    pub meta: serde_json::Value,
    pub tensors: Vec<TensorInfo>,
}

pub struct Weights {
    pub header: Header,
    pub tensors: HashMap<String, TensorF32>,
}

impl Weights {
    pub fn get(&self, name: &str) -> Result<&TensorF32, String> {
        self.tensors.get(name).ok_or_else(|| format!("model is missing tensor {name}"))
    }
    pub fn cfg(&self, key: &str) -> Option<i64> {
        self.header.config.get(key).and_then(|v| v.as_i64())
    }
}

pub fn write(arch: &str, config: serde_json::Value, meta: serde_json::Value, sd: &StateDict) -> Vec<u8> {
    let mut data = Vec::new();
    let mut tensors = Vec::new();
    for (name, t) in sd {
        tensors.push(TensorInfo { name: name.clone(), shape: t.shape.clone(), offset: data.len(), dtype: None });
        for v in &t.data {
            data.extend_from_slice(&f32_to_f16(*v).to_le_bytes());
        }
    }
    let header = serde_json::to_vec(&Header { arch: arch.into(), config, meta, tensors }).unwrap();
    let mut out = Vec::with_capacity(header.len() + data.len() + 32);
    out.extend_from_slice(b"PBNN");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(header.len() as u32).to_le_bytes());
    out.extend_from_slice(&header);
    while out.len() % 16 != 0 {
        out.push(0);
    }
    out.extend_from_slice(&data);
    out
}

pub fn read(bytes: &[u8]) -> Result<Weights, String> {
    if bytes.len() < 12 || &bytes[..4] != b"PBNN" {
        return Err("not a .pbnn model".into());
    }
    let ver = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if ver != 1 && ver != 2 {
        return Err(format!("unsupported .pbnn version {ver}"));
    }
    let hlen = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let header: Header = serde_json::from_slice(&bytes[12..12 + hlen]).map_err(|e| e.to_string())?;
    let data_start = (12 + hlen).div_ceil(16) * 16;
    let data = &bytes[data_start..];
    let mut tensors = HashMap::new();
    for t in &header.tensors {
        let n: usize = t.shape.iter().product();
        let v = match t.dtype.as_deref() {
            None | Some("f16") => {
                let raw = data.get(t.offset..t.offset + n * 2).ok_or_else(|| format!("{} is truncated", t.name))?;
                raw.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()
            }
            Some("f32") => {
                let raw = data.get(t.offset..t.offset + n * 4).ok_or_else(|| format!("{} is truncated", t.name))?;
                raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
            }
            Some("q8") => {
                let co = t.shape.first().copied().unwrap_or(1).max(1);
                let per = n / co;
                let raw = data.get(t.offset..t.offset + co * 4 + n).ok_or_else(|| format!("{} is truncated", t.name))?;
                let scales: Vec<f32> = raw[..co * 4].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                raw[co * 4..].iter().enumerate().map(|(i, q)| *q as i8 as f32 * scales[i / per]).collect()
            }
            Some(d) => return Err(format!("{}: unknown dtype {d}", t.name)),
        };
        tensors.insert(t.name.clone(), TensorF32 { shape: t.shape.clone(), data: v });
    }
    Ok(Weights { header, tensors })
}

/// Version 2 file with full f32 tensors (networks whose folded weights
/// don't survive fp16, e.g. batch-norm-folded segmentation nets).
pub fn write_f32(arch: &str, config: serde_json::Value, meta: serde_json::Value, sd: &StateDict) -> Vec<u8> {
    let mut data = Vec::new();
    let mut tensors = Vec::new();
    for (name, t) in sd {
        tensors.push(TensorInfo { name: name.clone(), shape: t.shape.clone(), offset: data.len(), dtype: Some("f32".into()) });
        for v in &t.data {
            data.extend_from_slice(&v.to_le_bytes());
        }
    }
    let header = serde_json::to_vec(&Header { arch: arch.into(), config, meta, tensors }).unwrap();
    let mut out = Vec::with_capacity(header.len() + data.len() + 32);
    out.extend_from_slice(b"PBNN");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(header.len() as u32).to_le_bytes());
    out.extend_from_slice(&header);
    while out.len() % 16 != 0 {
        out.push(0);
    }
    out.extend_from_slice(&data);
    out
}

/// Re-encode a model as version 2 with int8 conv weights (per output
/// channel scales); 1-D tensors (biases, PReLU) stay fp16. Returns the new
/// file and the worst relative weight error.
pub fn quantize_q8(w: &Weights) -> (Vec<u8>, f32) {
    let mut data = Vec::new();
    let mut tensors = Vec::new();
    let mut worst = 0.0f32;
    let mut names: Vec<&String> = w.tensors.keys().collect();
    // Keep the original tensor order.
    names.sort_by_key(|n| w.header.tensors.iter().position(|t| &t.name == *n).unwrap_or(usize::MAX));
    for name in names {
        let t = &w.tensors[name];
        while data.len() % 4 != 0 {
            data.push(0);
        }
        let offset = data.len();
        if t.shape.len() >= 2 && t.data.len() >= 64 {
            let co = t.shape[0];
            let per = t.data.len() / co;
            let scales: Vec<f32> = (0..co).map(|c| t.data[c * per..(c + 1) * per].iter().fold(0.0f32, |m, v| m.max(v.abs())) / 127.0).collect();
            for s in &scales {
                data.extend_from_slice(&s.to_le_bytes());
            }
            for (i, v) in t.data.iter().enumerate() {
                let s = scales[i / per];
                let q = if s > 0.0 { (v / s).round().clamp(-127.0, 127.0) } else { 0.0 };
                if s > 0.0 {
                    worst = worst.max((q * s - v).abs() / (s * 127.0));
                }
                data.push(q as i8 as u8);
            }
            tensors.push(TensorInfo { name: name.clone(), shape: t.shape.clone(), offset, dtype: Some("q8".into()) });
        } else {
            for v in &t.data {
                data.extend_from_slice(&f32_to_f16(*v).to_le_bytes());
            }
            tensors.push(TensorInfo { name: name.clone(), shape: t.shape.clone(), offset, dtype: None });
        }
    }
    let mut meta = w.header.meta.clone();
    if let Some(m) = meta.as_object_mut() {
        m.insert("weights".into(), "q8 per output channel".into());
    }
    let header = serde_json::to_vec(&Header { arch: w.header.arch.clone(), config: w.header.config.clone(), meta, tensors }).unwrap();
    let mut out = Vec::with_capacity(header.len() + data.len() + 32);
    out.extend_from_slice(b"PBNN");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(header.len() as u32).to_le_bytes());
    out.extend_from_slice(&header);
    while out.len() % 16 != 0 {
        out.push(0);
    }
    out.extend_from_slice(&data);
    (out, worst)
}

/// Network interpolation (DNI): `t * a + (1 - t) * b`, tensor by tensor.
/// Both models must share an architecture (e.g. realesr-general-x4v3 and its
/// `wdn` denoise twin) — this is how the denoise-strength slider works.
pub fn blend(a: &Weights, b: &Weights, t: f32) -> Result<Weights, String> {
    if a.header.arch != b.header.arch || a.header.config != b.header.config {
        return Err("can only blend models of the same architecture".into());
    }
    let mut tensors = HashMap::new();
    for (name, ta) in &a.tensors {
        let tb = b.get(name)?;
        if ta.shape != tb.shape {
            return Err(format!("{name}: shape mismatch"));
        }
        let data = ta.data.iter().zip(&tb.data).map(|(x, y)| t * x + (1.0 - t) * y).collect();
        tensors.insert(name.clone(), TensorF32 { shape: ta.shape.clone(), data });
    }
    Ok(Weights { header: a.header.clone(), tensors })
}
