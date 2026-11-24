//! Salient-object segmentation networks: U²-Net (u2netp, Apache-2.0,
//! Qin et al. 2020) and IS-Net / DIS (isnet-general-use, Apache-2.0,
//! Qin et al. 2022). Both are stacks of RSU blocks — nested U-Nets of
//! 3x3 conv + batch-norm + ReLU with max-pool down, bilinear up and
//! dilated convs at the bottom — run on pb-nn's own conv kernels.
//!
//! Batch-norm is folded into the convs when the checkpoint is converted
//! (`fold`), so a REBNCONV is one conv with a ReLU epilogue. Dilated convs
//! run as plain 3x3 convs on the d x d polyphase components of the map
//! (exactly equal, zero padding included).

use crate::conv::{conv3x3, Act, Conv3x3, Epi};
use crate::fmap::Fmap;
use crate::pbm::Weights;
use crate::pth::{StateDict, TensorF32};
use rayon::prelude::*;
use std::collections::HashMap;

/// One conv of the network (BN folded), with its dilation.
struct C {
    conv: Conv3x3,
    dil: usize,
    relu: bool,
}

/// RSU block: `l` levels (7, 6, 5, 4) or the dilated "F" variant (4F).
struct Rsu {
    inp: C,
    enc: Vec<C>,
    dec: Vec<C>,
    f: bool,
}

pub struct SegNet {
    pub variant: String,
    /// Square input side the network was trained at.
    pub size: usize,
    conv_in: Option<C>,
    stages: Vec<Rsu>,
    /// 6 encoder stages then 5 decoder stages (5d, 4d, 3d, 2d, 1d).
    sides: Vec<C>,
    outconv: Option<(Vec<f32>, f32)>,
}

fn t<'a>(w: &'a HashMap<String, TensorF32>, n: &str) -> Result<&'a TensorF32, String> {
    w.get(n).ok_or_else(|| format!("segmentation model is missing {n}"))
}

/// Fold BatchNorm into the preceding conv: `w' = w * g / sqrt(v + eps)`,
/// `b' = (b - mean) * g / sqrt(v + eps) + beta`.
pub fn fold(sd: &StateDict) -> Result<StateDict, String> {
    let m: HashMap<&str, &TensorF32> = sd.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let mut out: StateDict = Vec::new();
    for (k, v) in sd {
        if let Some(p) = k.strip_suffix(".conv_s1.weight") {
            let g = m.get(format!("{p}.bn_s1.weight").as_str()).ok_or("bn weight")?;
            let beta = m.get(format!("{p}.bn_s1.bias").as_str()).ok_or("bn bias")?;
            let mean = m.get(format!("{p}.bn_s1.running_mean").as_str()).ok_or("bn mean")?;
            let var = m.get(format!("{p}.bn_s1.running_var").as_str()).ok_or("bn var")?;
            let b = m.get(format!("{p}.conv_s1.bias").as_str());
            let co = v.shape[0];
            let per = v.data.len() / co;
            let mut w = v.data.clone();
            let mut bias = vec![0.0f32; co];
            for c in 0..co {
                let s = g.data[c] / (var.data[c] + 1e-5).sqrt();
                for x in &mut w[c * per..(c + 1) * per] {
                    *x *= s;
                }
                let b0 = b.map(|b| b.data[c]).unwrap_or(0.0);
                bias[c] = (b0 - mean.data[c]) * s + beta.data[c];
            }
            out.push((format!("{p}.w"), TensorF32 { shape: v.shape.clone(), data: w }));
            out.push((format!("{p}.b"), TensorF32 { shape: vec![co], data: bias }));
        } else if k.contains(".conv_s1.") || k.contains(".bn_s1.") {
            continue;
        } else {
            out.push((k.clone(), v.clone()));
        }
    }
    Ok(out)
}

/// Is this a U²-Net family checkpoint? -> (variant, input size).
pub fn detect(sd: &StateDict) -> Option<(&'static str, usize)> {
    let has = |n: &str| sd.iter().any(|(k, _)| k == n);
    if !has("stage1.rebnconvin.conv_s1.weight") && !has("stage1.rebnconvin.w") {
        return None;
    }
    if has("conv_in.weight") {
        Some(("isnet", 1024))
    } else {
        Some(("u2netp", 320))
    }
}

fn conv_of(w: &HashMap<String, TensorF32>, wn: &str, bn: &str, dil: usize, relu: bool) -> Result<C, String> {
    let wt = t(w, wn)?;
    let b = t(w, bn)?;
    if wt.shape.len() != 4 || wt.shape[2] != 3 || wt.shape[3] != 3 {
        return Err(format!("{wn}: expected a 3x3 conv, got {:?}", wt.shape));
    }
    let c = Conv3x3::from_torch(&wt.data, Some(&b.data), wt.shape[0], wt.shape[1]);
    // f32 in memory: folded batch-norm weights lose too much in f16.
    Ok(C { conv: c, dil, relu })
}

fn rsu(w: &HashMap<String, TensorF32>, name: &str, f: bool) -> Result<Rsu, String> {
    let l = (1..=8).take_while(|i| w.contains_key(&format!("{name}.rebnconv{i}.w"))).count();
    if l < 4 {
        return Err(format!("{name}: RSU with {l} levels"));
    }
    let c = |p: &str, d: usize| conv_of(w, &format!("{name}.{p}.w"), &format!("{name}.{p}.b"), d, true);
    let inp = c("rebnconvin", 1)?;
    let mut enc = Vec::new();
    let mut dec = Vec::new();
    for i in 1..=l {
        // Plain RSU: dilation 2 only at the bottom; 4F: 1, 2, 4, 8.
        let d = if f { 1 << (i - 1) } else if i == l { 2 } else { 1 };
        enc.push(c(&format!("rebnconv{i}"), d)?);
    }
    for i in (1..l).rev() {
        let d = if f { 1 << (i - 1) } else { 1 };
        dec.push(c(&format!("rebnconv{i}d"), d)?);
    }
    Ok(Rsu { inp, enc, dec, f })
}

impl SegNet {
    pub fn from_weights(wts: &Weights) -> Result<SegNet, String> {
        if wts.header.arch != "u2net" {
            return Err(format!("not a segmentation model ({})", wts.header.arch));
        }
        let variant = wts.header.config.get("variant").and_then(|v| v.as_str()).unwrap_or("u2netp").to_string();
        let size = wts.header.config.get("size").and_then(|v| v.as_u64()).unwrap_or(320) as usize;
        let w = &wts.tensors;
        let conv_in = if w.contains_key("conv_in.weight") { Some(conv_of(w, "conv_in.weight", "conv_in.bias", 1, false)?) } else { None };
        let mut stages = Vec::new();
        for (n, f) in [("stage1", false), ("stage2", false), ("stage3", false), ("stage4", false), ("stage5", true), ("stage6", true), ("stage5d", true), ("stage4d", false), ("stage3d", false), ("stage2d", false), ("stage1d", false)] {
            stages.push(rsu(w, n, f)?);
        }
        let mut sides = Vec::new();
        for i in 1..=6 {
            sides.push(conv_of(w, &format!("side{i}.weight"), &format!("side{i}.bias"), 1, false)?);
        }
        let outconv = match (w.get("outconv.weight"), w.get("outconv.bias")) {
            (Some(a), Some(b)) => Some((a.data.clone(), b.data[0])),
            _ => None,
        };
        Ok(SegNet { variant, size, conv_in, stages, sides, outconv })
    }

    /// Multiply-accumulates for one forward pass at `n` x `n`.
    pub fn macs(&self, n: usize) -> f64 {
        // Rough: count each conv at the resolution it runs at.
        let mut total = 0.0;
        let mut res = if self.conv_in.is_some() { (n / 4) as f64 } else { n as f64 };
        if let Some(c) = &self.conv_in {
            total += (n * n / 4) as f64 * c.conv.macs_per_px() as f64;
        }
        for (i, s) in self.stages.iter().enumerate() {
            let r = if i < 6 { res / (1u64 << i) as f64 } else { res / (1u64 << (10 - i)) as f64 };
            let mut lr = r;
            total += r * r * s.inp.conv.macs_per_px() as f64;
            for (k, c) in s.enc.iter().enumerate() {
                total += lr * lr * c.conv.macs_per_px() as f64;
                if !s.f && k + 2 < s.enc.len() {
                    lr /= 2.0;
                }
            }
            for c in &s.dec {
                total += lr * lr * c.conv.macs_per_px() as f64;
                if !s.f {
                    lr = (lr * 2.0).min(r);
                }
            }
        }
        res = res.max(1.0);
        let _ = res;
        total
    }

    /// `x`: normalised RGB at the network's size (any size works; multiples
    /// of 32 avoid odd rounding). Returns the sigmoid saliency map at the
    /// input's size.
    pub fn forward(&self, x: &Fmap) -> Vec<f32> {
        let (h0, w0) = (x.h, x.w);
        let hx = match &self.conv_in {
            Some(c) => {
                // Stride-2 conv: full conv, keep every second pixel.
                let full = run(c, x, 0, None);
                maxpool(&subsample2(&full))
            }
            None => x.clone(),
        };
        let mut enc_out: Vec<Fmap> = Vec::new();
        let mut cur = hx;
        for i in 0..6 {
            let o = self.stages[i].forward(&cur);
            if i < 5 {
                cur = maxpool(&o);
            }
            enc_out.push(o);
        }
        // Decoder.
        let mut d = enc_out[5].clone();
        let mut dec_out: Vec<Fmap> = vec![d.clone()];
        for (k, si) in (6..11).enumerate() {
            let skip = &enc_out[4 - k];
            let up = upsample_to(&d, skip.h, skip.w);
            let cat = concat(&up, skip);
            d = self.stages[si].forward(&cat);
            dec_out.push(d.clone());
        }
        // dec_out: [hx6, hx5d, hx4d, hx3d, hx2d, hx1d]
        let hx1d = &dec_out[5];
        let side1 = run(&self.sides[0], hx1d, 0, None);
        let d1 = plane(&side1);
        let (h1, w1) = (hx1d.h, hx1d.w);
        let logits: Vec<f32> = match &self.outconv {
            Some((wv, b)) => {
                // U²-Net: fuse all six side outputs (upsampled to side 1).
                let mut sides: Vec<Vec<f32>> = vec![d1];
                for (i, m) in [&dec_out[4], &dec_out[3], &dec_out[2], &dec_out[1], &dec_out[0]].iter().enumerate() {
                    let s = run(&self.sides[i + 1], m, 0, None);
                    let s = upsample_to(&s, h1, w1);
                    sides.push(plane(&s));
                }
                (0..h1 * w1).map(|p| b + (0..6).map(|k| wv[k] * sides[k][p]).sum::<f32>()).collect()
            }
            None => d1,
        };
        // Back to the input size (IS-Net's side 1 is at 1/4).
        let mut lm = Fmap::new(h1, w1, 1);
        for y in 0..h1 {
            for xx in 0..w1 {
                lm.px_mut(y, xx)[0] = logits[y * w1 + xx];
            }
        }
        let lm = if (h1, w1) != (h0, w0) { upsample_to(&lm, h0, w0) } else { lm };
        plane(&lm).into_iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect()
    }
}

impl Rsu {
    fn forward(&self, x: &Fmap) -> Fmap {
        let hxin = run(&self.inp, x, 0, None);
        let l = self.enc.len();
        let mut skips: Vec<Fmap> = Vec::new();
        let mut cur = hxin.clone();
        for (i, c) in self.enc.iter().enumerate() {
            let o = run(c, &cur, 0, None);
            if i + 1 < l {
                // Pool after levels 1 .. l-2 (plain RSU); 4F never pools.
                cur = if !self.f && i + 2 < l { maxpool(&o) } else { o.clone() };
                skips.push(o);
            } else {
                cur = o;
            }
        }
        // cur = bottom (dilated) output; skips[l-2] is the level above it.
        let mut d = cur;
        for (k, c) in self.dec.iter().enumerate() {
            let skip = &skips[l - 2 - k];
            let up = if d.h != skip.h || d.w != skip.w { upsample_to(&d, skip.h, skip.w) } else { d };
            let cat = concat(&up, skip);
            let last = k + 1 == self.dec.len();
            d = run(c, &cat, 0, if last { Some(&hxin) } else { None });
        }
        d
    }
}

/// conv (+ReLU) (+ residual `res` added after the activation).
fn run(c: &C, x: &Fmap, cin_off: usize, res: Option<&Fmap>) -> Fmap {
    let mut out = Fmap::new(x.h, x.w, c.conv.cout);
    let act = if c.relu { Act::LeakyRelu(0.0) } else { Act::None };
    let epi = Epi { act, scale: 1.0, res: [res.map(|r| (r, 0, 1.0)), None] };
    if c.dil == 1 {
        conv3x3(x, cin_off, &c.conv, &mut out, 0, epi);
        return out;
    }
    assert!(res.is_none(), "dilated conv with residual");
    // Polyphase: the (py, px) component of a dilation-d conv is a plain
    // 3x3 conv of the (py, px) component of the input.
    let d = c.dil;
    let (h, w) = (x.h, x.w);
    for py in 0..d.min(h) {
        for px in 0..d.min(w) {
            let (sh, sw) = ((h - py).div_ceil(d), (w - px).div_ceil(d));
            let mut sub = Fmap::new(sh, sw, c.conv.cin);
            for y in 0..sh {
                for xx in 0..sw {
                    let s = x.off((py + y * d) as isize, (px + xx * d) as isize) + cin_off;
                    let o = sub.off(y as isize, xx as isize);
                    sub.data[o..o + c.conv.cin].copy_from_slice(&x.data[s..s + c.conv.cin]);
                }
            }
            let mut so = Fmap::new(sh, sw, c.conv.cout);
            conv3x3(&sub, 0, &c.conv, &mut so, 0, Epi { act, scale: 1.0, res: [None, None] });
            for y in 0..sh {
                for xx in 0..sw {
                    let s = so.off(y as isize, xx as isize);
                    let o = out.off((py + y * d) as isize, (px + xx * d) as isize);
                    out.data[o..o + c.conv.cout].copy_from_slice(&so.data[s..s + c.conv.cout]);
                }
            }
        }
    }
    out
}

/// 2x2 max pool, stride 2, ceil mode.
fn maxpool(x: &Fmap) -> Fmap {
    let (h, w) = (x.h.div_ceil(2), x.w.div_ceil(2));
    let c = x.c;
    let mut out = Fmap::new(h, w, c);
    let rs = out.row_stride();
    let base = out.off(0, 0);
    out.data[base..base + h * rs].par_chunks_mut(rs).enumerate().for_each(|(y, row)| {
        for xx in 0..w {
            let o = xx * c;
            for ch in 0..c {
                let mut m = f32::NEG_INFINITY;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let (sy, sx) = (2 * y + dy, 2 * xx + dx);
                        if sy < x.h && sx < x.w {
                            m = m.max(x.data[x.off(sy as isize, sx as isize) + ch]);
                        }
                    }
                }
                row[o + ch] = m;
            }
        }
    });
    out
}

fn subsample2(x: &Fmap) -> Fmap {
    let (h, w) = (x.h.div_ceil(2), x.w.div_ceil(2));
    let mut out = Fmap::new(h, w, x.c);
    for y in 0..h {
        for xx in 0..w {
            let s = x.off((2 * y) as isize, (2 * xx) as isize);
            let o = out.off(y as isize, xx as isize);
            out.data[o..o + x.c].copy_from_slice(&x.data[s..s + x.c]);
        }
    }
    out
}

/// Bilinear resize to `h` x `w` (PyTorch `align_corners=False`).
pub fn upsample_to(x: &Fmap, h: usize, w: usize) -> Fmap {
    let c = x.c;
    let mut out = Fmap::new(h, w, c);
    let (sy, sx) = (x.h as f32 / h as f32, x.w as f32 / w as f32);
    let src = |o: usize, s: f32, n: usize| {
        let v = ((o as f32 + 0.5) * s - 0.5).max(0.0);
        let i0 = (v as usize).min(n - 1);
        let i1 = (i0 + 1).min(n - 1);
        (i0, i1, v - i0 as f32)
    };
    let xs: Vec<(usize, usize, f32)> = (0..w).map(|o| src(o, sx, x.w)).collect();
    let rs = out.row_stride();
    let base = out.off(0, 0);
    out.data[base..base + h * rs].par_chunks_mut(rs).enumerate().for_each(|(y, row)| {
        let (y0, y1, fy) = src(y, sy, x.h);
        for (xx, &(x0, x1, fx)) in xs.iter().enumerate() {
            let (a, b, cc, d) = (x.off(y0 as isize, x0 as isize), x.off(y0 as isize, x1 as isize), x.off(y1 as isize, x0 as isize), x.off(y1 as isize, x1 as isize));
            let o = xx * c;
            for ch in 0..c {
                let top = x.data[a + ch] + (x.data[b + ch] - x.data[a + ch]) * fx;
                let bot = x.data[cc + ch] + (x.data[d + ch] - x.data[cc + ch]) * fx;
                row[o + ch] = top + (bot - top) * fy;
            }
        }
    });
    out
}

/// Channel concat (a first, then b), same size.
fn concat(a: &Fmap, b: &Fmap) -> Fmap {
    assert_eq!((a.h, a.w), (b.h, b.w));
    let mut out = Fmap::new(a.h, a.w, a.c + b.c);
    out.copy_channels(0, a, 0, a.c);
    out.copy_channels(a.c, b, 0, b.c);
    out
}

fn plane(x: &Fmap) -> Vec<f32> {
    let mut v = Vec::with_capacity(x.h * x.w);
    for y in 0..x.h {
        for xx in 0..x.w {
            v.push(x.data[x.off(y as isize, xx as isize)]);
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive dilated conv for checking the polyphase path.
    fn naive(x: &Fmap, w: &[f32], b: &[f32], cout: usize, cin: usize, d: usize) -> Vec<f32> {
        let mut o = vec![0.0; x.h * x.w * cout];
        for y in 0..x.h as isize {
            for xx in 0..x.w as isize {
                for co in 0..cout {
                    let mut acc = b[co];
                    for ci in 0..cin {
                        for ky in 0..3isize {
                            for kx in 0..3isize {
                                let (sy, sx) = (y + (ky - 1) * d as isize, xx + (kx - 1) * d as isize);
                                if sy < 0 || sx < 0 || sy >= x.h as isize || sx >= x.w as isize {
                                    continue;
                                }
                                acc += w[((co * cin + ci) * 3 + ky as usize) * 3 + kx as usize] * x.data[x.off(sy, sx) + ci];
                            }
                        }
                    }
                    o[(y as usize * x.w + xx as usize) * cout + co] = acc.max(0.0);
                }
            }
        }
        o
    }

    #[test]
    fn polyphase_dilation_matches_naive() {
        let (cin, cout) = (5, 7);
        let mut seed = 3u64;
        let mut r = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let w: Vec<f32> = (0..cout * cin * 9).map(|_| r()).collect();
        let b: Vec<f32> = (0..cout).map(|_| r()).collect();
        for (h, wd, d) in [(9, 11, 2), (10, 7, 4), (13, 13, 8), (5, 6, 2)] {
            let mut x = Fmap::new(h, wd, cin);
            for y in 0..h {
                for xx in 0..wd {
                    for c in 0..cin {
                        x.px_mut(y, xx)[c] = r();
                    }
                }
            }
            let c = C { conv: Conv3x3::from_torch(&w, Some(&b), cout, cin), dil: d, relu: true };
            let got = run(&c, &x, 0, None);
            let want = naive(&x, &w, &b, cout, cin, d);
            for y in 0..h {
                for xx in 0..wd {
                    for co in 0..cout {
                        let (g, e) = (got.px(y, xx)[co], want[(y * wd + xx) * cout + co]);
                        assert!((g - e).abs() < 1e-4, "{h}x{wd} d{d}: {g} vs {e}");
                    }
                }
            }
        }
    }

    #[test]
    fn pool_and_upsample() {
        let mut x = Fmap::new(5, 3, 1);
        for y in 0..5 {
            for xx in 0..3 {
                x.px_mut(y, xx)[0] = (y * 3 + xx) as f32;
            }
        }
        let p = maxpool(&x);
        assert_eq!((p.h, p.w), (3, 2));
        assert_eq!(p.px(0, 0)[0], 4.0);
        assert_eq!(p.px(2, 1)[0], 14.0);
        // 2x2 -> 4x4 bilinear, align_corners=False: corners replicate.
        let mut s = Fmap::new(2, 2, 1);
        for (i, v) in [0.0, 1.0, 2.0, 3.0].iter().enumerate() {
            s.px_mut(i / 2, i % 2)[0] = *v;
        }
        let u = upsample_to(&s, 4, 4);
        assert_eq!(u.px(0, 0)[0], 0.0);
        assert!((u.px(0, 1)[0] - 0.25).abs() < 1e-6);
        assert_eq!(u.px(3, 3)[0], 3.0);
    }
}
