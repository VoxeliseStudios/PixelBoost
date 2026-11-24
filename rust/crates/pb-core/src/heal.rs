//! Healing brush: texture-aware fill of a masked area.
//!
//! Diffusion inpaint (`edit::inpaint`) fills holes with smooth colour —
//! fine for scratches, smeared on grass, brick or skin. Healing rebuilds
//! the texture: Wexler-style completion with PatchMatch (Barnes et al.
//! 2009) from coarse to fine. Every 7x7 patch touching the hole finds its
//! most similar patch in the untouched picture nearby, and each hole pixel
//! becomes the weighted vote of the patches that cover it; a few rounds
//! per level settle it, and the next finer level starts from the upsampled
//! result and matches.
//!
//! Only a window around the hole is processed (the hole's box plus a
//! margin of three times its size), so a speck on a 48 MP photo costs the
//! same as on a thumbnail.

use crate::image::Image;
use crate::resample::{resize, Filter};
use rayon::prelude::*;

const P: isize = 3; // patch radius (7x7)

#[derive(Clone, Copy, Debug, Default)]
pub struct HealReport {
    pub hole_px: usize,
    pub levels: usize,
    pub window: (usize, usize),
    /// Mean patch distance at the finest level (per channel, 0..1 units).
    pub patch_error: f32,
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn range(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

/// Fill `mask` (> 0.5 = heal) in `img`.
pub fn heal(img: &Image, mask: &[f32]) -> (Image, HealReport) {
    let (w, h) = (img.w, img.h);
    let hole: Vec<bool> = mask.iter().map(|m| *m > 0.5).collect();
    let n_hole = hole.iter().filter(|b| **b).count();
    let mut rep = HealReport { hole_px: n_hole, ..Default::default() };
    if n_hole == 0 || n_hole == w * h {
        return (img.clone(), rep);
    }
    // Window: hole bounding box + 3x its size (sources come from here).
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    for (i, b) in hole.iter().enumerate() {
        if *b {
            let (x, y) = (i % w, i / w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
    }
    let m = ((x1 - x0).max(y1 - y0) * 3).max(48);
    let (wx0, wy0) = (x0.saturating_sub(m), y0.saturating_sub(m));
    let (wx1, wy1) = ((x1 + m + 1).min(w), (y1 + m + 1).min(h));
    let (ww, wh) = (wx1 - wx0, wy1 - wy0);
    rep.window = (ww, wh);
    let win = img.crop(wx0, wy0, ww, wh);
    let mut whole = vec![false; ww * wh];
    for y in 0..wh {
        for x in 0..ww {
            whole[y * ww + x] = hole[(y + wy0) * w + x + wx0];
        }
    }
    // Pyramid: halve until the hole is a few pixels across.
    let mut levels: Vec<(Image, Vec<bool>)> = vec![(win, whole)];
    loop {
        let (li, lm) = levels.last().unwrap();
        let hole_side = (x1 - x0).max(y1 - y0) >> (levels.len() - 1);
        if li.w < 48 || li.h < 48 || hole_side < 12 {
            break;
        }
        let (nw, nh) = (li.w / 2, li.h / 2);
        let small = resize(li, nw, nh, Filter::Box);
        let mut sm = vec![false; nw * nh];
        for y in 0..nh {
            for x in 0..nw {
                // Any hole pixel under it makes it a hole.
                sm[y * nw + x] = (0..2).any(|dy| (0..2).any(|dx| lm[(2 * y + dy).min(li.h - 1) * li.w + (2 * x + dx).min(li.w - 1)]));
            }
        }
        levels.push((small, sm));
    }
    rep.levels = levels.len();
    // Coarsest level: start from the diffusion fill.
    let (ci, cm) = levels.last().unwrap();
    let mf: Vec<f32> = cm.iter().map(|b| if *b { 1.0 } else { 0.0 }).collect();
    let mut cur = crate::edit::inpaint(ci, &mf);
    let mut nnf: Vec<(i32, i32)> = Vec::new();
    for li in (0..levels.len()).rev() {
        let (src, hm) = &levels[li];
        if li + 1 < levels.len() {
            // Upsample the previous result into this level's hole.
            let up = resize(&cur, src.w, src.h, Filter::CatmullRom);
            let mut next = src.clone();
            for (i, b) in hm.iter().enumerate() {
                if *b {
                    next.px[i] = up.px[i];
                }
            }
            cur = next;
            // And the matches (doubled).
            let (pw, ph) = (levels[li + 1].0.w, levels[li + 1].0.h);
            let old = std::mem::take(&mut nnf);
            nnf = (0..src.w * src.h)
                .map(|i| {
                    let (x, y) = (i % src.w, i / src.w);
                    let (px, py) = ((x / 2).min(pw - 1), (y / 2).min(ph - 1));
                    let (sx, sy) = old.get(py * pw + px).copied().unwrap_or((0, 0));
                    ((sx * 2 + (x % 2) as i32).min(src.w as i32 - 1 - P as i32), (sy * 2 + (y % 2) as i32).min(src.h as i32 - 1 - P as i32))
                })
                .collect();
        }
        let iters = if li == 0 { 3 } else { 5 };
        let err = complete_level(&mut cur, src, hm, &mut nnf, iters, li as u64);
        if li == 0 {
            rep.patch_error = err;
        }
    }
    // Paste the window back; feather the hole edge by one pixel.
    let mut out = img.clone();
    for y in 0..wh {
        for x in 0..ww {
            let i = (y + wy0) * w + x + wx0;
            if hole[i] {
                let mut p = cur.px[y * ww + x];
                p[3] = img.px[i][3];
                out.px[i] = p;
            }
        }
    }
    (out, rep)
}

/// Is a source patch centred at (x, y) valid (inside, no hole pixels)?
fn valid(hm: &[bool], w: usize, h: usize, x: i32, y: i32, ok: &[bool]) -> bool {
    let _ = hm;
    x >= P as i32 && y >= P as i32 && x < (w as isize - P) as i32 && y < (h as isize - P) as i32 && ok[y as usize * w + x as usize]
}

fn dist(img: &Image, ax: usize, ay: usize, bx: usize, by: usize, best: f32) -> f32 {
    let w = img.w;
    let mut d = 0.0f32;
    for dy in -P..=P {
        let ra = (ay as isize + dy) as usize * w;
        let rb = (by as isize + dy) as usize * w;
        for dx in -P..=P {
            let a = img.px[ra + (ax as isize + dx) as usize];
            let b = img.px[rb + (bx as isize + dx) as usize];
            d += (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
        }
        if d > best {
            return d;
        }
    }
    d
}

/// EM rounds at one level: PatchMatch for every patch centre whose patch
/// touches the hole, then vote. Returns the mean match distance.
fn complete_level(cur: &mut Image, src: &Image, hm: &[bool], nnf: &mut Vec<(i32, i32)>, iters: usize, seed: u64) -> f32 {
    let (w, h) = (src.w, src.h);
    // Source centres: patches entirely outside the hole.
    let hole_f: Vec<f32> = hm.iter().map(|b| if *b { 1.0 } else { 0.0 }).collect();
    let near_hole = crate::filters::dilate_plane(&hole_f, w, h, P as usize);
    let ok: Vec<bool> = near_hole.iter().map(|v| *v < 0.5).collect();
    let sources: Vec<(i32, i32)> = (0..w * h).filter(|i| ok[*i] && valid(hm, w, h, (i % w) as i32, (i / w) as i32, &ok)).map(|i| ((i % w) as i32, (i / w) as i32)).collect();
    if sources.is_empty() {
        return 0.0;
    }
    // Targets: centres whose patch overlaps the hole (and fit inside).
    let targets: Vec<usize> = (0..w * h)
        .filter(|i| {
            let (x, y) = ((i % w) as isize, (i / w) as isize);
            near_hole[*i] > 0.5 && x >= P && y >= P && x < w as isize - P && y < h as isize - P
        })
        .collect();
    if nnf.len() != w * h {
        nnf.clear();
        nnf.resize(w * h, (0, 0));
    }
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (seed + 1).wrapping_mul(0xA24B_AED4_963E_E407));
    for &t in &targets {
        let (sx, sy) = nnf[t];
        if !valid(hm, w, h, sx, sy, &ok) {
            nnf[t] = sources[rng.range(sources.len())];
        }
    }
    let mut mean_err = 0.0;
    for it in 0..iters {
        // PatchMatch: alternate scan order; propagation + random search.
        let fwd = it % 2 == 0;
        let order: Vec<usize> = if fwd { targets.clone() } else { targets.iter().rev().copied().collect() };
        let step: i32 = if fwd { 1 } else { -1 };
        let mut total = 0.0f32;
        for &t in &order {
            let (x, y) = ((t % w) as i32, (t / w) as i32);
            let (mut bx, mut by) = nnf[t];
            let mut bd = dist(cur, x as usize, y as usize, bx as usize, by as usize, f32::MAX);
            for (nx, ny) in [(x - step, y), (x, y - step)] {
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                    continue;
                }
                let (cx, cy) = nnf[ny as usize * w + nx as usize];
                let (cx, cy) = (cx + (x - nx), cy + (y - ny));
                if valid(hm, w, h, cx, cy, &ok) {
                    let d = dist(cur, x as usize, y as usize, cx as usize, cy as usize, bd);
                    if d < bd {
                        (bx, by, bd) = (cx, cy, d);
                    }
                }
            }
            let mut r = w.max(h) as i32;
            while r >= 1 {
                let cx = bx + (rng.range(2 * r as usize + 1) as i32 - r);
                let cy = by + (rng.range(2 * r as usize + 1) as i32 - r);
                if valid(hm, w, h, cx, cy, &ok) {
                    let d = dist(cur, x as usize, y as usize, cx as usize, cy as usize, bd);
                    if d < bd {
                        (bx, by, bd) = (cx, cy, d);
                    }
                }
                r /= 2;
            }
            nnf[t] = (bx, by);
            total += bd;
        }
        mean_err = total / (targets.len().max(1) as f32 * 49.0 * 3.0);
        // Vote: each hole pixel = weighted mean of its covering patches.
        let sigma2 = {
            let mut ds: Vec<f32> = targets.iter().map(|&t| {
                let (x, y) = (t % w, t / w);
                let (sx, sy) = nnf[t];
                dist(cur, x, y, sx as usize, sy as usize, f32::MAX)
            }).collect();
            ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
            ds[(ds.len() * 3 / 4).min(ds.len() - 1)].max(1e-4)
        };
        let mut acc = vec![[0.0f32; 4]; w * h];
        for &t in &targets {
            let (x, y) = ((t % w) as isize, (t / w) as isize);
            let (sx, sy) = nnf[t];
            let d = dist(cur, x as usize, y as usize, sx as usize, sy as usize, f32::MAX);
            let wt = (-d / (2.0 * sigma2)).exp().max(1e-6);
            for dy in -P..=P {
                for dx in -P..=P {
                    let i = ((y + dy) as usize) * w + (x + dx) as usize;
                    if !hm[i] {
                        continue;
                    }
                    let s = src.px[((sy as isize + dy) as usize) * w + (sx as isize + dx) as usize];
                    let a = &mut acc[i];
                    for c in 0..3 {
                        a[c] += wt * s[c];
                    }
                    a[3] += wt;
                }
            }
        }
        cur.px.par_iter_mut().zip(acc.par_iter()).zip(hm.par_iter()).for_each(|((p, a), h)| {
            if *h && a[3] > 0.0 {
                p[0] = a[0] / a[3];
                p[1] = a[1] / a[3];
                p[2] = a[2] / a[3];
            }
        });
    }
    mean_err
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heals_a_stripe_texture() {
        // Diagonal stripes with a square hole: healing should continue the
        // stripes, diffusion would make a flat blob.
        let (w, h) = (160, 120);
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = if ((x + y) / 6) % 2 == 0 { 0.85 } else { 0.15 };
                img.px[y * w + x] = [v, v * 0.8, 0.3, 1.0];
            }
        }
        let truth = img.clone();
        let mut mask = vec![0.0f32; w * h];
        for y in 50..74 {
            for x in 70..94 {
                mask[y * w + x] = 1.0;
                img.px[y * w + x] = [1.0, 0.0, 1.0, 1.0];
            }
        }
        let (healed, rep) = heal(&img, &mask);
        let diffused = crate::edit::inpaint(&img, &mask);
        let err = |a: &Image| -> f32 {
            let mut e = 0.0;
            for y in 50..74 {
                for x in 70..94 {
                    e += (a.px[y * w + x][0] - truth.px[y * w + x][0]).abs();
                }
            }
            e / (24.0 * 24.0)
        };
        let (eh, ed) = (err(&healed), err(&diffused));
        assert!(eh < ed * 0.6, "heal error {eh} vs diffusion {ed} ({rep:?})");
        // Contrast survives (stripes, not grey).
        let c: Vec<f32> = (70..94).map(|x| healed.px[62 * w + x][0]).collect();
        let (mn, mx) = c.iter().fold((1.0f32, 0.0f32), |(a, b), v| (a.min(*v), b.max(*v)));
        assert!(mx - mn > 0.4, "healed stripes lost contrast: {mn}..{mx}");
        assert_eq!(rep.hole_px, 24 * 24);
    }
}
