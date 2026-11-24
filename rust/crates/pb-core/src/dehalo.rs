//! Ringing / halo from earlier processing (sharpening, older upscalers).
//!
//! A sharpened edge overshoots: just beside it the picture is brighter
//! than the bright side and darker than the dark side. Those rims look
//! like fine detail to every other measure (the real-resolution test, the
//! network), so a blown-up and sharpened meme kept its halos and came out
//! soft. Here:
//!
//! - [`halo_score`] samples the profile across strong edges and measures
//!   the overshoot beyond the plateaus a few pixels out, relative to the
//!   edge contrast (0 = clean edges, ~0.15+ = visible halos).
//! - [`dehalo`] clamps every pixel into the range of a smoothed copy's
//!   local neighbourhood (thin rims fall outside it, real texture doesn't
//!   move much), blended by strength and only near edges.

use crate::filters::{box_blur_plane, gauss_plane};
use crate::image::Image;
use rayon::prelude::*;

fn luma(img: &Image) -> Vec<f32> {
    img.px.par_iter().map(|p| Image::luma(*p) * 255.0).collect()
}

/// Halo score at the native size and at 1/2 and 1/3 (a picture blown up
/// before it was sharpened has wide halos): the strongest reading, and the
/// edges counted for it.
pub fn halo_score(img: &Image) -> (f32, usize) {
    let (s, n, _) = halo_scale(img);
    (s, n)
}

/// Score, edges measured, and the scale (1, 2 or 3) the halos live at.
pub fn halo_scale(img: &Image) -> (f32, usize, usize) {
    let (s0, n0) = halo_score_at(img);
    let mut best = (s0, n0, 1usize);
    for f in [2usize, 3] {
        if img.w / f < 48 || img.h / f < 48 {
            break;
        }
        let s = crate::resample::resize(img, img.w / f, img.h / f, crate::resample::Filter::Box);
        let r = halo_score_at(&s);
        if r.0 > best.0 {
            best = (r.0, r.1, f);
        }
    }
    best
}

/// Mean relative overshoot across strong edges (and how many edges were
/// measured), at this size.
pub fn halo_score_at(img: &Image) -> (f32, usize) {
    let (w, h) = (img.w, img.h);
    if w < 24 || h < 24 {
        return (0.0, 0);
    }
    let l = luma(img);
    let at = |x: isize, y: isize| l[(y.clamp(0, h as isize - 1) as usize) * w + x.clamp(0, w as isize - 1) as usize];
    // Candidate edges on a sparse grid: strongest gradient in each 4x4 cell.
    let rows: Vec<(f32, f32)> = (8..h - 8)
        .into_par_iter()
        .step_by(4)
        .map(|y| {
            let (mut acc, mut n) = (0.0f32, 0.0f32);
            for x in (8..w - 8).step_by(4) {
                let (xi, yi) = (x as isize, y as isize);
                let gx = at(xi + 1, yi) - at(xi - 1, yi);
                let gy = at(xi, yi + 1) - at(xi, yi - 1);
                let g = (gx * gx + gy * gy).sqrt();
                if g < 40.0 {
                    continue;
                }
                let (dx, dy) = (gx / g, gy / g);
                let prof: Vec<f32> = (-7..=7).map(|k| {
                    let (sx, sy) = (x as f32 + dx * k as f32, y as f32 + dy * k as f32);
                    at(sx.round() as isize, sy.round() as isize)
                }).collect();
                // Plateaus 5..7 px out on both sides.
                let lo_side = (prof[0] + prof[1] + prof[2]) / 3.0;
                let hi_side = (prof[12] + prof[13] + prof[14]) / 3.0;
                let contrast = (hi_side - lo_side).abs();
                if contrast < 30.0 {
                    continue;
                }
                // Plateaus must be flat (else it's texture, not an edge).
                let flat = |a: f32, b: f32, c: f32| (a - b).abs().max((b - c).abs()) < contrast * 0.15;
                if !flat(prof[0], prof[1], prof[2]) || !flat(prof[12], prof[13], prof[14]) {
                    continue;
                }
                let (mn, mx) = (lo_side.min(hi_side), lo_side.max(hi_side));
                let inner = &prof[3..12];
                // Sharpening overshoots on BOTH sides (bright rim and dark
                // rim); an ink line or a light outline only on one.
                let up = inner.iter().fold(0.0f32, |m, v| m.max(v - mx));
                let down = inner.iter().fold(0.0f32, |m, v| m.max(mn - v));
                let over = 2.0 * up.min(down);
                acc += (over / contrast).min(1.0);
                n += 1.0;
            }
            (acc, n)
        })
        .collect();
    let (acc, n) = rows.iter().fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    if n < 12.0 {
        return (0.0, n as usize);
    }
    (acc / n, n as usize)
}

/// Strength to use for a measured score (0 = leave alone).
pub fn strength_for(score: f32) -> f32 {
    crate::filters::smoothstep(0.10, 0.24, score)
}

/// Remove halos: near edges, clamp each pixel into the range of the
/// plateaus 4-5 px away on both sides across the edge (the same profile
/// the detector measures). Texture has no flat plateaus, so its range is
/// wide and it barely moves.
pub fn dehalo(img: &Image, strength: f32) -> Image {
    let k = strength.clamp(0.0, 1.0);
    if k <= 0.0 {
        return img.clone();
    }
    let (w, h) = (img.w, img.h);
    // Work at the scale the halos were found at (wide halos on pictures
    // that were blown up before they were sharpened).
    let probe = if w > 768 || h > 768 { img.crop((w - w.min(768)) / 2, (h - h.min(768)) / 2, w.min(768), h.min(768)) } else { img.clone() };
    let f = halo_scale(&probe).2 as f32;
    let l = gauss_plane(&luma(img), w, h, 1.2 * f);
    let g: Vec<f32> = (0..w * h)
        .into_par_iter()
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let a = |xx: usize, yy: usize| l[yy * w + xx];
            let gx = a((x + 1).min(w - 1), y) - a(x.saturating_sub(1), y);
            let gy = a(x, (y + 1).min(h - 1)) - a(x, y.saturating_sub(1));
            (gx * gx + gy * gy).sqrt()
        })
        .collect();
    // Edge pixels and their 3 px surroundings carry the rims.
    let edge: Vec<f32> = g.par_iter().map(|v| ((v * f - 12.0) / 30.0).clamp(0.0, 1.0)).collect();
    let near = crate::filters::dilate_plane(&edge, w, h, (3.0 * f) as usize);
    let near = box_blur_plane(&near, w, h, f as usize);
    // Edge direction near each pixel: gradient of a wider blur.
    let lw = gauss_plane(&l, w, h, 2.0 * f);
    let step = 2isize * f as isize;
    let mut out = img.clone();
    out.px.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, p) in row.iter_mut().enumerate() {
            let t = k * near[y * w + x];
            if t <= 0.01 {
                continue;
            }
            let a = |xx: isize, yy: isize| lw[(yy.clamp(0, h as isize - 1) as usize) * w + xx.clamp(0, w as isize - 1) as usize];
            let (xi, yi) = (x as isize, y as isize);
            let (gx, gy) = (a(xi + step, yi) - a(xi - step, yi), a(xi, yi + step) - a(xi, yi - step));
            let gl = (gx * gx + gy * gy).sqrt();
            if gl < 1e-3 {
                continue;
            }
            let (dx, dy) = (gx / gl, gy / gl);
            let mut lo = [f32::MAX; 3];
            let mut hi = [f32::MIN; 3];
            for d in [-5.0f32, -4.0, 4.0, 5.0] {
                let q = crate::filters::sample_bilinear(img, x as f32 + dx * d * f, y as f32 + dy * d * f);
                for c in 0..3 {
                    lo[c] = lo[c].min(q[c]);
                    hi[c] = hi[c].max(q[c]);
                }
            }
            for c in 0..3 {
                let v = p[c];
                let cl = v.clamp(lo[c] - 0.004, hi[c] + 0.004);
                p[c] = v + (cl - v) * t;
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(w: usize, h: usize, halo: f32) -> Image {
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                // A few vertical edges, with sharpening overshoot of `halo`.
                let p = x % 32;
                let mut v = if p < 16 { 0.25 } else { 0.75 };
                if p == 15 || p == 14 {
                    v -= halo * 0.5;
                }
                if p == 16 || p == 17 {
                    v += halo * 0.5;
                }
                if p == 31 || p == 30 {
                    v += halo * 0.5;
                }
                if p == 0 || p == 1 {
                    v -= halo * 0.5;
                }
                let _ = y;
                img.px[y * w + x] = [v.clamp(0.0, 1.0), v.clamp(0.0, 1.0), v.clamp(0.0, 1.0), 1.0];
            }
        }
        img
    }

    #[test]
    fn detects_and_removes_halos() {
        let clean = step(256, 96, 0.0);
        let halo = step(256, 96, 0.3);
        let (s0, n0) = halo_score(&clean);
        let (s1, n1) = halo_score(&halo);
        assert!(n0 > 20 && n1 > 20, "edges found {n0} {n1}");
        assert!(s0 < 0.03, "clean edges score {s0}");
        assert!(s1 > 0.15, "haloed edges score {s1}");
        let fixed = dehalo(&halo, strength_for(s1));
        let (s2, _) = halo_score(&fixed);
        assert!(s2 < s1 * 0.5, "after dehalo {s2} (was {s1})");
        // The edge itself stays put.
        assert!((fixed.px[48 * 256 + 8][0] - 0.25).abs() < 0.03);
        assert!((fixed.px[48 * 256 + 24][0] - 0.75).abs() < 0.03);
    }
}
