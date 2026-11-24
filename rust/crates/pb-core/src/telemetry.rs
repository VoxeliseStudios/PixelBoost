//! Telemetry (see `pb_nn::telemetry`, shared by every PixelBoost crate)
//! plus helpers that need pb-core types.

pub use pb_nn::telemetry::*;

/// Content fingerprint of an image for joining events about the same photo
/// (analysis, upscale, edits, feedback): a hash of its size and a 16x16
/// luma thumbnail. It cannot be turned back into the picture.
pub fn fingerprint(img: &crate::image::Image) -> String {
    let (w, h) = (img.w.max(1), img.h.max(1));
    let mut x: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |b: u64| {
        x ^= b;
        x = x.wrapping_mul(0x100_0000_01b3);
    };
    feed(w as u64);
    feed(h as u64);
    for gy in 0..16 {
        for gx in 0..16 {
            let (x0, x1) = (gx * w / 16, ((gx + 1) * w / 16).max(gx * w / 16 + 1).min(w));
            let (y0, y1) = (gy * h / 16, ((gy + 1) * h / 16).max(gy * h / 16 + 1).min(h));
            let (mut s, mut n) = (0.0f32, 0.0f32);
            let step = ((x1 - x0).max(y1 - y0) / 8).max(1);
            for y in (y0..y1).step_by(step) {
                for xx in (x0..x1).step_by(step) {
                    s += crate::image::Image::luma(img.px[y * w + xx]);
                    n += 1.0;
                }
            }
            feed((s / n.max(1.0) * 63.0).round() as u64);
        }
    }
    format!("{x:016x}")
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fingerprint_is_stable_and_short() {
        let mut img = crate::image::Image::new(40, 30);
        let f1 = fingerprint(&img);
        img.px[5] = [1.0, 1.0, 1.0, 1.0];
        assert_eq!(f1.len(), 16);
        let _ = fingerprint(&img);
        assert_eq!(fingerprint(&crate::image::Image::new(40, 30)), f1);
    }
}
