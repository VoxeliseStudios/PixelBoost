//! IEEE 754 binary16 <-> binary32 conversion (round-to-nearest-even).

/// Expand a run of finite f16 values (weights): branch-free bit math the
/// compiler vectorises (exact for normals, subnormals and zero).
pub fn expand(src: &[u16], dst: &mut [f32]) {
    let magic = f32::from_bits((254 - 15) << 23);
    for (d, s) in dst.iter_mut().zip(src) {
        let h = *s as u32;
        let v = f32::from_bits((h & 0x7fff) << 13) * magic;
        *d = f32::from_bits(v.to_bits() | ((h & 0x8000) << 16));
    }
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // Subnormal: normalise.
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let man = x & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1 << shift) - 1);
        let mut r = m >> shift;
        if rem > half || (rem == half && r & 1 == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        r += 1;
    }
    sign | r as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip() {
        for v in [0.0f32, -0.0, 1.0, -2.5, 0.2, 65504.0, 1e-5, 6.1e-5, -3.25, 1234.5] {
            let r = f16_to_f32(f32_to_f16(v));
            assert!((r - v).abs() <= v.abs() * 1e-3 + 1e-7, "{v} -> {r}");
        }
        for h in 0u16..0x7c00 {
            assert_eq!(f32_to_f16(f16_to_f32(h)), h);
            if (h & 0x7c00) != 0x7c00 {
                let mut d = [0.0f32];
                expand(&[h], &mut d);
                assert_eq!(d[0].to_bits(), f16_to_f32(h).to_bits(), "{h:04x}");
            }
        }
    }
}
