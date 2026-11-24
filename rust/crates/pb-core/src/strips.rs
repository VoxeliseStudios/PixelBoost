//! Full-resolution edits with bounded memory.
//!
//! The editor applies one op per step to the full photo. A 48 MP photo as
//! f32 RGBA is 770 MB, and the general path needs the input and the output
//! at once. Pointwise ops (light & colour, curves, HSL, grading, looks,
//! LUTs, film grain, vignette) don't need the whole picture: they run here
//! on 256-row strips of the 8-bit image, so the peak is the 8-bit picture
//! (4 B/px) plus one strip. The intermediate files are 8-bit already, so
//! the result is identical to the whole-image path.
//!
//! Everything else goes through [`apply_owned`], which edits in place where
//! the op allows it instead of cloning first.

use crate::edit::Region;
use crate::image::{Format, Image};
use crate::ops::{self, Op};
use crate::stream::Rgba8;
use rayon::prelude::*;

const STRIP: usize = 256;

/// Can this op run strip by strip with an identical result?
pub fn strip_safe(op: &Op) -> bool {
    match op {
        Op::Adjust(a) => crate::edit::adjust_is_pointwise(a),
        Op::Look { .. } | Op::Hsl(_) | Op::Grade(_) | Op::Curves(_) | Op::Lut { .. } => true,
        Op::Chain { ops } => !ops.is_empty() && ops.iter().all(strip_safe),
        _ => false,
    }
}

/// Apply a strip-safe op to one strip that starts at row `y0` of a
/// `full_w` x `full_h` picture.
fn apply_strip(img: &mut Image, op: &Op, reg: Region, lut: &mut Option<crate::edit::Lut3d>) -> Result<(), String> {
    match op {
        Op::Adjust(a) => crate::edit::adjust_at(img, a, reg),
        Op::Look { name, amount } => {
            if !crate::edit::LOOKS.contains(&name.as_str()) {
                return Err(format!("unknown look {name}"));
            }
            crate::edit::apply_look(img, name, *amount)
        }
        Op::Hsl(p) => crate::develop::hsl(img, p),
        Op::Grade(g) => crate::develop::grade(img, g),
        Op::Curves(c) => crate::develop::curves(img, c),
        Op::Lut { path, amount } => {
            if lut.is_none() {
                let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
                *lut = Some(crate::edit::parse_cube(&text)?);
            }
            crate::edit::apply_lut(img, lut.as_ref().unwrap(), *amount)
        }
        Op::Chain { ops } => {
            for o in ops {
                // Chains rarely hold more than one LUT; parse per strip then.
                let mut l = None;
                apply_strip(img, o, reg, if matches!(o, Op::Lut { .. }) { &mut l } else { lut })?;
            }
        }
        _ => return Err("op is not strip-safe".into()),
    }
    Ok(())
}

/// Decode straight to 8-bit RGBA (4 B/px).
pub fn open_rgba8(path: &str) -> Result<Rgba8, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let img = image::load_from_memory(&bytes).map_err(|e| format!("decode: {e}"))?;
    drop(bytes);
    let (w, h) = (img.width() as usize, img.height() as usize);
    let data = img.into_rgba8().into_raw();
    Ok(Rgba8 { w, h, data })
}

/// Run a strip-safe op over an 8-bit picture in place.
pub fn apply_rgba8(buf: &mut Rgba8, op: &Op) -> Result<(), String> {
    let (w, h) = (buf.w, buf.h);
    let mut lut = None;
    for y0 in (0..h).step_by(STRIP) {
        let n = STRIP.min(h - y0);
        let rows = &mut buf.data[y0 * w * 4..(y0 + n) * w * 4];
        let mut img = Image { w, h: n, px: rows.par_chunks_exact(4).map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0, c[3] as f32 / 255.0]).collect() };
        apply_strip(&mut img, op, Region { y0, full_w: w, full_h: h }, &mut lut)?;
        rows.par_chunks_exact_mut(4).zip(img.px.par_iter()).for_each(|(d, p)| {
            for k in 0..4 {
                d[k] = (p[k].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        });
    }
    Ok(())
}

/// Ops that change the picture in place (no input copy needed).
fn in_place(img: &mut Image, op: &Op) -> Option<Result<(), String>> {
    let r = match op {
        Op::Adjust(a) => {
            crate::edit::adjust(img, a);
            Ok(())
        }
        Op::Look { name, amount } => {
            if !crate::edit::LOOKS.contains(&name.as_str()) {
                Err(format!("unknown look {name}"))
            } else {
                crate::edit::apply_look(img, name, *amount);
                Ok(())
            }
        }
        Op::AutoEnhance => {
            crate::edit::auto_enhance(img);
            Ok(())
        }
        Op::Hsl(p) => {
            crate::develop::hsl(img, p);
            Ok(())
        }
        Op::Grade(g) => {
            crate::develop::grade(img, g);
            Ok(())
        }
        Op::Curves(c) => {
            crate::develop::curves(img, c);
            Ok(())
        }
        Op::Draw { marks } => {
            crate::draw::draw(img, marks);
            Ok(())
        }
        Op::Text(t) => {
            crate::text::draw_text(img, t);
            Ok(())
        }
        Op::Sharpen { amount, radius } => {
            crate::edit::sharpen(img, *amount, *radius);
            Ok(())
        }
        _ => return None,
    };
    Some(r)
}

/// Apply an op consuming the input: in-place ops reuse its memory, chains
/// never hold more than two pictures.
pub fn apply_owned(mut img: Image, op: &Op) -> Result<Image, String> {
    if let Op::Chain { ops } = op {
        for o in ops {
            img = apply_owned(img, o)?;
        }
        return Ok(img);
    }
    let t0 = std::time::Instant::now();
    let px = img.w * img.h;
    match in_place(&mut img, op) {
        Some(r) => {
            if crate::telemetry::enabled() {
                ops::report(op, px, t0, r.as_ref().map(|_| ()).map_err(|e| e.as_str()));
            }
            r.map(|_| img)
        }
        None => ops::apply(&img, op),
    }
}

/// Edit a file at full resolution with the least memory: strips for
/// pointwise ops, in-place / consuming application for the rest. Returns
/// (width, height, has_alpha) of the result.
pub fn edit_file(input: &str, output: &str, op: &Op, quality: u8) -> Result<(usize, usize, bool), String> {
    let fmt = Format::from_path(output);
    let t0 = std::time::Instant::now();
    if strip_safe(op) && fmt != Format::Tiff {
        let mut buf = open_rgba8(input)?;
        apply_rgba8(&mut buf, op)?;
        buf.save(output, quality)?;
        let a = buf.has_alpha();
        let (_, peak) = crate::telemetry::memory_mb();
        crate::telemetry::event("editor", "full").since(t0).tag("op", ops::op_name(op)).tag("path", "strips").metric("mp", (buf.w * buf.h) as f64 / 1e6).metric("mem_peak_mb", peak).emit();
        return Ok((buf.w, buf.h, a));
    }
    let img = Image::open(input)?;
    let mp = (img.w * img.h) as f64 / 1e6;
    let out = apply_owned(img, op)?;
    let bytes = out.encode(fmt, quality)?;
    std::fs::write(output, bytes).map_err(|e| format!("{output}: {e}"))?;
    let (_, peak) = crate::telemetry::memory_mb();
    crate::telemetry::event("editor", "full").since(t0).tag("op", ops::op_name(op)).tag("path", "image").metric("mp", mp).metric("mem_peak_mb", peak).emit();
    Ok((out.w, out.h, out.has_alpha()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn photo(w: usize, h: usize) -> Image {
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let t = (x as f32 / w as f32, y as f32 / h as f32);
                img.px[y * w + x] = [t.0, t.1, (t.0 * 7.0 + t.1 * 3.0).sin() * 0.5 + 0.5, 1.0];
            }
        }
        img
    }

    fn q(img: &Image) -> Image {
        Image { w: img.w, h: img.h, px: img.px.iter().map(|p| p.map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5).floor() / 255.0)).collect() }
    }

    #[test]
    fn strips_match_the_whole_image() {
        // 600 rows = 3 strips (256 + 256 + 88); vignette + grain depend on
        // the position in the picture.
        let img = q(&photo(320, 600));
        let op: Op = ops::parse(r#"{"op":"chain","ops":[
            {"op":"adjust","exposure":0.3,"vignette":0.6,"grain":0.5,"grain_size":0.4,"saturation":1.2,"curve":[0,0.2,0.5,0.8,1]},
            {"op":"look","name":"warm","amount":0.7},
            {"op":"hsl","red":{"hue":0.1,"sat":0.2,"lum":0.0}}]}"#)
        .unwrap();
        assert!(strip_safe(&op));
        let whole = ops::apply(&img, &op).unwrap();
        let mut buf = Rgba8 { w: img.w, h: img.h, data: img.to_rgba8() };
        apply_rgba8(&mut buf, &op).unwrap();
        let want = whole.to_rgba8();
        let worst = buf.data.iter().zip(&want).map(|(a, b)| (*a as i32 - *b as i32).abs()).max().unwrap();
        assert!(worst <= 1, "strip result differs by {worst} levels");
        assert!(!strip_safe(&ops::parse(r#"{"op":"adjust","clarity":0.5}"#).unwrap()));
        assert!(!strip_safe(&ops::parse(r#"{"op":"blur","radius":3}"#).unwrap()));
    }

    #[test]
    fn owned_equals_borrowed() {
        let img = photo(64, 48);
        for j in [r#"{"op":"adjust","contrast":1.3}"#, r#"{"op":"blur","radius":2}"#, r#"{"op":"chain","ops":[{"op":"sharpen","amount":0.5},{"op":"rotate","clockwise":true}]}"#] {
            let op = ops::parse(j).unwrap();
            let a = ops::apply(&img, &op).unwrap();
            let b = apply_owned(img.clone(), &op).unwrap();
            assert_eq!((a.w, a.h), (b.w, b.h));
            assert!(a.px.iter().zip(&b.px).all(|(p, q)| (0..4).all(|k| (p[k] - q[k]).abs() < 1e-6)), "{j}");
        }
    }
}
