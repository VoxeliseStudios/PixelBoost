//! Animated pictures and video frames: upscale frame by frame without
//! flicker.
//!
//! GAN upscalers invent texture independently on every frame, so a still
//! background shimmers. [`Temporal`] keeps, per source pixel, the source
//! value its current output was made from; where a new frame hasn't
//! really changed there, the previous output is kept (blended in softly
//! at the boundary of moving areas), so only what moves is redrawn.
//! Identical frames are not upscaled again at all.
//!
//! Files: GIF, APNG and animated WebP in; APNG (lossless) or GIF out,
//! streamed frame by frame so memory stays at a couple of frames. Video
//! frames come from the app (AVFoundation decodes/encodes) through the
//! same [`Temporal`] stage.

use crate::fusion::Upscaler;
use crate::image::Image;
use crate::pipeline::Settings;
use crate::stream::{self, Limits, Rgba8};
use rayon::prelude::*;

pub struct Frame {
    pub img: Rgba8,
    pub delay_ms: u32,
}

/// Decode every frame (composited, full size) of a GIF / APNG / WebP.
/// A still picture comes back as one frame.
pub fn decode(bytes: &[u8]) -> Result<Vec<Frame>, String> {
    use image::AnimationDecoder;
    let cur = std::io::Cursor::new(bytes);
    let fmt = image::guess_format(bytes).map_err(|e| e.to_string())?;
    let frames: Vec<image::Frame> = match fmt {
        image::ImageFormat::Gif => image::codecs::gif::GifDecoder::new(cur).map_err(|e| e.to_string())?.into_frames().collect_frames().map_err(|e| e.to_string())?,
        image::ImageFormat::Png => {
            let d = image::codecs::png::PngDecoder::new(cur).map_err(|e| e.to_string())?;
            if d.is_apng().map_err(|e| e.to_string())? {
                d.apng().map_err(|e| e.to_string())?.into_frames().collect_frames().map_err(|e| e.to_string())?
            } else {
                let i = image::load_from_memory(bytes).map_err(|e| e.to_string())?.into_rgba8();
                vec![image::Frame::new(i)]
            }
        }
        image::ImageFormat::WebP => {
            let d = image::codecs::webp::WebPDecoder::new(cur).map_err(|e| e.to_string())?;
            if d.has_animation() {
                d.into_frames().collect_frames().map_err(|e| e.to_string())?
            } else {
                vec![image::Frame::new(image::load_from_memory(bytes).map_err(|e| e.to_string())?.into_rgba8())]
            }
        }
        _ => vec![image::Frame::new(image::load_from_memory(bytes).map_err(|e| e.to_string())?.into_rgba8())],
    };
    Ok(frames
        .into_iter()
        .map(|f| {
            let (n, d) = f.delay().numer_denom_ms();
            let delay_ms = if d == 0 { 100 } else { (n / d).max(10) };
            let b = f.into_buffer();
            let (w, h) = (b.width() as usize, b.height() as usize);
            Frame { img: Rgba8 { w, h, data: b.into_raw() }, delay_ms }
        })
        .collect())
}

/// Frame count without keeping the frames (0 if not an animation).
pub fn frame_count(bytes: &[u8]) -> usize {
    decode(bytes).map(|f| f.len()).unwrap_or(0)
}

fn to_image(r: &Rgba8) -> Image {
    Image { w: r.w, h: r.h, px: r.data.par_chunks_exact(4).map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0, c[3] as f32 / 255.0]).collect() }
}

/// Temporal stabiliser + upscaler state for one clip.
pub struct Temporal {
    /// Per source pixel: the source colour the current output came from.
    reference: Option<Rgba8>,
    prev_out: Option<Rgba8>,
    prev_src: Option<Rgba8>,
    /// Change (0..255 levels) below which a pixel counts as unchanged.
    pub threshold: f32,
    pub frames: usize,
    pub reused: usize,
    pub static_share: f64,
    pub ms: f64,
}

impl Default for Temporal {
    fn default() -> Self {
        Temporal { reference: None, prev_out: None, prev_src: None, threshold: 6.0, frames: 0, reused: 0, static_share: 0.0, ms: 0.0 }
    }
}

impl Temporal {
    /// Upscale one frame. `progress` gets the tile events (return false to
    /// cancel).
    pub fn push(&mut self, engine: &dyn Upscaler, src: &Rgba8, st: &Settings, lim: Limits, progress: &mut dyn FnMut(stream::StreamEvent<'_>) -> bool) -> Option<Rgba8> {
        let t0 = std::time::Instant::now();
        self.frames += 1;
        // Same frame as before: same output.
        if let (Some(p), Some(o)) = (&self.prev_src, &self.prev_out) {
            if p.w == src.w && p.h == src.h && p.data == src.data {
                self.reused += 1;
                self.static_share += 1.0;
                return Some(o.clone());
            }
        }
        let (out, _, _) = stream::upscale(engine, &to_image(src), st, lim, progress)?;
        let out = match (&mut self.reference, &self.prev_out) {
            (Some(rf), Some(po)) if rf.w == src.w && rf.h == src.h && po.w == out.w && po.h == out.h => {
                let (w, h) = (src.w, src.h);
                // Motion map: change against the reference, softened.
                let d: Vec<f32> = rf.data.par_chunks_exact(4).zip(src.data.par_chunks_exact(4)).map(|(a, b)| (0..3).map(|k| (a[k] as f32 - b[k] as f32).abs()).fold(0.0, f32::max)).collect();
                let thr = self.threshold;
                let m0: Vec<f32> = d.par_iter().map(|v| crate::filters::smoothstep(thr * 0.6, thr * 1.6, *v)).collect();
                let m = crate::filters::box_blur_plane(&crate::filters::dilate_plane(&m0, w, h, 1), w, h, 1);
                self.static_share += m.iter().filter(|v| **v < 0.02).count() as f64 / m.len().max(1) as f64;
                // Update the reference where the output is redrawn.
                rf.data.par_chunks_exact_mut(4).zip(src.data.par_chunks_exact(4)).zip(m.par_iter()).for_each(|((r, s), t)| {
                    if *t > 0.5 {
                        r.copy_from_slice(s);
                    }
                });
                let (ow, oh) = (out.w, out.h);
                let mut blended = out;
                blended.data.par_chunks_mut(ow * 4).zip(po.data.par_chunks(ow * 4)).enumerate().for_each(|(y, (row, prow))| {
                    let sy = ((y as f32 + 0.5) * h as f32 / oh as f32 - 0.5).max(0.0);
                    for x in 0..ow {
                        let sx = ((x as f32 + 0.5) * w as f32 / ow as f32 - 0.5).max(0.0);
                        let t = crate::filters::sample_plane(&m, w, h, sx, sy);
                        if t >= 0.999 {
                            continue;
                        }
                        for k in 0..4 {
                            let (a, b) = (prow[x * 4 + k] as f32, row[x * 4 + k] as f32);
                            row[x * 4 + k] = (a + (b - a) * t).round() as u8;
                        }
                    }
                });
                blended
            }
            _ => {
                self.reference = Some(src.clone());
                out
            }
        };
        self.prev_src = Some(src.clone());
        self.prev_out = Some(out.clone());
        self.ms += t0.elapsed().as_secs_f64() * 1000.0;
        Some(out)
    }

    pub fn report(&self, what: &str, model: &str) {
        crate::telemetry::event("animation", what)
            .ms(self.ms)
            .tag("model", model)
            .metric("frames", self.frames as f64)
            .metric("reused", self.reused as f64)
            .metric("static_share", self.static_share / self.frames.max(1) as f64)
            .metric("ms_per_frame", self.ms / self.frames.max(1) as f64)
            .emit();
    }
}

/// Streaming writer for the upscaled frames.
pub enum Writer {
    Png(png::Writer<std::io::BufWriter<std::fs::File>>),
    Gif(image::codecs::gif::GifEncoder<std::io::BufWriter<std::fs::File>>),
}

impl Writer {
    /// `path` ending in .gif writes GIF (256 colours per frame), anything
    /// else APNG.
    pub fn create(path: &str, w: usize, h: usize, frames: usize) -> Result<Writer, String> {
        let f = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(path).map_err(|e| format!("{path}: {e}"))?);
        if path.to_lowercase().ends_with(".gif") {
            let mut g = image::codecs::gif::GifEncoder::new_with_speed(f, 10);
            g.set_repeat(image::codecs::gif::Repeat::Infinite).map_err(|e| e.to_string())?;
            return Ok(Writer::Gif(g));
        }
        let mut e = png::Encoder::new(f, w as u32, h as u32);
        e.set_color(png::ColorType::Rgba);
        e.set_depth(png::BitDepth::Eight);
        e.set_compression(png::Compression::Fast);
        if frames > 1 {
            e.set_animated(frames as u32, 0).map_err(|e| e.to_string())?;
        }
        Ok(Writer::Png(e.write_header().map_err(|e| e.to_string())?))
    }

    pub fn frame(&mut self, img: &Rgba8, delay_ms: u32) -> Result<(), String> {
        match self {
            Writer::Png(w) => {
                let _ = w.set_frame_delay(delay_ms.min(65535) as u16, 1000);
                w.write_image_data(&img.data).map_err(|e| e.to_string())
            }
            Writer::Gif(g) => {
                let buf = image::RgbaImage::from_raw(img.w as u32, img.h as u32, img.data.clone()).ok_or("frame size")?;
                let fr = image::Frame::from_parts(buf, 0, 0, image::Delay::from_numer_denom_ms(delay_ms, 1));
                g.encode_frame(fr).map_err(|e| e.to_string())
            }
        }
    }

    pub fn finish(self) -> Result<(), String> {
        match self {
            Writer::Png(w) => w.finish().map_err(|e| e.to_string()),
            Writer::Gif(g) => {
                drop(g);
                Ok(())
            }
        }
    }
}

/// Upscale a whole animated file. `on_frame(done, total)` returns false to
/// cancel. Returns (width, height, frames).
pub fn upscale_file(engine: &dyn Upscaler, input: &str, output: &str, st: &Settings, lim: Limits, on_frame: &mut dyn FnMut(usize, usize) -> bool) -> Result<(usize, usize, usize), String> {
    let bytes = std::fs::read(input).map_err(|e| format!("{input}: {e}"))?;
    let frames = decode(&bytes)?;
    drop(bytes);
    let n = frames.len();
    let mut tp = Temporal::default();
    let mut writer: Option<Writer> = None;
    let mut size = (0, 0);
    for (i, f) in frames.iter().enumerate() {
        let out = tp.push(engine, &f.img, st, lim, &mut |_| true).ok_or("cancelled")?;
        if writer.is_none() {
            size = (out.w, out.h);
            writer = Some(Writer::create(output, out.w, out.h, n)?);
        }
        writer.as_mut().unwrap().frame(&out, f.delay_ms)?;
        if !on_frame(i + 1, n) {
            return Err("cancelled".into());
        }
    }
    writer.ok_or("no frames")?.finish()?;
    tp.report("file", &engine.label());
    Ok((size.0, size.1, n))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trivial 2x "network" with per-call noise (like a GAN's invented
    /// texture) to test the stabiliser.
    struct Noisy(std::sync::atomic::AtomicU32);
    impl Upscaler for Noisy {
        fn scale(&self) -> usize {
            2
        }
        fn forward(&self, r: &Image) -> Image {
            let k = self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut o = crate::resample::resize(r, r.w * 2, r.h * 2, crate::resample::Filter::Box);
            for (i, p) in o.px.iter_mut().enumerate() {
                let n = crate::filters::hash_noise(i as u32, k, 3) * 0.05;
                p[0] = (p[0] + n).clamp(0.0, 1.0);
            }
            o
        }
        fn label(&self) -> String {
            "noisy".into()
        }
        fn macs_per_px(&self) -> f64 {
            1.0
        }
    }

    #[test]
    fn static_areas_do_not_flicker_and_motion_updates() {
        let (w, h) = (64, 48);
        let mk = |shift: usize| {
            let mut r = Rgba8::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    let moving = (x >= 10 + shift && x < 20 + shift) && y < 16;
                    let v = if moving { 230 } else { (40 + (x * 3 + y * 2) % 60) as u8 };
                    r.data[(y * w + x) * 4..(y * w + x) * 4 + 4].copy_from_slice(&[v, v, v, 255]);
                }
            }
            r
        };
        let eng = Noisy(Default::default());
        let st = Settings { scale: 2.0, fidelity: crate::fidelity::Fidelity::Off, detail: 0.0, tile: 64, ..Settings::default() };
        let mut tp = Temporal::default();
        let a = tp.push(&eng, &mk(0), &st, Limits::default(), &mut |_| true).unwrap();
        let b = tp.push(&eng, &mk(4), &st, Limits::default(), &mut |_| true).unwrap();
        // Static background (bottom half) identical between frames.
        let row = |o: &Rgba8, y: usize| o.data[y * o.w * 4..(y + 1) * o.w * 4].to_vec();
        assert_eq!(row(&a, 80), row(&b, 80));
        // Moving block redrawn: new position is bright in b.
        let px = |o: &Rgba8, x: usize, y: usize| o.data[(y * o.w + x) * 4];
        assert!(px(&b, 2 * 22, 10) > 200, "moved block not drawn");
        // Same frame again is reused.
        let c = tp.push(&eng, &mk(4), &st, Limits::default(), &mut |_| true).unwrap();
        assert_eq!(c.data, b.data);
        assert_eq!(tp.reused, 1);
    }

    #[test]
    fn apng_round_trip() {
        let dir = std::env::temp_dir().join(format!("pbanim-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.png");
        let mut wr = Writer::create(p.to_str().unwrap(), 8, 6, 3).unwrap();
        for k in 0..3u8 {
            let mut f = Rgba8::new(8, 6);
            f.data.iter_mut().step_by(4).for_each(|v| *v = k * 80);
            wr.frame(&f, 120).unwrap();
        }
        wr.finish().unwrap();
        let fr = decode(&std::fs::read(&p).unwrap()).unwrap();
        assert_eq!(fr.len(), 3);
        assert_eq!(fr[2].img.data[0], 160);
        assert_eq!(fr[1].delay_ms, 120);
        let g = dir.join("a.gif");
        let mut wr = Writer::create(g.to_str().unwrap(), 8, 6, 3).unwrap();
        for f in &fr {
            wr.frame(&f.img, f.delay_ms).unwrap();
        }
        wr.finish().unwrap();
        assert_eq!(decode(&std::fs::read(&g).unwrap()).unwrap().len(), 3);
        let _ = std::fs::remove_dir_all(dir);
    }
}
