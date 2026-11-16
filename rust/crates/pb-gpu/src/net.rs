//! Networks as GPU programs: a list of conv / tail dispatches over a fixed
//! set of feature-map buffers, recorded into one command buffer per call.

use crate::shaders::ConvCfg;
use crate::Gpu;
use pb_nn::fmap::{pixel_unshuffle, Fmap};
use pb_nn::pbm::Weights;
use pb_nn::SrNetwork;
use std::sync::{Arc, Mutex};

pub(crate) const PARAMS_BYTES: u64 = 128;

/// A feature map buffer: `mul` x the network's working resolution, `ch`
/// channels, padded HWC like `pb_nn::Fmap`.
#[derive(Clone, Copy)]
struct BufSpec {
    mul: usize,
    ch: usize,
}

struct ConvW {
    cin: usize,
    cout: usize,
    cout_pad: usize,
    /// Float offsets into the weight arena.
    wofs: usize,
    bofs: usize,
    aofs: usize,
}

#[derive(Clone, Copy)]
enum Act {
    None,
    Lrelu(f32),
    Prelu,
}

#[derive(Clone, Copy)]
struct Op {
    /// `None` = the SRVGG pixel-shuffle tail.
    conv: Option<usize>,
    src: usize,
    cin_off: usize,
    dst: usize,
    cout_off: usize,
    act: Act,
    scale: f32,
    res: [Option<(usize, usize, f32)>; 2],
    up: bool,
}

impl Op {
    fn conv(conv: usize, src: usize, dst: usize) -> Op {
        Op { conv: Some(conv), src, cin_off: 0, dst, cout_off: 0, act: Act::None, scale: 1.0, res: [None, None], up: false }
    }
    fn act(mut self, a: Act) -> Op {
        self.act = a;
        self
    }
    fn at(mut self, cin_off: usize, cout_off: usize) -> Op {
        self.cin_off = cin_off;
        self.cout_off = cout_off;
        self
    }
    fn up(mut self) -> Op {
        self.up = true;
        self
    }
    fn epi(mut self, scale: f32, r1: Option<(usize, usize, f32)>, r2: Option<(usize, usize, f32)>) -> Op {
        self.scale = scale;
        self.res = [r1, r2];
        self
    }
}

struct Program {
    scale: usize,
    unshuffle: usize,
    bufs: Vec<BufSpec>,
    ops: Vec<Op>,
    out: usize,
    convs: Vec<ConvW>,
    arena: Vec<f32>,
    macs: f64,
}

/// GPU buffers sized for the largest tile seen so far.
struct State {
    shape: (usize, usize),
    caps: Vec<u64>,
    bufs: Vec<wgpu::Buffer>,
    staging: wgpu::Buffer,
    params: Option<wgpu::Buffer>,
    groups: Vec<wgpu::BindGroup>,
    parts: Vec<Part>,
}

/// Rebuilds the CPU network if the GPU fails mid-run.
pub type CpuFallback = Box<dyn Fn() -> Result<Box<dyn SrNetwork>, String> + Send + Sync>;

pub struct GpuNet {
    gpu: Arc<Gpu>,
    prog: Program,
    wts: wgpu::Buffer,
    dummy: wgpu::Buffer,
    state: Mutex<Option<State>>,
    fallback: CpuFallback,
    cpu: Mutex<Option<Arc<dyn SrNetwork>>>,
    /// This network's GPU path failed: CPU from now on.
    failed: std::sync::atomic::AtomicBool,
}

fn tensor<'a>(w: &'a Weights, name: &str) -> Result<&'a pb_nn::pth::TensorF32, String> {
    w.get(name)
}

impl Program {
    fn add_conv(&mut self, w: &Weights, name: &str, alpha: Option<&str>) -> Result<usize, String> {
        let wt = tensor(w, &format!("{name}.weight"))?;
        if wt.shape.len() != 4 || wt.shape[2] != 3 || wt.shape[3] != 3 {
            return Err(format!("{name}: expected a 3x3 conv, got {:?}", wt.shape));
        }
        let (cout, cin) = (wt.shape[0], wt.shape[1]);
        let bn = ConvCfg::bn_for(cout);
        let cout_pad = cout.div_ceil(bn) * bn;
        let wofs = self.arena.len();
        self.arena.resize(wofs + 9 * cin * cout_pad, 0.0);
        for co in 0..cout {
            for ci in 0..cin {
                for k in 0..9 {
                    self.arena[wofs + (k * cin + ci) * cout_pad + co] = wt.data[(co * cin + ci) * 9 + k];
                }
            }
        }
        let bofs = self.arena.len();
        self.arena.resize(bofs + cout_pad, 0.0);
        if let Ok(b) = tensor(w, &format!("{name}.bias")) {
            self.arena[bofs..bofs + cout].copy_from_slice(&b.data[..cout]);
        }
        let mut aofs = 0;
        if let Some(a) = alpha {
            let a = tensor(w, a)?;
            aofs = self.arena.len();
            self.arena.resize(aofs + cout_pad, 0.0);
            self.arena[aofs..aofs + a.data.len().min(cout)].copy_from_slice(&a.data[..a.data.len().min(cout)]);
        }
        self.convs.push(ConvW { cin, cout, cout_pad, wofs, bofs, aofs });
        Ok(self.convs.len() - 1)
    }

    fn buf(&mut self, mul: usize, ch: usize) -> usize {
        self.bufs.push(BufSpec { mul, ch });
        self.bufs.len() - 1
    }

    fn new(scale: usize, unshuffle: usize) -> Program {
        Program { scale, unshuffle, bufs: Vec::new(), ops: Vec::new(), out: 0, convs: Vec::new(), arena: Vec::new(), macs: 0.0 }
    }

    fn rrdb(w: &Weights) -> Result<Program, String> {
        const SLOPE: f32 = 0.2;
        let n = w.cfg("num_block").ok_or("num_block missing")? as usize;
        let scale = w.cfg("scale").unwrap_or(4) as usize;
        let unshuffle = match scale {
            4 => 1,
            2 => 2,
            1 => 4,
            s => return Err(format!("unsupported RRDB scale {s}")),
        };
        let mut p = Program::new(scale, unshuffle);
        let first = p.add_conv(w, "conv_first", None)?;
        let (inc, f) = (p.convs[first].cin, p.convs[first].cout);
        let mut blocks = Vec::with_capacity(n);
        for i in 0..n {
            let mut b = [[0usize; 5]; 3];
            for (r, rdb) in b.iter_mut().enumerate() {
                for (k, c) in rdb.iter_mut().enumerate() {
                    *c = p.add_conv(w, &format!("body.{i}.rdb{}.conv{}", r + 1, k + 1), None)?;
                }
            }
            blocks.push(b);
        }
        let g = blocks.first().map(|b| p.convs[b[0][0]].cout).unwrap_or(32);
        let body = p.add_conv(w, "conv_body", None)?;
        let up1 = p.add_conv(w, "conv_up1", None)?;
        let up2 = p.add_conv(w, "conv_up2", None)?;
        let hr = p.add_conv(w, "conv_hr", None)?;
        let last = p.add_conv(w, "conv_last", None)?;
        let dc = f + 4 * g;

        let input = p.buf(1, inc);
        let d = [p.buf(1, dc), p.buf(1, dc), p.buf(1, dc)];
        let feat = p.buf(1, f);
        let trunk = p.buf(1, f);
        let a2 = p.buf(2, f);
        let b4 = p.buf(4, f);
        let c4 = p.buf(4, f);
        let out = p.buf(4, p.convs[last].cout);
        p.out = out;

        let lr = Act::Lrelu(SLOPE);
        p.ops.push(Op::conv(first, input, feat));
        p.ops.push(Op::conv(first, input, d[0]));
        let mut cur = 0;
        for blk in &blocks {
            let (a, b, c) = (cur, (cur + 1) % 3, (cur + 2) % 3);
            let rdb = |p: &mut Program, convs: &[usize; 5], buf: usize| {
                for (k, &cv) in convs.iter().take(4).enumerate() {
                    p.ops.push(Op::conv(cv, buf, buf).act(lr).at(0, f + k * g));
                }
            };
            rdb(&mut p, &blk[0], d[a]);
            p.ops.push(Op::conv(blk[0][4], d[a], d[b]).epi(0.2, Some((d[a], 0, 1.0)), None));
            rdb(&mut p, &blk[1], d[b]);
            p.ops.push(Op::conv(blk[1][4], d[b], d[c]).epi(0.2, Some((d[b], 0, 1.0)), None));
            rdb(&mut p, &blk[2], d[c]);
            p.ops.push(Op::conv(blk[2][4], d[c], d[b]).epi(0.04, Some((d[c], 0, 0.2)), Some((d[a], 0, 1.0))));
            cur = b;
        }
        p.ops.push(Op::conv(body, d[cur], trunk).epi(1.0, Some((feat, 0, 1.0)), None));
        p.ops.push(Op::conv(up1, trunk, a2).act(lr).up());
        p.ops.push(Op::conv(up2, a2, b4).act(lr).up());
        p.ops.push(Op::conv(hr, b4, c4).act(lr));
        p.ops.push(Op::conv(last, c4, out));

        let m = |c: &ConvW| (c.cin * c.cout * 9) as f64;
        let lr_macs: f64 = p.ops.iter().filter(|o| p.bufs[o.dst].mul == 1).filter_map(|o| o.conv).map(|c| m(&p.convs[c])).sum::<f64>() - m(&p.convs[first]);
        let up_macs = m(&p.convs[up1]) * 4.0 + (m(&p.convs[up2]) + m(&p.convs[hr]) + m(&p.convs[last])) * 16.0;
        p.macs = (lr_macs + up_macs) / (unshuffle * unshuffle) as f64;
        Ok(p)
    }

    fn srvgg(w: &Weights) -> Result<Program, String> {
        let n = w.cfg("num_conv").ok_or("num_conv missing")? as usize;
        let scale = w.cfg("scale").unwrap_or(4) as usize;
        let mut p = Program::new(scale, 1);
        let mut convs = Vec::new();
        for i in 0..n + 2 {
            let alpha = if i < n + 1 { Some(format!("body.{}.weight", i * 2 + 1)) } else { None };
            convs.push(p.add_conv(w, &format!("body.{}", i * 2), alpha.as_deref())?);
        }
        let f = p.convs[convs[0]].cout;
        let input = p.buf(1, 3);
        let a = p.buf(1, f);
        let b = p.buf(1, f);
        let last = *convs.last().unwrap();
        let o = p.buf(1, p.convs[last].cout);
        let out = p.buf(scale, 3);
        p.out = out;
        p.ops.push(Op::conv(convs[0], input, a).act(Act::Prelu));
        let (mut x, mut y) = (a, b);
        for &cv in &convs[1..convs.len() - 1] {
            p.ops.push(Op::conv(cv, x, y).act(Act::Prelu));
            std::mem::swap(&mut x, &mut y);
        }
        p.ops.push(Op::conv(last, x, o));
        p.ops.push(Op { conv: None, src: o, cin_off: 0, dst: out, cout_off: 0, act: Act::None, scale: 1.0, res: [Some((input, 0, 1.0)), None], up: false });
        p.macs = convs.iter().map(|&c| (p.convs[c].cin * p.convs[c].cout * 9) as f64).sum();
        Ok(p)
    }

    fn cfg(&self, op: &Op) -> ConvCfg {
        let c = &self.convs[op.conv.expect("conv op")];
        let ch = self.bufs[op.src].ch;
        ConvCfg::pick(c.cout, op.up, c.cin % 4 == 0 && ch % 4 == 0 && op.cin_off % 4 == 0)
    }

    fn bytes(&self, i: usize, h: usize, w: usize) -> u64 {
        let b = self.bufs[i];
        ((h * b.mul + 2) * (w * b.mul + 2) * b.ch * 4) as u64
    }
}

/// Work per dispatch part and per queue submission. Small submissions keep
/// any one GPU job short (desktop / iOS watchdogs reset jobs that run for
/// seconds) and let the display keep drawing between them.
const PART_FLOP: f64 = 4e9;
const SUBMIT_FLOP: f64 = 12e9;

/// One dispatch: its uniform block, workgroup counts and cost.
struct Part {
    op: usize,
    /// Kernel the dispatch counts were computed for (`None` = tail).
    cfg: Option<ConvCfg>,
    v: [u32; 32],
    groups: [u32; 3],
    flop: f64,
}

fn parts(p: &Program, k: usize, h: usize, w: usize, out: &mut Vec<Part>) {
    let op = &p.ops[k];
    let mut v = [0u32; 32];
    let src = p.bufs[op.src];
    let dst = p.bufs[op.dst];
    let (oh, ow) = (h * dst.mul, w * dst.mul);
    let grid = |n: usize| {
        let bx = n.clamp(1, 32768);
        [bx as u32, n.div_ceil(bx).max(1) as u32]
    };
    match op.conv {
        Some(ci) => {
            let c = &p.convs[ci];
            let cfg = p.cfg(op);
            v[0] = oh as u32;
            v[1] = ow as u32;
            v[2] = (w * src.mul) as u32;
            v[3] = src.ch as u32;
            v[4] = op.cin_off as u32;
            v[5] = c.cin as u32;
            v[6] = dst.ch as u32;
            v[7] = op.cout_off as u32;
            v[8] = c.cout as u32;
            v[9] = c.cout_pad as u32;
            let (act, slope) = match op.act {
                Act::None => (0, 0.0),
                Act::Lrelu(s) => (1, s),
                Act::Prelu => (2, 0.0),
            };
            v[10] = act;
            v[11] = f32::to_bits(slope);
            v[12] = f32::to_bits(op.scale);
            v[13] = c.wofs as u32;
            v[14] = c.bofs as u32;
            v[15] = c.aofs as u32;
            for (r, res) in op.res.iter().enumerate() {
                if let Some((b, off, coef)) = res {
                    v[16 + r * 4] = 1;
                    v[17 + r * 4] = p.bufs[*b].ch as u32;
                    v[18 + r * 4] = *off as u32;
                    v[19 + r * 4] = f32::to_bits(*coef);
                }
            }
            let blocks = (oh * ow).div_ceil(cfg.bm as usize);
            let per_block = (cfg.bm as usize * c.cout_pad * c.cin * 9 * 2) as f64;
            let step = ((PART_FLOP / per_block) as usize).max(1);
            let zb = (c.cout_pad / cfg.bn as usize) as u32;
            let mut b0 = 0;
            while b0 < blocks {
                let n = step.min(blocks - b0);
                let [bx, by] = grid(n);
                let mut pv = v;
                pv[24] = bx;
                pv[26] = b0 as u32;
                // Blocks past `b0 + n` in the last grid row would redo the
                // next part's pixels: harmless (same values), but bound them.
                pv[27] = (b0 + n) as u32;
                out.push(Part { op: k, cfg: Some(cfg), v: pv, groups: [bx, by, zb], flop: per_block * n as f64 });
                b0 += n;
            }
        }
        None => {
            // Tail: h, w are the low-res dims of `src`.
            let r = dst.mul / src.mul;
            v[0] = (h * src.mul) as u32;
            v[1] = (w * src.mul) as u32;
            v[3] = src.ch as u32;
            v[25] = r as u32;
            let [bx, by] = grid((h * w * r * r).div_ceil(64));
            v[24] = bx;
            out.push(Part { op: k, cfg: None, v, groups: [bx, by, 1], flop: 0.0 });
        }
    }
}

impl GpuNet {
    /// Build the GPU program for a `.pbnn` model's weights.
    pub fn new(gpu: Arc<Gpu>, w: &Weights, fallback: CpuFallback) -> Result<GpuNet, String> {
        let prog = match w.header.arch.as_str() {
            "rrdb" => Program::rrdb(w)?,
            "srvgg" => Program::srvgg(w)?,
            a => return Err(format!("unknown architecture {a}")),
        };
        let bytes: &[u8] = bytemuck_f32(&prog.arena);
        if bytes.len() as u64 > gpu.max_binding {
            return Err(format!("weights ({} MB) exceed the GPU's binding limit", bytes.len() >> 20));
        }
        use wgpu::util::DeviceExt;
        let wts = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("weights"), contents: bytes, usage: wgpu::BufferUsages::STORAGE });
        let dummy = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("dummy"), size: 16, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        // Compile every kernel now so the first tile isn't slow.
        for op in &prog.ops {
            match op.conv {
                Some(_) => {
                    gpu.conv_pipeline(prog.cfg(op));
                }
                None => {
                    gpu.tail_pipeline();
                }
            }
        }
        let mut prog = prog;
        prog.arena = Vec::new();
        Ok(GpuNet { gpu, prog, wts, dummy, state: Mutex::new(None), fallback, cpu: Mutex::new(None), failed: Default::default() })
    }

    /// A test network for tuning: `reps` copies of one conv (cin -> cout)
    /// reading the same input, so one forward is `reps` x the conv's work.
    pub(crate) fn synthetic(gpu: Arc<Gpu>, cin: usize, cout: usize, reps: usize) -> Result<GpuNet, String> {
        let mut prog = Program::new(1, 1);
        let bn = ConvCfg::bn_for(cout);
        let cout_pad = cout.div_ceil(bn) * bn;
        prog.arena = (0..9 * cin * cout_pad + cout_pad).map(|i| ((i * 7919 % 211) as f32 / 211.0 - 0.5) * 0.05).collect();
        prog.convs.push(ConvW { cin, cout, cout_pad, wofs: 0, bofs: 9 * cin * cout_pad, aofs: 0 });
        let a = prog.buf(1, cin);
        let b = prog.buf(1, cout);
        prog.out = b;
        for _ in 0..reps {
            prog.ops.push(Op::conv(0, a, b));
        }
        prog.macs = (cin * cout * 9 * reps) as f64;
        use wgpu::util::DeviceExt;
        let wts = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("tune weights"), contents: bytemuck_f32(&prog.arena), usage: wgpu::BufferUsages::STORAGE });
        let dummy = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("dummy"), size: 16, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        prog.arena = Vec::new();
        Ok(GpuNet { gpu, prog, wts, dummy, state: Mutex::new(None), fallback: Box::new(|| Err("synthetic".into())), cpu: Mutex::new(None), failed: Default::default() })
    }

    /// Run on the GPU only (no fallback), for tuning.
    pub(crate) fn try_forward(&self, x: &Fmap) -> Result<Fmap, String> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.run_gpu(x))).map_err(|_| "GPU panicked".to_string())?
    }

    pub fn device_name(&self) -> String {
        self.gpu.describe()
    }

    /// Largest buffer this tile needs, in bytes.
    fn peak(&self, h: usize, w: usize) -> u64 {
        (0..self.prog.bufs.len()).map(|i| self.prog.bytes(i, h, w)).max().unwrap_or(0)
    }

    fn cpu_net(&self) -> Result<Arc<dyn SrNetwork>, String> {
        let mut g = self.cpu.lock().unwrap();
        if g.is_none() {
            *g = Some(Arc::from((self.fallback)()?));
        }
        Ok(g.as_ref().unwrap().clone())
    }

    /// Make the buffers big enough for an `h` x `w` input and the dispatch
    /// list match that shape. Returns true when the shape changed.
    fn ensure(&self, st: &mut Option<State>, h: usize, w: usize) -> bool {
        let gpu = &self.gpu;
        let p = &self.prog;
        let need: Vec<u64> = (0..p.bufs.len()).map(|i| p.bytes(i, h, w).max(16)).collect();
        let grow = match st {
            None => true,
            Some(s) => s.caps.iter().zip(&need).any(|(c, n)| n > c),
        };
        if grow {
            // Grow by 25% so slightly larger edge tiles don't realloc.
            let caps: Vec<u64> = match st {
                Some(s) => s.caps.iter().zip(&need).map(|(c, n)| if n > c { (n + n / 4).min(gpu.max_binding).div_ceil(4) * 4 } else { *c }).collect(),
                None => need.clone(),
            };
            *st = None;
            let bufs: Vec<wgpu::Buffer> = caps
                .iter()
                .enumerate()
                .map(|(i, &c)| {
                    gpu.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("fmap{i}")),
                        size: c,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    })
                })
                .collect();
            let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: caps[p.out],
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            *st = Some(State { shape: (0, 0), caps, bufs, staging, params: None, groups: Vec::new(), parts: Vec::new() });
        }
        let s = st.as_mut().unwrap();
        if s.shape == (h, w) {
            return false;
        }
        s.parts.clear();
        for k in 0..p.ops.len() {
            parts(p, k, h, w, &mut s.parts);
        }
        let stride = gpu.param_stride as usize;
        let mut bytes = vec![0u8; stride * s.parts.len()];
        for (k, part) in s.parts.iter().enumerate() {
            for (i, x) in part.v.iter().enumerate() {
                bytes[k * stride + i * 4..k * stride + i * 4 + 4].copy_from_slice(&x.to_le_bytes());
            }
        }
        let params = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&params, 0, &bytes);
        let bufs = &s.bufs;
        s.groups = s
            .parts
            .iter()
            .enumerate()
            .map(|(k, part)| {
                let op = &p.ops[part.op];
                let res = |i: usize| -> wgpu::BindingResource<'_> {
                    match op.res[i] {
                        Some((b, _, _)) => bufs[b].as_entire_binding(),
                        None => self.dummy.as_entire_binding(),
                    }
                };
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &gpu.layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.wts.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: bufs[op.src].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: bufs[op.dst].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: res(0) },
                        wgpu::BindGroupEntry { binding: 4, resource: res(1) },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &params, offset: gpu.param_stride * k as u64, size: wgpu::BufferSize::new(PARAMS_BYTES) }),
                        },
                    ],
                })
            })
            .collect();
        s.params = Some(params);
        s.shape = (h, w);
        true
    }

    fn run_gpu(&self, x: &Fmap) -> Result<Fmap, String> {
        let gpu = &self.gpu;
        let p = &self.prog;
        let (h, w) = (x.h, x.w);
        let profile = std::env::var("PB_GPU_PROFILE").is_ok();
        let mut guard = self.state.lock().unwrap();
        let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let oom = gpu.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let reshaped = self.ensure(&mut guard, h, w);
        let s = guard.as_ref().unwrap();
        let new_enc = || gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("forward") });
        let mut enc = new_enc();
        if reshaped {
            // New tile shape: zero every map so the padding ring is zero.
            // (Not the input: its queued upload, padding included, lands
            // before this command buffer runs.)
            for b in &s.bufs[1..] {
                enc.clear_buffer(b, 0, None);
            }
        }
        let in_len = (h + 2) * (w + 2) * x.c;
        gpu.queue.write_buffer(&s.bufs[0], 0, bytemuck_f32(&x.data[..in_len]));
        let mut acc = 0.0;
        let mut prof: Vec<(String, f64, f64)> = Vec::new();
        let mut k = 0;
        while k < s.parts.len() {
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("net"), timestamp_writes: None });
                while k < s.parts.len() {
                    let part = &s.parts[k];
                    let pipe = match part.cfg {
                        Some(cfg) => gpu.conv_pipeline(cfg),
                        None => gpu.tail_pipeline(),
                    };
                    pass.set_pipeline(&pipe);
                    pass.set_bind_group(0, &s.groups[k], &[]);
                    let [dx, dy, dz] = part.groups;
                    pass.dispatch_workgroups(dx, dy, dz);
                    acc += part.flop;
                    k += 1;
                    if profile || acc >= SUBMIT_FLOP {
                        break;
                    }
                }
            }
            if k < s.parts.len() {
                let t = std::time::Instant::now();
                gpu.queue.submit([std::mem::replace(&mut enc, new_enc()).finish()]);
                if profile {
                    gpu.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| e.to_string())?;
                    let part = &s.parts[k - 1];
                    let op = &p.ops[part.op];
                    let name = match (op.conv, part.cfg) {
                        (Some(c), Some(cfg)) => format!("{cfg:?} {}->{}", p.convs[c].cin, p.convs[c].cout),
                        _ => "tail".into(),
                    };
                    prof.push((name, part.flop, t.elapsed().as_secs_f64()));
                }
                acc = 0.0;
            }
        }
        if profile {
            let mut agg: std::collections::BTreeMap<String, (f64, f64, usize)> = Default::default();
            for (n, f, t) in prof {
                let e = agg.entry(n).or_default();
                e.0 += f;
                e.1 += t;
                e.2 += 1;
            }
            for (n, (f, t, c)) in agg {
                eprintln!("  {n:<60} x{c:<4} {:>8.1} ms {:>7.1} GFLOP/s", t * 1e3, f / t / 1e9);
            }
        }
        let out_bytes = p.bytes(p.out, h, w);
        enc.copy_buffer_to_buffer(&s.bufs[p.out], 0, &s.staging, 0, Some(out_bytes));
        gpu.queue.submit([enc.finish()]);
        let slice = s.staging.slice(..out_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| e.to_string())?;
        if let Some(e) = pollster::block_on(oom.pop()) {
            return Err(e.to_string());
        }
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(e.to_string());
        }
        rx.recv().map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
        let ob = p.bufs[p.out];
        let mut out = Fmap::new(h * ob.mul, w * ob.mul, ob.ch);
        {
            let view = slice.get_mapped_range();
            let n = out_bytes as usize / 4;
            for (d, c) in out.data[..n].iter_mut().zip(view.chunks_exact(4)) {
                *d = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        s.staging.unmap();
        Ok(out)
    }

    /// Forward a (already unshuffled) map, splitting it if one pass would
    /// exceed the GPU's buffer limit.
    fn forward_lr(&self, x: &Fmap) -> Result<Fmap, String> {
        if self.peak(x.h, x.w) <= self.gpu.max_binding || (x.h < 64 && x.w < 64) {
            return self.run_gpu(x);
        }
        // Halve the longer side, with a context overlap.
        const OV: usize = 16;
        let s = self.prog.bufs[self.prog.out].mul;
        let vertical = x.h >= x.w;
        let n = if vertical { x.h } else { x.w };
        let mid = n / 2;
        let parts = [(0, (mid + OV).min(n)), (mid.saturating_sub(OV), n)];
        let mut outs = Vec::new();
        for &(a, b) in &parts {
            let sub = if vertical { sub_fmap(x, a, 0, b - a, x.w) } else { sub_fmap(x, 0, a, x.h, b - a) };
            outs.push(self.forward_lr(&sub)?);
        }
        let mut out = Fmap::new(x.h * s, x.w * s, outs[0].c);
        let c = out.c;
        for y in 0..out.h {
            for xx in 0..out.w {
                let (k, sy, sx) = if vertical {
                    if y < mid * s {
                        (0, y, xx)
                    } else {
                        (1, y - parts[1].0 * s, xx)
                    }
                } else if xx < mid * s {
                    (0, y, xx)
                } else {
                    (1, y, xx - parts[1].0 * s)
                };
                let o = out.off(y as isize, xx as isize);
                let i = outs[k].off(sy as isize, sx as isize);
                out.data[o..o + c].copy_from_slice(&outs[k].data[i..i + c]);
            }
        }
        Ok(out)
    }
}

fn sub_fmap(x: &Fmap, y0: usize, x0: usize, h: usize, w: usize) -> Fmap {
    let mut o = Fmap::new(h, w, x.c);
    for y in 0..h {
        let s = x.off((y0 + y) as isize, x0 as isize);
        let d = o.off(y as isize, 0);
        o.data[d..d + w * x.c].copy_from_slice(&x.data[s..s + w * x.c]);
    }
    o
}

fn bytemuck_f32(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding or invalid bit patterns; u8 alignment is 1.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

impl SrNetwork for GpuNet {
    fn scale(&self) -> usize {
        self.prog.scale
    }
    fn align(&self) -> usize {
        self.prog.unshuffle
    }
    fn macs_per_px(&self) -> f64 {
        self.prog.macs
    }
    fn backend(&self) -> String {
        if self.failed.load(std::sync::atomic::Ordering::Relaxed) {
            "cpu (gpu failed)".into()
        } else {
            format!("gpu: {}", self.gpu.describe())
        }
    }
    fn forward(&self, x: &Fmap) -> Fmap {
        use std::sync::atomic::Ordering;
        if !self.failed.load(Ordering::Relaxed) && !crate::paused() && !self.gpu.is_lost() {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if self.prog.unshuffle > 1 {
                    self.forward_lr(&pixel_unshuffle(x, self.prog.unshuffle))
                } else {
                    self.forward_lr(x)
                }
            }));
            match r {
                Ok(Ok(o)) => return o,
                Ok(Err(e)) => {
                    self.failed.store(true, Ordering::Relaxed);
                    self.gpu.mark_broken(&e)
                }
                Err(_) => {
                    // A poisoned state lock must not take the CPU path down too.
                    self.state.clear_poison();
                    self.failed.store(true, Ordering::Relaxed);
                    self.gpu.mark_broken("GPU call panicked (device lost?)")
                }
            }
        }
        match self.cpu_net() {
            Ok(n) => n.forward(x),
            Err(e) => panic!("GPU failed and the CPU fallback could not load: {e}"),
        }
    }
}
