//! WGSL kernels, generated per tile shape.
//!
//! `conv3x3` is an implicit GEMM: M = output pixels, N = output channels,
//! K = 9 taps x input channels. A workgroup owns a BM-pixel x BN-channel
//! output block; each step stages BK input channels of one tap (gathered
//! straight from the padded HWC map, no im2col buffer) and the matching
//! BK x BN weight slab in workgroup memory, then every thread accumulates a
//! TM x TN register tile. The epilogue fuses bias, LeakyReLU / PReLU,
//! scaling and two residual adds, exactly like the CPU kernels in pb-nn.
//!
//! The `UP` variant reads its input through a nearest-neighbour 2x upsample
//! (index arithmetic only), so the 2x / 4x upsampled feature maps of RRDBNet
//! never exist in memory.

/// One conv kernel configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConvCfg {
    pub bm: u32,
    pub bn: u32,
    pub bk: u32,
    pub tm: u32,
    pub tn: u32,
    pub up: bool,
    /// Input channels load as vec4 (cin, channel stride and offset all
    /// multiples of 4).
    pub vec: bool,
}

impl ConvCfg {
    /// Block width used to pad a conv's output channels.
    pub fn bn_for(cout: usize) -> usize {
        Self::pick(cout, false, false).bn as usize
    }

    pub fn threads(&self) -> u32 {
        (self.bm / self.tm) * (self.bn / self.tn)
    }

    /// Tile shapes per output-channel block width.
    pub fn pick(cout: usize, up: bool, vec: bool) -> ConvCfg {
        let bn = if cout % 64 == 0 {
            64
        } else if cout % 32 == 0 {
            32
        } else {
            16
        };
        let t = tiles()[slot(bn)];
        ConvCfg { bm: t.bm, bn: bn as u32, bk: t.bk, tm: t.tm, tn: t.tn, up, vec }
    }
}

/// Workgroup tile shape for one output-channel block width.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tile {
    pub bm: u32,
    pub bk: u32,
    pub tm: u32,
    pub tn: u32,
}

impl Tile {
    pub const fn new(bm: u32, bk: u32, tm: u32, tn: u32) -> Tile {
        Tile { bm, bk, tm, tn }
    }

    /// Whether the generator can build this shape for block width `bn`.
    pub fn valid(&self, bn: u32) -> bool {
        let (bm, bk, tm, tn) = (self.bm, self.bk, self.tm, self.tn);
        if bm == 0 || tm == 0 || tn == 0 || bk == 0 || bm % tm != 0 || bn % tn != 0 || tm % 4 != 0 || tn % 4 != 0 || bk % 4 != 0 {
            return false;
        }
        let nt = (bm / tm) * (bn / tn);
        // Both the vec4 (bk/4 lanes) and scalar (bk lanes) loaders must tile.
        (32..=256).contains(&nt) && [bk / 4, bk].iter().all(|&kw| nt % kw == 0 && (kw * bm) % nt == 0) && bk * (bm + bn) * 4 <= 16384
    }
}

/// Block widths in table order.
pub const WIDTHS: [u32; 3] = [64, 32, 16];

/// Defaults (tuned on an RDNA2 iGPU; `tune` replaces them per device).
pub const DEFAULT_TILES: [Tile; 3] = [Tile::new(128, 16, 8, 8), Tile::new(64, 16, 8, 4), Tile::new(128, 16, 4, 4)];

fn slot(bn: usize) -> usize {
    match bn {
        64 => 0,
        32 => 1,
        _ => 2,
    }
}

static TILES: std::sync::RwLock<Option<[Tile; 3]>> = std::sync::RwLock::new(None);

/// Current tile table (`PB_CFG64=bm,bk,tm,tn` etc. override the defaults).
pub fn tiles() -> [Tile; 3] {
    if let Some(t) = *TILES.read().unwrap() {
        return t;
    }
    let mut t = DEFAULT_TILES;
    for (i, bn) in WIDTHS.iter().enumerate() {
        if let Ok(v) = std::env::var(format!("PB_CFG{bn}")) {
            let n: Vec<u32> = v.split(',').filter_map(|x| x.parse().ok()).collect();
            if n.len() == 4 && Tile::new(n[0], n[1], n[2], n[3]).valid(*bn) {
                t[i] = Tile::new(n[0], n[1], n[2], n[3]);
            }
        }
    }
    *TILES.write().unwrap() = Some(t);
    t
}

pub fn set_tiles(t: [Tile; 3]) {
    *TILES.write().unwrap() = Some(t);
}

/// Uniform block shared by every kernel (layout must match `Params` in
/// net.rs: 32 x 4-byte fields).
const PARAMS: &str = r#"
struct P {
  h: u32, w: u32, in_w: u32, in_c: u32,
  cin_off: u32, cin: u32, out_c: u32, cout_off: u32,
  cout: u32, cout_pad: u32, act: u32, slope: f32,
  scale: f32, wofs: u32, bofs: u32, aofs: u32,
  r1_on: u32, r1_c: u32, r1_off: u32, r1_coef: f32,
  r2_on: u32, r2_c: u32, r2_off: u32, r2_coef: f32,
  blocks_x: u32, r: u32, p2: u32, p3: u32,
  p4: u32, p5: u32, p6: u32, p7: u32,
}
@group(0) @binding(0) var<storage, read> wts: array<vec4<f32>>;
$SRCDECL
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read_write> r1: array<f32>;
@group(0) @binding(4) var<storage, read_write> r2: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
"#;

/// SRVGGNetCompact tail: pixel shuffle of the last conv's output plus the
/// nearest-upsampled input (`r1`), written as a padded 3-channel map.
const TAIL: &str = r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = p.r;
  let W = p.w * r;
  let H = p.h * r;
  let idx = gid.y * (p.blocks_x * 64u) + gid.x;
  if (idx >= W * H) {
    return;
  }
  let Y = idx / W;
  let X = idx % W;
  let lp = (Y / r + 1u) * (p.w + 2u) + X / r + 1u;
  let sub = (Y % r) * r + X % r;
  let o = ((Y + 1u) * (W + 2u) + X + 1u) * 3u;
  for (var ch = 0u; ch < 3u; ch++) {
    dst[o + ch] = src[lp * p.in_c + ch * r * r + sub] + r1[lp * 3u + ch];
  }
}
"#;

/// Generate the conv kernel for one configuration. Everything a thread
/// keeps in registers (accumulators, staged coordinates, operand vectors)
/// is unrolled into named variables: GPU compilers turn dynamically indexed
/// private arrays into slow scratch memory.
pub fn conv_source(c: ConvCfg) -> String {
    use std::fmt::Write;
    let kw = if c.vec { c.bk / 4 } else { c.bk };
    assert!(c.bm % c.tm == 0 && c.bn % c.tn == 0 && c.tm % 4 == 0 && c.tn % 4 == 0 && c.bk % 4 == 0);
    let nt = c.threads();
    assert!(nt % kw == 0 && (kw * c.bm) % nt == 0, "{c:?}");
    let la = c.bm * kw / nt;
    let mstep = nt / kw;
    let (tm4, tn4) = (c.tm / 4, c.tn / 4);
    let (bm, bn, bk) = (c.bm, c.bn, c.bk);
    let mut o = String::new();
    let (decl, load) = if c.vec {
        ("@group(0) @binding(1) var<storage, read_write> src4: array<vec4<f32>>;", "src4[(I) / 4u]")
    } else {
        ("@group(0) @binding(1) var<storage, read_write> src: array<f32>;", "vec4<f32>(src[I], 0.0, 0.0, 0.0)")
    };
    o.push_str(&PARAMS.replace("$SRCDECL", decl));
    // Scalar array: threads writing different lanes of one shared vec4 race
    // on Apple GPUs (lane stores compile to read-modify-write of the vector).
    let _ = writeln!(o, "var<workgroup> As: array<f32, {}>;", bk * bm);
    let _ = writeln!(o, "var<workgroup> Bs: array<vec4<f32>, {}>;", bk * bn / 4);
    let _ = writeln!(o, "@compute @workgroup_size({nt})");
    o.push_str("fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {\n");
    o.push_str("  let blk = wg.y * p.blocks_x + wg.x + p.p2;\n  if (blk >= p.p3) { return; }\n");
    let _ = writeln!(o, "  let m0 = blk * {bm}u;\n  let n0 = wg.z * {bn}u;\n  let hw = p.h * p.w;");
    let _ = writeln!(o, "  let tn = tid % {}u;\n  let tm = tid / {}u;\n  let kl = tid % {kw}u;", bn / c.tn, bn / c.tn);
    o.push_str("  let in_rs = (p.in_w + 2u) * p.in_c;\n");
    for t in 0..la {
        let _ = writeln!(o, "  let am{t} = tid / {kw}u + {}u;", t * mstep);
        let _ = writeln!(o, "  let ok{t} = m0 + am{t} < hw;\n  let g{t} = min(m0 + am{t}, hw - 1u);\n  let ly{t} = g{t} / p.w;\n  let lx{t} = g{t} % p.w;");
    }
    for i in 0..c.tm {
        for q in 0..tn4 {
            let _ = writeln!(o, "  var c{i}_{q} = vec4<f32>(0.0);");
        }
    }
    o.push_str("  for (var tap = 0u; tap < 9u; tap++) {\n    let ky = tap / 3u;\n    let kx = tap % 3u;\n");
    o.push_str("    for (var c0 = 0u; c0 < p.cin; c0 += BKu) {\n".replace("BK", &bk.to_string()).as_str());
    if c.vec {
        o.push_str("      let ci = c0 + kl * 4u;\n");
    } else {
        o.push_str("      let ci = c0 + kl;\n");
    }
    for t in 0..la {
        let (row, col) = if c.up {
            (format!("(ly{t} + ky + 1u) / 2u"), format!("(lx{t} + kx + 1u) / 2u"))
        } else {
            (format!("ly{t} + ky"), format!("lx{t} + kx"))
        };
        let idx = format!("({row}) * in_rs + ({col}) * p.in_c + p.cin_off + ci");
        let _ = writeln!(o, "      var v{t} = vec4<f32>(0.0);\n      if (ok{t} && ci < p.cin) {{ v{t} = {}; }}", load.replace('I', &idx));
        if c.vec {
            for e in 0..4 {
                let comp = ["x", "y", "z", "w"][e];
                let _ = writeln!(o, "      As[(kl * 4u + {e}u) * {bm}u + am{t}] = v{t}.{comp};");
            }
        } else {
            let _ = writeln!(o, "      As[kl * {bm}u + am{t}] = v{t}.x;");
        }
    }
    let _ = writeln!(o, "      for (var e = tid; e < {}u; e += {nt}u) {{", bk * bn / 4);
    let _ = writeln!(o, "        let nq = e % {}u;\n        let k = e / {}u;", bn / 4, bn / 4);
    o.push_str("        var v = vec4<f32>(0.0);\n        if (c0 + k < p.cin) { v = wts[(p.wofs + (tap * p.cin + c0 + k) * p.cout_pad + n0) / 4u + nq]; }\n        Bs[e] = v;\n      }\n");
    o.push_str("      workgroupBarrier();\n");
    let _ = writeln!(o, "      let abase = tm * {}u;\n      let bbase = tn * {tn4}u;", c.tm);
    for k in 0..bk {
        for i4 in 0..tm4 {
            let b0 = format!("{}u + abase + {}u", k * bm, i4 * 4);
            let _ = writeln!(o, "      let a{k}_{i4} = vec4<f32>(As[{b0}], As[{b0} + 1u], As[{b0} + 2u], As[{b0} + 3u]);");
        }
        for q in 0..tn4 {
            let _ = writeln!(o, "      let b{k}_{q} = Bs[{}u + bbase + {q}u];", k * bn / 4);
        }
        for i in 0..c.tm {
            let comp = ["x", "y", "z", "w"][(i % 4) as usize];
            for q in 0..tn4 {
                let _ = writeln!(o, "      c{i}_{q} = fma(vec4<f32>(a{k}_{}.{comp}), b{k}_{q}, c{i}_{q});", i / 4);
            }
        }
    }
    o.push_str("      workgroupBarrier();\n    }\n  }\n");
    // Epilogue.
    for i in 0..c.tm {
        let _ = writeln!(o, "  {{\n    let gm = m0 + tm * {}u + {i}u;\n    if (gm < hw) {{", c.tm);
        o.push_str("      let pix = (gm / p.w + 1u) * (p.w + 2u) + gm % p.w + 1u;\n");
        for q in 0..tn4 {
            let _ = writeln!(o, "      epi(pix, n0 + tn * {}u + {}u, c{i}_{q});", c.tn, q * 4);
        }
        o.push_str("    }\n  }\n");
    }
    o.push_str("}\n");
    o.push_str(EPI);
    o
}

/// Bias, activation, scale, residuals and store for four channels.
const EPI: &str = r#"
fn epi(pix: u32, n: u32, a: vec4<f32>) {
  for (var j = 0u; j < 4u; j++) {
    let ch = n + j;
    if (ch >= p.cout) {
      return;
    }
    let bi = p.bofs + ch;
    var v = a[j] + wts[bi / 4u][bi % 4u];
    if (p.act == 1u) {
      if (v < 0.0) { v = v * p.slope; }
    } else if (p.act == 2u) {
      let ai = p.aofs + ch;
      if (v < 0.0) { v = v * wts[ai / 4u][ai % 4u]; }
    }
    v = v * p.scale;
    if (p.r1_on != 0u) {
      v = v + p.r1_coef * r1[pix * p.r1_c + p.r1_off + ch];
    }
    if (p.r2_on != 0u) {
      v = v + p.r2_coef * r2[pix * p.r2_c + p.r2_off + ch];
    }
    dst[pix * p.out_c + p.cout_off + ch] = v;
  }
}
"#;

pub fn tail_source() -> String {
    format!("{}{TAIL}", PARAMS.replace("$SRCDECL", "@group(0) @binding(1) var<storage, read_write> src: array<f32>;"))
}
