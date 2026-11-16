# pb-gpu — to do

PixelBoost's GPU inference backend. Overall tracker: ../../TODO.md (5.4).

How it works: `GpuNet` turns a `.pbnn` model into a list of dispatches over
padded-HWC feature-map buffers (same layout as pb-nn's `Fmap`), recorded per
tile shape; `shaders::conv_source` generates an unrolled WGSL kernel per
tile configuration; `tune` picks the configuration per device.

Debug knobs: `PB_GPU=0` (off), `PB_GPU=any` (allow software adapters),
`PB_BACKEND=cpu|gpu`, `PB_GPU_PROFILE=1` (sync after every dispatch, print
per-kernel GFLOP/s), `PB_CFG64|32|16=bm,bk,tm,tn` (force a tile),
`PB_GPU_TUNE=file` (tuning cache for the CLI).

- [x] Conv (implicit GEMM, fused epilogue, upsample-in-index), SRVGG tail.
- [x] Matches pb-nn (tests/match_cpu.rs: random RRDB x4/x2, SRVGG, real
      models when Models/pbnn is present).
- [x] Short submissions; error scopes; CPU fallback; pause / reset.
- [x] Autotune + cache.
- [ ] fp16 storage / `shader-f16` arithmetic where the adapter has it.
- [ ] Double-buffered workgroup tiles (load k+1 while computing k).
- [ ] Persistent bind groups across shape changes (now rebuilt per shape).
- [ ] Split one forward across GPU + CPU for big tiles on desktop.
