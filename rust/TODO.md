# rust/ — to do

Overall tracker: ../RUST-REMAKE.md

- [x] GPU backend: `crates/pb-gpu` (own WGSL kernels on wgpu: Metal on
      iOS, Vulkan here), autotuned per device — see 5.4 below.
- [ ] fp16 / int8 NEON kernel variants (halve weight bandwidth on iPhone).
- [ ] Own segmentation network for Cutout (needs strided conv, pooling,
      bilinear upsample in pb-nn).
- [ ] Render-denoise network (OIDN U-Net) — same new ops as above.
- [x] Perceptual metric for `pixelboost eval`: `texture_match` (per-patch
      fine-detail level vs ground truth; punishes blur *and* invented grain).
- [ ] Classifier: learn thresholds from more labelled gallery images
      (`pixelboost classify`).

## Photo editor (5.2)

- [x] filters / develop / mask / fx / draw / text / compose / presets
      modules + pb-ffi editor API (see ../RUST-REMAKE.md, 5.2).
- [ ] Zoom / spin blur are the slowest previews (~140 ms at 1280 px here):
      sample count could scale with the streak length.
- [ ] Full-resolution edits hold the image as f32 RGBA (16 B/px): a 48 MP
      photo needs ~770 MB per copy. Stream / tile the pointwise ops or use
      f16 planes for big images.
- [ ] Healing brush on top of inpaint (texture-aware patch match).
- [ ] Own subject segmentation (Portrait blur and Cutout use Vision now).
- [ ] Text: kerning is per pair only (no shaping) — Arabic / Indic scripts
      need a shaper (rustybuzz).

## Smart settings / replace / shapes (5.3)

- [x] `analyze` (noise, blocking, softness, skin, kind -> settings within
      a compute budget), `meta` (EXIF/ICC incl. HEIC), `shape` (16 SDF
      shapes, 8 ring styles), pb-ffi `smart.rs`, CLI analyze/replace/shape.
- [ ] Skin share is a colour-box heuristic: anime skin tones read as skin
      too (only used for photos). A small face detector would be better.
- [ ] Learn the analyzer thresholds from more labelled photos (phone
      shots with real sensor noise; the gallery set is mostly renders/art).
- [ ] Analyzer could also pick the cloud's model per region (faces Clean,
      background Photo) once tiles can mix models.
- [ ] Dotted / dashed rings space marks by angle: on shapes with deep
      notches (flower) the spacing is uneven; arc-length spacing would fix it.
- [ ] HEIC output for replacements (now JPEG q95, PNG when transparent).

## 5.4 — GPU engine, Adaptive texture (2026-11-16)

- [x] `pb-gpu`: implicit-GEMM 3x3 conv generated per tile shape as fully
      unrolled WGSL (named registers — dynamically indexed private arrays
      went to scratch memory and halved the speed), vec4 channel loads,
      fused bias/LReLU/PReLU/scale/2 residuals, upsample folded into the
      conv's index math (RRDB's 2x/4x upsampled maps never exist), SRVGG
      pixel-shuffle tail. Matches pb-nn to ~1e-6 on all real models
      (`cargo test -p pb-gpu`).
- [x] Work split into <= 4 GFLOP dispatch parts and ~12 GFLOP submissions:
      one 3 s submission tripped the amdgpu ring timeout (GPU reset).
- [x] Autotune (`pb_gpu::tune`): times 11 candidate tiles per block width on
      a representative conv, keeps the fastest + GPU and CPU GFLOP/s, cached
      per device (`PB_GPU_TUNE=file`, app: Caches/pixelboost-gpu-tune.json).
      Auto stays on the CPU when the GPU is not 10% faster.
- [x] Failure handling: validation / OOM error scopes, panics caught, device
      loss -> CPU fallback network (rebuilt from the model file), `Gpu::reset`
      when the app returns; pause while backgrounded.
- [x] Engine: `Backend` Auto/Cpu/Gpu (`PB_BACKEND`), `Tuning` (denoise,
      realism), `Engine::backend()`.
- [x] Realism: photo weights interpolated with Clean (Real-ESRGAN was
      fine-tuned from RealESRNet). Mid values drift in colour on flat art
      (bench g3986), so realism < 1 always gets Natural clarity.
- [x] `fusion`: Adaptive = Photo + Clean mixed per pixel by a source texture
      map (fine-structure energy, soft skin mask); seam-free inside the 24 px
      tile margin. `Stack` resolves settings -> networks for app, CLI, worker.
- [x] Analyzer: portraits go Photo + Adaptive (realism 0.8) when 2x compute
      fits the budget, else Clean as before; photos with big flat areas and
      every cloud photo go Adaptive.
- [x] CLI: `bench` prints the backend, `gpu-tune`, `upscale --adaptive
      --realism r`, `eval --variants r0.5,fusion,...`, `PB_GPU_PROFILE=1`
      per-kernel timings.
- Bench (26 gallery crops, `--only photo,clean --variants ...`):
  photo nat+d 22.78 dB / tex 0.666, fusion nat+d 23.31 / 0.638,
  r0.5 nat+d 23.78 / 0.604, clean nat+d 24.07 / 0.555.
- Speed here (Radeon 610M iGPU vs Ryzen 3 7320U, 176 px tile): GPU 286-295
  GFLOP/s, CPU 204-213 (1.4x) and the CPU stays free; iPhone GPUs should
  gain far more (measured on device by the tuner).
- [ ] fp16 storage on the GPU (`shader-f16`): halves the 0.4 GB of tile
      buffers on iPhone and the weight bandwidth.
- [ ] Overlap the CPU post-processing of tile n with the GPU network of tile
      n+1 (GPU idles ~10-20% of a run now), or split tiles across CPU + GPU.
- [ ] Cloud worker on the GPU: needs /dev/dri + Mesa Vulkan in the container
      (worker builds pb-core without `gpu` for now).
- [ ] Fusion map v2: a small learned texture/skin classifier instead of the
      hand-tuned energy + YCbCr mask.
