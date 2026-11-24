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
- [x] Metal: the first CI run on Apple's GPU gave garbage — threads wrote
      different lanes of one workgroup vec4, which Apple compiles to a
      read-modify-write of the whole vector. Shared input tile is now a
      scalar array; Metal matches the CPU to ~3e-6 (CI runner GPU).
- [x] Self-check: a GPU is used only after it reproduces a random RRDB /
      SRVGG on the CPU (`pb_gpu::check`); reason shown in Settings.
- [x] Released as v5.4.0 (2026-11-16): IPA signed (app + 2 extensions with
      the App Group), AltStore source serves it, repo back to private.
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

## 5.5 — Auto quality / clarity / detail, the analyzer really reads the photo (2026-11-16)

- [x] Found: `classify` called every soft or compressed photo "anime" (it
      only counted flat 3x3 neighbourhoods, which blurry photos also have) —
      all 12 test memes, incl. 6 real photos, read as drawn art.
- [x] Real resolution (`classify::effective_scale`): shrink a native crop by
      1.25..4x and scale back; while that loses < 1.2 levels RMS the pixels
      hold nothing finer. Memes blown up 3-4x are found (Bernie 4x, airport
      4x, reaper 3x, Tom & Jerry 3x); sharp art/photos stay 1x (38 images).
- [x] Prescale: the network runs on the real detail (`Settings.prescale`,
      stream shrinks first, output size unchanged). Two model passes only
      when blown up >= 3x or the 4x intermediate stays <= 2.5 MP, else the
      factor is capped at 6x (SpongeBob 347 s -> 68 s).
- [x] New drawn-vs-photo evidence at the real resolution (`classify::ink`):
      continuous thin dark lines (noise-aware) over a concentrated palette,
      or pixel-flat fills (+ grain-free when there are no lines). No photo
      of the 38 labelled images reads as drawn; soft cartoons whose lines
      washed out go to the photo models (fine on cartoons, unlike the
      reverse). Tried and rejected: global flat/texture stats, palette
      alone, model probes scored by PSNR / texture — they overlap.
- [x] `analyze::recommend_fixed`: decide only the fields left on Auto
      around the user's choices; FFI `resolve()`, `upscale()` resolves any
      Auto left, the worker resolves with the server budget.
- [x] App: Quality / Clarity have an Auto chip and Detail an Auto toggle
      (new defaults); the manual view shows what Auto picked and the real
      resolution; learned GFLOP/s accounts for prescale.
- Test (12 user memes, GPU here, 3 min budget, `upscale --smart`): crisp
  lines on the cartoons (#1 #2 #5), sharper faces (#6), rebuilt texture
  (#8), #3 sharp after the cost-based cap; #7 stays soft (Clean on a 4x
  blown-up face; Adaptive on the cloud), #12 still soft (its fine detail is
  sharpening halos, which read as real detail).
- [ ] #12-type images: halos from an earlier upscaler fool the real-
      resolution test; detect ringing separately.
- [ ] A learned content classifier (semantics) would catch soft cartoons.
- [x] Released as v5.5.0 (2026-11-16): signed IPA verified from the release and the AltStore source (identical), repo back to private.

## 5.6 — Pixel Art engine (2026-11-17)

- [x] `pb-core/src/pixelart.rs` (see ../RUST-REMAKE.md 5.6), FFI
      `pb-ffi/src/pixel.rs`, CLI `edit in out pixel:<style>[:json]`.
- [ ] Faces: pale skin next to white areas can share one palette colour
      at low widths (g4170 bench image). A face/skin-aware weight in the
      palette k-means would keep a skin ramp.
- [ ] Sparkles: specular highlights on dark glossy surfaces (leather) can
      read as stars; a shape test on the source peak (round, tiny) would
      separate them. Users can lower the Sparkles slider meanwhile.
- [ ] Outlines from the edge map can follow fabric creases (bed sheet in
      g4170); an orientation-consistent hysteresis would keep only object
      contours.
- [ ] Optional own segmentation (Cutout mask) as the "subject" for rim
      light and outlines, instead of edge contrast alone.
- [ ] Animated pixel art (GIF / video frames with a palette shared across
      frames and temporal dither stability).
- [ ] GPU path not needed (64 MP in ~1 s on CPU); revisit only if previews
      feel slow on older iPhones.

## 5.7 — "Everything pass": 15 upgrades + telemetry (2026-11-23/24)

- [x] T  Telemetry: `pb_nn::telemetry` (events / metrics / tags, run ids,
         thread-scoped run, spans, sampled distributions for live previews,
         memory footprint) re-exported by pb-core (+ image fingerprint).
         Instrumented: upscale runs (net vs post time, tile p50/p90, GPU
         stall, overlap, prescale, dehalo, memory), analyzer (every measure
         + decision + budget + fixed fields), real-resolution loss curves,
         GPU init / self-check / tune / disable / device loss, editor ops
         (full-res events, preview distributions, drafts), pixel art,
         cutout, healing, batch, compare, animation, model downloads,
         worker jobs. pb-ffi `Telemetry` (JSONL queue in Caches, 400-event
         batches, device context, opt-out deletes the queue, feedback:
         kept / discarded / rating / rerun / override / compare_pick /
         mask_fix, opt-in example crops). pb-server: `telemetry_events`,
         `telemetry_samples` (30-day TTL), `/log/telemetry` (POST batch,
         GET export), `/log/telemetry/summary`, `/log/telemetry/feedback`,
         `/log/telemetry/sample(s)`; bridge pulls the worker's events every
         minute (`/telemetry/drain`). CLI `pixelboost telemetry pull|report`,
         `PB_TELEMETRY=file.jsonl` for any command. E2E test
         `pb-ffi/tests/telemetry_e2e.rs` (scratch Postgres + local bridge).
- [x] 1  GPU f16 storage (`shader-f16`): maps, weights, residuals as f16,
         f32 math; self-check retries f32 if f16 fails. Radeon 610M: 311 vs
         267 GFLOP/s best case, max |cpu-gpu| 2e-3 on real models.
- [x] 2  CPU: packed weights stay f16 in memory, expanded per row (vectorised
         bit trick): half the RAM per network, same speed, identical output.
         (An f16-arithmetic NEON kernel was not done: the 6x16 kernel is
         compute-bound, and its speed can't be measured off-device.)
- [x] 3  GPU network of tile n+1 overlaps CPU post-processing of tile n
         (`PB_NO_OVERLAP=1` to compare): -4..10 % time here, identical bytes.
- [x] 4  `strips`: pointwise ops on 256-row strips of the 8-bit picture
         (48 MP adjust: 810 -> 325 MB, 13 -> 2.6 s), in-place / consuming
         application for the rest (clarity 2.2 -> 1.5 GB). Blur-type ops
         still need input + output as f32 (2.9 GB at 48 MP).
- [x] 5  Zoom / spin blur: samples follow streak length + box mips for long
         streaks (no ghosting); live previews render as drafts (smaller, same
         look) when an op misses the 18 ms budget, then once in full.
- [x] 6  `.pbnn` v2 (`q8`, `f32` dtypes), `pixelboost quantize`; bundled
         models 111 -> 47 MB; Native 2x + HQ cutout as downloads
         (ModelStore: resume, SHA-256, atomic install; bridge
         `/models/file/{name}`; analyzer / FFI fall back to Photo when Native
         2x is missing).
- [x] 7  Worker image: `--features gpu`, Mesa RADV (trixie), /dev/dri,
         `PB_GPU_TUNE` cache; gamemode hooks create `/jobs/.gaming` -> GPU
         paused (CPU) while a game runs.
- [x] 8  `pb_nn::seg` (U²-Net / IS-Net: dilated conv via polyphase, ceil
         max-pool, bilinear, BN folding, legacy torch checkpoints) +
         `pb_core::segment` (guided-filter edge refinement), FFI
         `subject_mask` (tap = keep that subject). Replaces Vision.
- [x] 9  `dehalo`: two-sided overshoot score at 1x / 2x / 3x, directional
         plateau clamp at the halo's scale; analyzer judges real resolution
         on an un-sharpened copy when halos are clear (blown-up + sharpened
         bench: eff 1.29 -> 1.94; clean set unchanged). `Settings.dehalo`,
         Auto = -1.
- [x] 10 `heal`: Wexler / PatchMatch completion, coarse to fine, in a window
         around the hole; op `heal`.
- [x] 12 FFI `batch_edit` (strips path per photo).
- [x] 13 FFI `compare_models` (crop through models / variants, fidelity +
         detail scores), `loupe` (nearest-neighbour pixels).
- [x] 14 `animate`: GIF / APNG / WebP decode, `Temporal` stabiliser (keeps
         output where the source didn't change, dedups frames), streamed
         APNG / GIF writer; FFI `upscale_animation`, `VideoUpscaler`
         (`push_bgra` for AVFoundation frames).
- [x] 15 Pixel art: skin-weighted palette k-means, star shape test on the
         source (drops glossy streaks), `extract_palette`, custom palettes
         (`PaletteStore`, settings `custom`); shape rings: dashes / dots
         spaced by arc length.

### Still open after 5.7
- [ ] Verify on device: f16 Metal speed, BGContinuedProcessingTask (GPU
      grant), video pump memory on 4K, cutout speed of IS-Net on A-series.
- [ ] Blur-type ops at 48 MP: tile them with a halo (radius-bounded).
- [ ] Dehalo sharpened-original false positives (4/26 bench images read
      >= 2x); a learned real-resolution estimator would fix both.
- [ ] Telemetry-driven re-fit of analyzer thresholds once data arrives
      (`pixelboost telemetry report`).
- [ ] Pale anime skin still shares palette colours at low widths.
