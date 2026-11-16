# PixelBoost/ (SwiftUI shell) — to do

Overall tracker: ../RUST-REMAKE.md. Swift here is screens only; every pixel
operation is a call into PixelBoostCore (Rust).

- [x] Builds in CI; v5.0.0 released. Needs a test on a real iPhone.
- [x] Sideload widgets/share: App Group resolved at runtime (`Shared/AppGroup.swift`: literal or `.<TeamID>` suffix) and release IPAs ad-hoc signed with `group.com.pixelboost.shared` so SideStore/AltStore/Sideloadly register it (v5.1.1).
- [ ] Verify on device: widget shows last upscale and Share extension hands photos over on a SideStore install.
- [ ] Compare-models sweep screen (server already logs /log/comparison).
- [ ] Pinch-zoom pixel inspector on results.
- [ ] Background-task continuation for long upscales.

## 5.2 photo editor (2026-11-12)

- [x] Edit tools regrouped: Presets & Looks, Light & Colour, Local, Retouch,
      Geometry, Effects, Create (27 tools).
- [x] New panels (`Views/Editor/ToolPanels.swift`): Presets (thumbnails,
      amount, save/delete), Curves (RGB), Colour Mix (HSL), Grading wheels,
      Blur (7 modes), Effects, Text, Draw, Layer, Frame.
- [x] Adjust tabs (Light / Colour / Effects / Curve) + white-balance
      eyedropper; Brush / Linear / Radial local adjustments with handles;
      Skin smoothing; Perspective with grid; more crop ratios.
- [x] Canvas: histogram, press-and-hold before/after, pinch-zoom + double-tap
      reset (no tool open).
- [x] AppModel timeline: History sheet (jump to any step), copy / paste
      edits, Revert to Original; Export sheet (JPEG/PNG/WebP/TIFF, size).
- [ ] First CI build of 5.2 — fix any Swift compile errors it reports.
- [ ] On device: check gesture conflicts (pinch-zoom vs tool overlays,
      compare button), preview speed of zoom/spin blur on an iPhone.
- [ ] Text: on-canvas pinch to resize / rotate (now sliders).
- [ ] Layer: pick a cutout subject as a sticker directly.
- [ ] Mask refinement for Linear/Radial (intersect with luminance / colour
      range — the Rust masks already support it, needs UI).
- [ ] Batch: apply a preset to many photos (Rust side is just `edit_file`
      per photo with the preset JSON).

## 5.3 (2026-11-12)

- [x] Smart settings: `UpscaleChoice.smart` / `maxWait`, `SmartCache`
      (one Rust analysis per photo + budget), Smart card in UpscaleSheet,
      Studio / Batch / Cloud run with the per-photo settings.
- [x] Replace in Photos: `Platform/PhotoLibrary.swift` (edit in place or
      swap), `AppModel.sourceAsset` from picker imports, Settings ›
      "Replacing originals", Cloud jobs remember their asset
      (`cloud.assets`), Batch output picker.
- [x] Shape crops + rings: `Views/Editor/ShapeCrop.swift` (RingSpec,
      ShapeKit outlines from Rust, ShapeOverlay, ShapeCropControls).
- [ ] Try on a device: edit-in-place of a Live Photo and of an iCloud-only
      original (falls back to swap if Photos refuses the edit).
- [ ] Limited photo access: replacing needs the original in the allowed
      set; offer the limited-library picker when `notFound`.
- [ ] Export sheet could offer "Replace original" for edited photos too.
- [ ] Shape crop: drag on the canvas to rotate the shape (slider only now).

## 5.4 — GPU engine, Adaptive texture (2026-11-16)

- [x] Settings › Compute: Auto / GPU / CPU (`pb.compute`), GPU name and the
      measured GPU vs CPU speed (`computeInfo()`, first visit tunes once).
- [x] `Core.engine` sets the tuning cache (Caches/) and the compute mode at
      launch; `PixelBoostApp` pauses GPU work outside `.active` (iOS forbids
      background GPU work — running jobs continue on the CPU).
- [x] Upscale sheet › Texture card (Photo model): Adaptive toggle + Realism
      slider; Smart line shows them; estimates double for Adaptive; learned
      GFLOP/s accounts for Adaptive runs (`UpscaleOutcome.adaptive`).
- [x] Cloud jobs send `realism` / `adaptive`.
- [ ] On device: compare GPU vs CPU times on the History entries, watch for
      thermal throttling on long GPU runs, check memory on 6 GB iPhones
      (photo model, 128 px tiles ≈ 0.4 GB of GPU buffers).
- [ ] Show which backend ran in the result card (`UpscaleOutcome.backend`).
