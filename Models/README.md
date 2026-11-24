# Models

`pbnn/` holds the networks the app ships, in PixelBoost's own `.pbnn`
format (JSON header + tensors, see `rust/crates/pb-nn/src/pbm.rs`): v1 =
fp16, v2 adds per-tensor `f32` and `q8` (int8 with one scale per output
channel). The RRDB upscalers ship as q8 (half the size; the eval bench is
within 0.03 dB of fp16), the small SRVGG ones as fp16, the segmentation
nets as f32 (batch-norm-folded weights don't survive fp16).

`downloads/` holds optional models the app fetches on demand from the
PixelBoost server (`GET /models/file/{name}`, Range resume, SHA-256 pinned
in `rust/crates/pb-ffi/src/downloads.rs`): `native-x2.pbnn` (in git) and
`cutout-isnet.pbnn` (44 MB, not in git: regenerate with `pixelboost
convert weights-src/isnet-general-use.pth`, then copy to the server's
`MODEL_FILES_DIR`; update the pinned hash if the file changes).
They run on PixelBoost's own inference engine (`rust/crates/pb-nn`):
hand-written AVX2/NEON 3x3 convolution kernels, no Core ML / ONNX / PyTorch.

| File | Id | Architecture | Best for |
|---|---|---|---|
| `anime-x4.pbnn` | anime | RRDBNet, 6 blocks | Drawn art, anime, cel shading |
| `photo-x4.pbnn` | photo | RRDBNet, 23 blocks | Photos, game screenshots, 3D renders |
| `clean-x4.pbnn` | clean | RRDBNet, 23 blocks (no GAN loss) | Faces, anything where invented texture looks wrong |
| `fast-x4.pbnn` (+ `fast-denoise-x4.pbnn`) | fast | SRVGGNetCompact, 32 convs | Quick everyday upscale; the two are blended for the denoise slider |
| `downloads/native-x2.pbnn` | native2x | RRDBNet, 23 blocks, pixel-unshuffle input | 2x of photos (optional download) |
| `lineart-x4.pbnn` | lineart | SRVGGNetCompact, 16 convs | Anime video frames, line art |
| `cutout-u2netp.pbnn` | cutout | U²-Net-p (1.1 M params, 320 px), Apache-2.0 | Subject cutout (bundled) |
| `downloads/cutout-isnet.pbnn` | cutout HQ | IS-Net general use (44 M params, 1024 px), Apache-2.0 | Finer cutout edges (optional download) |

Reproduce from the official checkpoints (BSD-3-Clause, see
`THIRD_PARTY_NOTICES.md`):

```
cd rust
mkdir -p weights-src && cd weights-src
B=https://github.com/xinntao/Real-ESRGAN/releases/download
for u in v0.1.0/RealESRGAN_x4plus.pth v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth \
         v0.1.1/RealESRNet_x4plus.pth v0.2.5.0/realesr-general-x4v3.pth \
         v0.2.5.0/realesr-general-wdn-x4v3.pth v0.2.1/RealESRGAN_x2plus.pth \
         v0.2.5.0/realesr-animevideov3.pth; do curl -sLO $B/$u; done
cd .. && cargo run --release -p pb-cli -- convert-all weights-src ../Models/pbnn
```

Segmentation checkpoints (Apache-2.0, Xuebin Qin et al.): `u2netp.pth`
from the U-2-Net project (legacy torch format, read by pb-nn's own
loader), `isnet-general-use.pth` from DIS. `pixelboost convert` folds
batch-norm into the convs; `pixelboost cutout in out.png [--hq]` tries them.
The Rust U²-Net matches an independent NumPy reference to 3e-5.
