# Low-memory weights and runtime

## Supported standalone models

All first-party user-facing weights are ordinary **single-file** safetensors
with stock ComfyUI quantization metadata. No Diffusers directory, shard index,
repack manifest, or custom quantization kernel is required at runtime.

| File | ComfyUI directory |
|---|---|
| `bernini_v2_planner_{profile}.safetensors` | `models/text_encoders` |
| `umt5_xxl_bernini_v2_{profile}.safetensors` | `models/text_encoders` |
| `bernini_v2_high_noise_{profile}.safetensors` | `models/diffusion_models` |
| `bernini_v2_low_noise_{profile}.safetensors` | `models/diffusion_models` |

| Profile | Quantized components | Intended use |
|---|---|---|
| `renderer` | Wan high + Wan low | Conservative compatibility, but often still too large for 64 GB hosts |
| `balanced` | Wan pair + Qwen MLLM + T5 | Recommended memory/quality target |
| `full` | Balanced + connector + VIT decoder | Experimental; small extra size saving and higher quality risk |

The native format is `int8_tensorwise` with ConvRot metadata, an INT8 weight,
FP32 `[out,1]` scale, and dynamic INT8 activations. It is the stock format
documented by [Comfy Quants](https://github.com/Comfy-Org/comfy-quants/blob/main/docs/formats/int8_tensorwise.md),
not the retired `int8_w8a8` custom-node format.

The validated standalone `balanced` files occupy 45.63 GiB versus 83.04 GiB for the
true BF16 package, a 45.1% reduction. It quantizes 1,300 Linear layers (403 in
each Wan expert, 326 in Qwen, and 168 in UMT5). Reconstruction cosine is
0.999954 on average and 0.999938 at worst; relative error is 0.967% on average
and 1.116% at worst. No candidate crossed the quality-fallback threshold.

Conversion defaults are deliberately conservative:

- BF16 is the compute and non-quantized storage dtype.
- Only large two-dimensional Linear weights are considered.
- embeddings, time/conditioning modulation, patch inputs, and output/final
  projections remain BF16.
- ConvRot selects the largest compatible group from 256, 64, and 16.
- every quantized layer is reconstructed and checked; cosine below `0.99` or
  relative error above `2%` triggers a BF16 fallback.
- shards, progress metadata, final indexes, and manifests are written
  atomically and can be resumed only with the exact same source, recipe,
  PyTorch/CUDA, comfy-kitchen version, and target-device fingerprint. This
  prevents one package from silently mixing shards produced by different
  ConvRot rounding implementations.

Run a plan without writing data:

```powershell
python tools/quantize_repack.py `
  --source C:/ComfyUI/models/bernini_v2/Bernini-v2-bf16-native `
  --output C:/ComfyUI/models/bernini_v2/Bernini-v2-balanced-int8 `
  --profile balanced --dry-run
```

Then remove `--dry-run` to convert. The output contains
`quantization-report.json` with per-layer cosine/error/status evidence.

Export the conversion workspace to the four runtime files, then validate the
complete single-file set:

```powershell
python tools/export_single_files.py `
  --source C:/path/to/Bernini-v2-balanced-int8 `
  --output C:/path/to/Bernini-v2-single-int8 `
  --profile int8
python tools/validate_single_files.py `
  --planner C:/path/to/Bernini-v2-single-int8/bernini_v2_planner_int8.safetensors `
  --t5 C:/path/to/Bernini-v2-single-int8/umt5_xxl_bernini_v2_int8.safetensors `
  --high C:/path/to/Bernini-v2-single-int8/bernini_v2_high_noise_int8.safetensors `
  --low C:/path/to/Bernini-v2-single-int8/bernini_v2_low_noise_int8.safetensors
```

This reads safetensors headers and tiny `comfy_quant` markers rather than the
large weights. It requires all 40 Wan blocks, the Bernini v2 renderer boundary
keys, supported stock-Comfy quantization metadata, and an identical normalized
tensor contract for the high/low pair. It accepts both native component keys
and the `model.diffusion_model.` prefix used by stock diffusion-model files.

Stock INT8 requires a current ComfyUI/comfy-kitchen and an NVIDIA GPU with SM
7.5 or later for the optimized path. On RTX 50-series, use an isolated current
PyTorch CUDA 13 environment for performance validation; PyTorch 2.7/CUDA 12.8
loads the format but emits a warning and does not represent the optimized path.

## Runtime memory controls

`Bernini v2 Renderer Guider` exposes:

- `guidance_batch_size=auto`: one guidance arm at a time, lowest peak VRAM.
- `1`, `2`, or `all`: explicit throughput/memory tradeoff; guidance order and
  final formula are unchanged.
- `vae_encode_mode=tiled`: avoids first attempting a large regular VAE encode.

`vae_encode_mode=auto` delegates to Comfy's normal VAE encoder, which estimates
memory, batches where possible, and already retries tiled encoding after a
recognized OOM. There is no separate forced-regular mode because it would only
disable that safety fallback without improving the successful fast path.

The planner moves Qwen vision results to Comfy's intermediate device before
loading the language/auxiliary models and computes its three language branches
sequentially. VIDEO inputs can be connected directly; their frame rate is read
from metadata. IMAGE-batch video input remains supported for older workflows.

Inputs are rejected before sampling when dimensions are not multiples of 16,
video frame counts are not `4n+1`, fps is invalid, the base latent is extreme,
or the requested VIT target exceeds the checkpoint mask-token capacity.

## NVFP4 and GGUF compatibility

Native NVFP4/FP8 Wan safetensors, including the Bernini **v2** files from
[rzgar/Bernini-v2-ComfyUI](https://huggingface.co/rzgar/Bernini-v2-ComfyUI),
already use stock Comfy quantization markers. Place them in
`models/diffusion_models` and select them in the Bernini renderer loaders, or
load them with Core's `Load Diffusion Model` and connect the resulting models
to the guider. Runtime shard indexes are intentionally unsupported. Run
`validate_renderer_pair.py` before loading.

Do not mix Bernini-R weights with Bernini v2, do not pair experts from different
releases, and do not apply an additional FP8 cast to an already quantized file.
The Rzgar NVFP4 files expose `{"format":"nvfp4"}` markers and the stock
`model.diffusion_model.` key prefix. A production-step 640x368, 33-frame T2V
quality gate passes with the Balanced-INT8 planner and both external renderers.
That run uses PyTorch 2.11/CUDA 12.8 eager operations and a competing GPU
process, so it is not a speed or isolated-memory acceptance. NVFP4 remains a
Blackwell/current-CUDA performance lane; treat official BF16 output as the
visual reference and rerun on CUDA 13 before making performance claims.

The quality runner can keep the native planner while replacing only the
renderers:

```powershell
python tools/run_comfy_quality.py t2v `
  --width 640 --height 368 --length 33 `
  --renderer-loader native `
  --high-renderer "Bernini_v2_NVFP4/high.safetensors" `
  --low-renderer "Bernini_v2_NVFP4/low.safetensors"
```

Older `*_fp8_scaled.safetensors` files use the legacy `scaled_fp8` sentinel and
`.scale_weight` tensors instead of per-layer `comfy_quant` markers. The loader
now upgrades that contract through Comfy's own `convert_old_quants` path before
constructing mixed-precision modules, and the pair validator reports it as
`legacy_scaled_fp8` rather than incorrectly calling it unquantized.

GGUF is intentionally optional because ComfyUI does not load it natively. With
[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) installed, load the two
Wan GGUF experts with `Unet Loader (GGUF)` and connect their standard `MODEL`
outputs directly to `Bernini v2 Renderer Guider`. Keep the planner, Qwen, T5,
connector, VIT decoder, mask tokens, and VAE in the native package. This bridge
does not add a GGUF runtime dependency to the node pack and is not part of the
Core PR.

The published Q4_K_S pair occupies 16.31 GiB versus 26.70 GiB for the two
Balanced-INT8 Wan components, a 38.9% renderer-storage reduction. Each file is
8,756,353,664 bytes and contains 1,095 tensors: 356 Q4_K, 44 Q5_K, 6 BF16, and
689 F32. Both contracts contain all 40 Wan blocks and the repaired F32 5-D
`patch_embedding.weight`.

- High SHA-256: `b72df0b32d305b7acade0a7245edde37716f7202f5fbc99fda43aedf5d1ebc87`
- Low SHA-256: `d396d3dcf935deb5bb1d8e6c6735e644adfa23d277efe0c2865bba03d6b7c92b`
- Contract SHA-256: `e779b82a06707b08f85d03e812293e4a14a96eccf00f189f239443e366351b76`

Download directly into the GGUF loader's model directory:

```powershell
hf download t8star/Bernini-V2-Comfy `
  --include "Bernini-v2-GGUF-Q4_K_S/*" `
  --local-dir C:/path/to/ComfyUI/models/diffusion_models
```

The repository also provides a low-RAM conversion path for the indexed BF16
source. It validates every shard header and uses GGUF's disk-backed spool, so
only one source tensor is converted in memory at a time:

```powershell
python tools/convert_sharded_gguf.py `
  --src C:/path/to/Bernini-v2-bf16-native/wan_high `
  --dst C:/path/to/bernini_v2_high_noise-BF16.gguf `
  --fix C:/path/to/bernini_v2_high_noise-5d.safetensors `
  --temp-dir C:/path/to/large-temp-drive
```

Build `llama-quantize` from `llama.cpp` tag `b3962` with
ComfyUI-GGUF revision `6ea2651e7df66d7585f6ffee804b20e92fb38b8a`
`tools/lcpp.patch`, then quantize and repair atomically:

```powershell
python tools/quantize_gguf.py `
  --src C:/path/to/bernini_v2_high_noise-BF16.gguf `
  --dst C:/path/to/bernini_v2_high_noise-Q4_K_S.gguf `
  --fix C:/path/to/bernini_v2_high_noise-5d.safetensors `
  --quantizer C:/path/to/llama-quantize.exe `
  --type Q4_K_S
```

Run the same two commands for `wan_low`, then verify the pair without loading
its weights into Torch:

```powershell
python tools/validate_gguf_pair.py `
  --high C:/path/to/bernini_v2_high_noise-Q4_K_S.gguf `
  --low C:/path/to/bernini_v2_low_noise-Q4_K_S.gguf
```

The 640x368, 33-frame, 16-fps production-step T2V gate passes in 1,319.907
seconds with 33/33 unique decoded frames. Motion and scene composition remain
coherent with no frozen duplication or gross quantization noise. Compared with
Balanced INT8, the face is softer and some frames show a slightly elongated
front paw or mild rear-leg/tail merging. Q4_K_S is therefore accepted as an
experimental low-storage renderer lane, not the default quality profile.

That run used ComfyUI 0.33.0, PyTorch 2.7.0+cu128, `--lowvram`, and
ComfyUI-GGUF's partial-compile compatibility path. Another GPU process was
active, so its 21.048 GiB Comfy-visible peak, 33.801 GiB process RSS, and 44.996
GiB process VMS are not isolated-memory claims. CUDA-13 optimized performance,
repeat-lifecycle, and the other five task gates remain pending. Balanced INT8
remains the recommended quality profile; NVFP4 remains the smallest validated
renderer pair on Blackwell-capable hardware.

## Required regression matrix

- 33 frames at 16 fps (2.0625 seconds) for every video case.
- landscape 640x368 and portrait 368x640; preserve source aspect ratio.
- fixed prompts, seeds, and source media against the BF16 baseline.
- T2V, official V2V, five-reference R2V, and official RV2V visual checks.
- peak allocated/reserved VRAM, `nvidia-smi` peak, process commit/RSS, planner
  time, and per-render-step time.
- two consecutive jobs in one ComfyUI process, both default and `--lowvram`.
- current ComfyUI + CUDA 13 performance lane and the older CUDA 12.8
  compatibility lane kept separate.
