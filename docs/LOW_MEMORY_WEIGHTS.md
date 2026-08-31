# Low-memory weights and runtime

## Supported native packages

All first-party packages remain ordinary safetensors with stock ComfyUI
quantization metadata. No quantization kernel is vendored by this node pack.

| Profile | Quantized components | Intended use |
|---|---|---|
| `renderer` | Wan high + Wan low | Conservative compatibility, but often still too large for 64 GB hosts |
| `balanced` | Wan pair + Qwen MLLM + T5 | Recommended memory/quality target |
| `full` | Balanced + connector + VIT decoder | Experimental; small extra size saving and higher quality risk |

The native format is `int8_tensorwise` with ConvRot metadata, an INT8 weight,
FP32 `[out,1]` scale, and dynamic INT8 activations. It is the stock format
documented by [Comfy Quants](https://github.com/Comfy-Org/comfy-quants/blob/main/docs/formats/int8_tensorwise.md),
not the retired `int8_w8a8` custom-node format.

The validated `balanced` package occupies 45.62 GiB versus 83.03 GiB for the
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

Validate the two renderer experts independently of the package manifest:

```powershell
python tools/validate_renderer_pair.py `
  --high C:/ComfyUI/models/bernini_v2/Bernini-v2-balanced-int8/wan_high/model.safetensors.index.json `
  --low C:/ComfyUI/models/bernini_v2/Bernini-v2-balanced-int8/wan_low/model.safetensors.index.json
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
already use stock Comfy quantization markers. They can be loaded with Core's
`Load Diffusion Model` from `models/diffusion_models`, then connected directly
to the two `MODEL` inputs of `Bernini v2 Renderer Guider`. Alternatively, place
one file in a path containing `wan_high` or `wan_low` under
`models/bernini_v2`; this package's renderer loader accepts single safetensors
as well as shard indexes. Run `validate_renderer_pair.py` before loading.

Do not mix Bernini-R weights with Bernini v2, do not pair experts from different
releases, and do not apply an additional FP8 cast to an already quantized file.
The Rzgar NVFP4 files expose `{"format":"nvfp4"}` markers and the stock
`model.diffusion_model.` key prefix. NVFP4 is a Blackwell/current-CUDA
performance lane; the verified PyTorch 2.7/CUDA 12.8 compatibility environment
is not used to make an NVFP4 speed or quality claim. Treat official BF16 output
as the visual reference when evaluating external files.

Older `*_fp8_scaled.safetensors` files use the legacy `scaled_fp8` sentinel and
`.scale_weight` tensors instead of per-layer `comfy_quant` markers. The loader
now upgrades that contract through Comfy's own `convert_old_quants` path before
constructing mixed-precision modules, and the pair validator reports it as
`legacy_scaled_fp8` rather than incorrectly calling it unquantized.

GGUF is intentionally optional because ComfyUI does not load it natively. With
[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) installed, load the two
Wan GGUF experts with its Wan-compatible model loaders and connect their
standard `MODEL` outputs directly to `Bernini v2 Renderer Guider`. Keep the
Bernini planner auxiliary components in the native package. This bridge does
not add a GGUF dependency to the node pack and is not part of the future Core
submission path.

The node pack deliberately does not wrap or vendor ComfyUI-GGUF's converter.
Wan GGUF conversion has separate 5-D tensor repair and llama.cpp quantization
steps, and its loader is the authority for that storage format. The integration
boundary here is the standard Comfy `MODEL` type, so GGUF updates do not require
changes to Bernini planning, guidance, sampling, or the future Core patch.

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
