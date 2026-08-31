# ComfyUI Bernini v2

Native ComfyUI implementation of
[ByteDance/Bernini-Diffusers-v2](https://huggingface.co/ByteDance/Bernini-Diffusers-v2).

The official ByteDance implementation is the numerical oracle. It is not a
runtime dependency of this project. The runtime target is ComfyUI's own model
loading, device management, conditioning, guider, scheduler and sampling APIs.

## Status

- Baseline ComfyUI commit: `76135e557da1ec7dcb270160f01e597565e3e003`
- Baseline upstream Bernini source is cloned locally under `.upstream/` and is
  intentionally not vendored.
- Official model metadata is downloaded to `models/` and the full snapshot is
  resumable with `tools/download_model.py`.
- Streamed component repacking, native planner loaders, six-task planning,
  source-media conditioning, APG, continuous dual-expert switching, and the
  exact Diffusers-compatible flow-prediction UniPC BH2 solver are implemented.
- The pinned 180 GB checkpoint was repacked and independently validated. All
  six tasks pass cold-process, real-weight end-to-end smoke tests through the
  Comfy `/prompt` API. Production-setting visual baselines pass for T2I, I2I,
  T2V, official-case V2V, five-reference R2V, and official-case RV2V.
  The balanced INT8 package also passes a production-step 640x368, 33-frame
  T2V visual gate and two consecutive uncached jobs in one server process.
  Intermediate-tensor parity gates remain before proposing a Core PR.

## Intended runtime components

- Standard ComfyUI Wan2.2 high-noise and low-noise `MODEL` objects.
- Standard Wan UMT5 `CLIP` and Wan VAE objects.
- A device-managed `BERNINI_V2_PLANNER` containing Qwen2.5-VL, the connector,
  mask tokens and the VIT diffusion decoder.
- A `BERNINI_V2_PLAN` object and native guider implementing Bernini's renderer
  branches and APG in flow-prediction space.
- A Bernini-specific order-2 UniPC BH2 sampler using flow alpha/sigma updates.

The implementation deliberately does not wrap `diffusers.DiffusionPipeline`.

## Installation

Install `Bernini v2 (Native)` from ComfyUI-Manager, or install the repository
manually:

```powershell
cd C:/path/to/ComfyUI/custom_nodes
git clone https://github.com/T8mars/comfyui-bernini-v2-T8.git
```

Restart ComfyUI after installation. The Registry/Manager package ID is
`bernini-v2-t8`.

The model is not bundled with this node pack. Follow [Model preparation](#model-preparation)
before loading an example workflow. The pinned upstream checkpoint is roughly
180 GB. The corrected native BF16 repack is 83.03 GiB; the recommended
balanced INT8 ConvRot package is 45.62 GiB (45.1% smaller).

## Development

The following commands require a GitHub checkout. Development-only workflow
builders and regression runners are intentionally excluded from the Registry
archive so ComfyUI-Manager installs contain only runtime and model-preparation
utilities.

```powershell
python tools/download_model.py --metadata-only
python tools/repack_diffusers.py --source models/ByteDance/Bernini-Diffusers-v2 --dry-run
python -m pytest
```

Model data and generated repacks are ignored by Git.

## Model preparation

The official release is one roughly 180 GB sharded checkpoint. Downloading is
resumable and pinned to an immutable revision:

```powershell
python tools/download_model.py `
  --output models/ByteDance/Bernini-Diffusers-v2 `
  --revision 399cf6a18a4c523b367b2b1ac25a2a61009e7df3
```

Repack it into independently loadable native components under ComfyUI:

```powershell
python tools/repack_diffusers.py `
  --source models/ByteDance/Bernini-Diffusers-v2 `
  --output C:/path/to/ComfyUI/models/bernini_v2/Bernini-v2-bf16-native `
  --storage-dtype bfloat16 `
  --source-revision 399cf6a18a4c523b367b2b1ac25a2a61009e7df3

python tools/validate_repack.py `
  --root C:/path/to/ComfyUI/models/bernini_v2/Bernini-v2-bf16-native `
  --verify-hashes
```

The validator checks every index entry, shard key set, dtype/shape-derived byte
count, actual storage dtype, quantization marker/scale contract, manifest totals,
and optional SHA-256 hashes without loading the full checkpoint into RAM. Writes
are atomic and resumable; rerunning the same command resumes verified shards.

Create the recommended stock-Comfy INT8 ConvRot package from the BF16 repack:

```powershell
python tools/quantize_repack.py `
  --source C:/path/to/ComfyUI/models/bernini_v2/Bernini-v2-bf16-native `
  --output C:/path/to/ComfyUI/models/bernini_v2/Bernini-v2-balanced-int8 `
  --profile balanced --device cuda

python tools/validate_repack.py `
  --root C:/path/to/ComfyUI/models/bernini_v2/Bernini-v2-balanced-int8 `
  --verify-hashes

python tools/validate_renderer_pair.py `
  --high C:/path/to/ComfyUI/models/bernini_v2/Bernini-v2-balanced-int8/wan_high/model.safetensors.index.json `
  --low C:/path/to/ComfyUI/models/bernini_v2/Bernini-v2-balanced-int8/wan_low/model.safetensors.index.json
```

`balanced` quantizes the two Wan experts, Qwen planner, and T5 while retaining
the quality-sensitive planner auxiliary path in BF16. The validated package is
45.62 GiB, 45.1% smaller than the 83.03 GiB BF16 source. Every candidate layer
is reconstructed during conversion; layers below cosine `0.99` or above `2%`
relative error automatically remain BF16. The pair validator checks the 40
renderer blocks, quantization markers, and high/low structural match without
materializing the large tensors. See
[`docs/LOW_MEMORY_WEIGHTS.md`](docs/LOW_MEMORY_WEIGHTS.md) for profiles, CUDA
requirements, NVFP4, and optional GGUF wiring.

Bernini uses the standard Wan 2.1 VAE. Download it directly into ComfyUI's VAE
folder:

```powershell
python tools/download_vae.py --output C:/path/to/ComfyUI/models/vae
```

## Native workflow

The runtime graph is intentionally composed from standard Comfy types:

1. Load the repack with `Load Bernini v2 Planner`, two
   `Load Bernini v2 Wan Renderer` nodes, and `Load Bernini v2 T5`.
2. Encode positive and negative text with ordinary `CLIP Text Encode` nodes.
3. Run `Bernini v2 Plan` with one of `t2i`, `i2i`, `t2v`, `v2v`, `r2v`, or
   `rv2v` and the task's required source inputs.
4. Connect the plan, both Wan models, and the Wan VAE to
   `Bernini v2 Renderer Guider`.
5. Use `Bernini v2 UniPC Sigmas`, `RandomNoise`, `Bernini v2 Flow UniPC
   (BH2)`, and `SamplerCustomAdvanced`. Comfy's generic `uni_pc_bh2` assumes a
   VP schedule and must not be substituted.
6. Decode with the standard VAE decoder and save as image or video.

`use_task_defaults` reproduces the published per-task planning, step, and
guidance values. Disable it only when intentionally tuning the advanced fields.

Six generated API-format examples live in [`examples/api`](examples/api), and
six editable ComfyUI canvas workflows live in
[`examples/workflows`](examples/workflows). They cover every released task and
use only ComfyUI Core nodes plus this package:

- `t2i.json` and `t2v.json` need no source media.
- `i2i.json` and `r2v.json` expect `bernini_reference.png` in ComfyUI's input
  directory.
- `v2v.json` expects `bernini_source.mp4`; `rv2v.json` expects both placeholders.

Regenerate and structurally validate them after changing a node signature:

```powershell
python tools/build_example_workflows.py
python tools/build_frontend_workflows.py
python -m pytest tests/test_example_workflows.py
```

Queue one reduced real-weight smoke against a running Comfy server:

```powershell
python tools/run_comfy_smoke.py t2i
python tools/run_comfy_smoke.py rv2v `
  --reference-image bernini_reference.png `
  --source-video bernini_source.mp4
```

The runner never changes the versioned workflow. It uses the 640x368,
33-frame (2.0625-second) video budget by default, applies one MaskGIT step, one
VIT denoising step, and two renderer steps in memory, submits `/prompt`, and
waits for the corresponding history entry. Image tasks use one frame.

Queue a production-setting quality case with an explicit prompt and, for R2V
or RV2V, repeat `--reference-image` for every reference:

```powershell
python tools/run_comfy_quality.py r2v `
  --reference-image reference-0.png `
  --reference-image reference-1.png `
  --width 640 --height 368 --length 33 `
  --prompt "Describe the intended subject, references, scene, and motion."
```

At 16 fps, Bernini-compatible frame counts follow `4n+1`; 33 frames produce a
2.0625-second regression clip. Versioned video workflows now use 640x368 and 33
frames. Portrait/source-matched cases preserve aspect ratio and cap the long
edge at 640 rather than forcing square output.

The files can be submitted directly to ComfyUI's `/prompt` API. ComfyUI also
supports loading an API-format workflow for inspection; replace the placeholder
input filenames before queueing media-conditioned examples.

## Verified environment and memory notes

The six real-weight smoke tests were run on Windows with ComfyUI `0.33.0`
(`76135e557da1ec7dcb270160f01e597565e3e003`), an RTX 5090 Laptop GPU with
24 GB VRAM, 64 GB system RAM, and PyTorch `2.7.0+cu128`. ComfyUI warns that this
PyTorch build falls back to the legacy `ModelPatcher`; current Comfy/PyTorch is
preferred for normal use.

The original package called `Bernini-v2-bf16` was discovered to contain FP32
storage and occupied about 166 GiB. Do not use it as the memory baseline. The
repacker now performs and validates a real BF16 conversion. For 24 GB devices,
leave `guidance_batch_size=auto` so renderer arms execute sequentially, and use
the explicit tiled VAE mode if regular encode repeatedly approaches the limit.
The high/low expert switch still reserves activation memory through Comfy model
management. Two consecutive balanced-INT8 jobs now pass in the same server
process with memory returning to baseline. The production 640x368/33-frame T2V
run peaked at 16.511 GiB Comfy-visible VRAM and 29.885 GiB process RSS; details
are tracked in the smoke, quality, and low-memory records.

Exact smoke settings, output hashes, and remaining limitations are recorded in
[`docs/SMOKE_TESTS.md`](docs/SMOKE_TESTS.md).

Production-setting results, hashes, visual assessments, and the defects found
in the reduced T2I, weak synthetic V2V, and combined RV2V cases are recorded in
[`docs/QUALITY_TESTS.md`](docs/QUALITY_TESTS.md).

## Core preparation

The proposed upstream slicing and remaining validation gates are tracked in
[`docs/CORE_MERGE_PLAN.md`](docs/CORE_MERGE_PLAN.md) and
[`docs/PARITY_MATRIX.md`](docs/PARITY_MATRIX.md). No upstream PR is opened by
this worktree.
