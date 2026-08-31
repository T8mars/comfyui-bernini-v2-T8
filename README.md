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
180 GB, and the native repack is roughly 166 GiB, so verify disk, RAM, and VRAM
capacity first.

## Development

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
  --output C:/path/to/ComfyUI/models/bernini_v2/Bernini-v2-bf16

python tools/validate_repack.py `
  --root C:/path/to/ComfyUI/models/bernini_v2/Bernini-v2-bf16
```

The validator checks every index entry, shard key set, dtype/shape-derived byte
count, and manifest component total without loading the full checkpoint into
RAM. Add `--verify-hashes` for a slow full-file SHA-256 pass.

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

The runner never changes the versioned workflow; it applies 256 px, 1+1 planner
steps, and two renderer steps in memory, submits `/prompt`, and waits for the
corresponding history entry.

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
2.0625-second regression clip. Subsequent resource-bounded video quality tests
preserve source aspect ratio and cap the long edge at 640. The versioned
workflows keep the released 81-frame defaults and are not silently changed by
these test overrides.

The files can be submitted directly to ComfyUI's `/prompt` API. ComfyUI also
supports loading an API-format workflow for inspection; replace the placeholder
input filenames before queueing media-conditioned examples.

## Verified environment and memory notes

The six real-weight smoke tests were run on Windows with ComfyUI `0.33.0`
(`76135e557da1ec7dcb270160f01e597565e3e003`), an RTX 5090 Laptop GPU with
24 GB VRAM, 64 GB system RAM, and PyTorch `2.7.0+cu128`. ComfyUI warns that this
PyTorch build falls back to the legacy `ModelPatcher`; current Comfy/PyTorch is
preferred for normal use.

The released checkpoint is unusually large: the repack is about 166 GiB and a
full graph constructs UMT5-XXL, Qwen2.5-VL, the VIT decoder, and two Wan 14B
experts. On the verified 64 GB Windows host, full-resolution video validation
uses `--lowvram --reserve-vram 3 --disable-smart-memory --preview-method none
--cache-none` and restarts between cold cases. The renderer estimates activation
memory at the high/low expert switch and passes that reservation to Comfy model
management; without it, the first 848x480 run exhausted a 24 GB device while
loading the low-noise expert. Reusing the same process after a complete
generation can still exhaust host memory while the next loader graph is being
constructed. This lifecycle behavior remains a Core-readiness issue, not a
model-parity result.

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
