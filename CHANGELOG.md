# Changelog

## 0.3.2 - 2026-09-02

- Align the renderer loader's API fallback with its BF16 schema default and
  mark task-preset-controlled guider/scheduler inputs as advanced.
- Allocate planner MRoPE positions directly on the input device and reject
  invalid flow sigma schedules before they can produce NaN latents.
- Preserve the released ByteDance planner CFG branch/scale mapping explicitly.

## 0.3.1 - 2026-09-02

- Keep the standalone-weight runtime from 0.3.0 unchanged while removing all
  repository-only download, conversion, quantization, workflow-generation,
  and regression tools from the Comfy Registry installation archive.
- Remove unused sharded-index/manifest conversion modules from the Registry
  archive and stop importing conversion helpers during runtime package init.
- Retain those developer tools in the GitHub source repository so weight
  maintainers can still reproduce and validate the published single files.

## 0.3.0 - 2026-09-01

- Replace the runtime sharded-package/index/manifest contract with four
  Core-compatible standalone safetensors files in ComfyUI's standard
  `text_encoders` and `diffusion_models` directories.
- Embed the exact Qwen model config and tokenizer in the planner file and the
  official SentencePiece model in UMT5; use Core `CLIPLoader(type=wan)` for T5.
- Add a bounded-memory byte-preserving exporter and full standalone contract
  validator, and migrate all six API/frontend workflows to the single-file
  inputs.

## 0.2.3 - 2026-09-01

- Add explicit native, Bernini-loader, and ComfyUI-GGUF renderer overrides to
  the production quality runner so the planner and renderer formats can be
  tested independently.
- Normalize external model names for Windows ComfyUI servers and aggregate
  process-tree RSS/VMS when a virtual-environment launcher owns the server.
- Record the first production-step external-NVFP4 T2V visual acceptance and
  keep optimized CUDA-13 performance as an open gate.
- Add disk-spooled conversion from indexed Wan safetensors to base GGUF,
  atomic Q4_K_S quantization/5D repair, recovery of completed intermediates,
  and a high/low GGUF contract validator.
- Publish the validated 16.31 GiB Q4_K_S renderer pair. Its 640x368, 33-frame
  T2V gate passes with mild facial softness and occasional leg/tail merging;
  Balanced INT8 remains the default quality recommendation.

## 0.2.2 - 2026-09-01

- Remove the obsolete external package mirror from the published node package.
- Publish the pinned Wan 2.1 VAE companion weight alongside the Bernini
  packages on Hugging Face so all required downloads come from one repository.
- Point the VAE downloader and installation guide at the verified companion
  file without changing model or runtime behavior.

## 0.2.1 - 2026-09-01

- Accept the full 64-bit ComfyUI seed range in MaskGIT planning.
- Keep VIT flow schedules in FP32 and scope them to one sampling call.
- Build Qwen additive attention masks directly in the planner compute dtype.
- Remove an unused per-step target allocation.
- Preserve one-character negative prompts.
- Report clear visual-item and manifest-schema errors.
- Clarify that GGUF support is an external renderer bridge, not a full native
  Bernini package.

## 0.2.0 - 2026-09-01

### Added

- True BF16, atomic, resumable component repacking with source revision,
  per-shard hashes, dtype accounting, and full validation.
- Architecture-aware stock-Comfy `int8_tensorwise` + ConvRot conversion for
  renderer, balanced, and full profiles, including per-layer reconstruction
  gates and resumable reports.
- Single-file safetensors support and a lightweight high/low renderer-pair
  validator for native INT8, NVFP4/MXFP8, modern FP8, and legacy scaled-FP8
  metadata.
- Direct Comfy `VIDEO` input, source frame-rate propagation, generation
  preflight checks, repeat-job resource metrics, and 640-pixel-long-edge,
  33-frame video regression workflows.

### Changed

- Renderer guidance defaults to one condition arm at a time to reduce peak
  VRAM, with explicit `1`, `2`, and `all` throughput options.
- Qwen vision results, language branches, and final condition branches are
  staged sequentially so large intermediate tensors do not overlap.
- Quantized Wan, Qwen, and UMT5 modules use Comfy mixed-precision operations
  with BF16 compute by default.
- Video examples use 33 frames at 16 fps (2.0625 seconds), preserve aspect
  ratio, and cap the long edge at 640 for routine testing.

### Fixed

- Corrected the old `Bernini-v2-bf16` memory claim: it contained FP32 storage
  and occupied about 166 GiB. The validated BF16 standalone set is 83.04 GiB.
- Prevented quantized planner modules from being initialized on `meta`, which
  discarded quantization hooks during assign-loading.
- Upgraded legacy `scaled_fp8/.scale_weight` checkpoints through Comfy's own
  compatibility path before module construction.
- Reclaimed host and device memory between consecutive jobs in one ComfyUI
  process; two identical 640x368, 33-frame balanced-INT8 jobs now complete
  consecutively with identical decoded frames.
