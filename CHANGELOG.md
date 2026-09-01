# Changelog

## Unreleased

- Add explicit native, Bernini-loader, and ComfyUI-GGUF renderer overrides to
  the production quality runner so the planner and renderer formats can be
  tested independently.
- Normalize external model names for Windows ComfyUI servers and aggregate
  process-tree RSS/VMS when a virtual-environment launcher owns the server.
- Record the first production-step external-NVFP4 T2V visual acceptance and
  keep optimized CUDA-13 performance and GGUF acceptance as open gates.

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
  and occupied about 166 GiB. The validated BF16 package is 83.03 GiB.
- Prevented quantized planner modules from being initialized on `meta`, which
  discarded quantization hooks during assign-loading.
- Upgraded legacy `scaled_fp8/.scale_weight` checkpoints through Comfy's own
  compatibility path before module construction.
- Reclaimed host and device memory between consecutive jobs in one ComfyUI
  process; two identical 640x368, 33-frame balanced-INT8 jobs now complete
  consecutively with identical decoded frames.
