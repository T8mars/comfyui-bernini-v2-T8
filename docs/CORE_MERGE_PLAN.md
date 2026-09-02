# ComfyUI Core merge plan

The standalone-model Core implementation is under review in
[ComfyUI PR #16019](https://github.com/Comfy-Org/ComfyUI/pull/16019). It
supersedes and closes the earlier sharded-package proposal in
[#16001](https://github.com/Comfy-Org/ComfyUI/pull/16001). The sections below
record its implementation shape, validation evidence, and known follow-up work.

This repository is deliberately structured as a proving ground, not as the
final Core diff. The model runs through Comfy model patchers, the native Wan
implementation, standard `CLIP`, `VAE`, `GUIDER`, `SIGMAS`, `SAMPLER`, and
`LATENT` types. No Diffusers pipeline is used at runtime.

## Proposed upstream slices

1. **Single-file model plumbing**
   - Load one self-contained Planner safetensors from `models/text_encoders`.
   - Load each native Wan expert from one safetensors file in
     `models/diffusion_models`; reuse the existing Wan implementation.
   - Reuse Core `CLIPLoader(type=wan)` for the standalone UMT5 file and the
     existing VAE loader for Wan 2.1 VAE.
   - Do not add a Diffusers directory, runtime manifest, shard index, or custom
     model folder to Core.

2. **Planner modules**
   - Add the Qwen2.5-VL additive-mask inference path, connector, mask tokens,
     and VIT flow decoder under existing Comfy model management.
   - Keep language, vision, and auxiliary planner patchers independently
     offloadable.
   - Test the three official planner branches: full condition, text/media-free
     condition, and text-only image condition.

3. **Conditioning and guidance**
   - Add a small plan object and source-media VAE conditioning node.
   - Add the four-arm `vae_txt_vit_wapg` and five-arm `rv2v_wapg` guider.
   - Switch high/low Wan experts inside one guider so UniPC history remains
     continuous across the 0.875 boundary.
   - Add the flow-prediction UniPC order-2 BH2 sampler. Core's existing generic
     UniPC follows a VP schedule and cannot be reused for Bernini.

4. **Examples and documentation**
   - Supply workflows for `t2i`, `i2i`, `t2v`, `v2v`, `r2v`, and `rv2v`.
   - Document official task presets, standard model placement, and memory
     expectations.

## Submission status and review gates

- Minimal end-to-end generation and production-setting visual baselines pass on
  the pinned official BF16 checkpoint for all six tasks. Conditional-video
  baselines use an explicit 33-frame (2.0625-second) duration override; the
  latest RV2V case also uses a long-edge-640 resource budget.
- Renderer scheduling, flow-UniPC, Qwen hidden-state, VIT target, and both Wan
  expert prediction tensors match the official implementation within the
  recorded tolerances in `TENSOR_PARITY.md`.
- The flow sigma fixture and the dedicated UniPC trajectory match Diffusers;
  1/2/3/8/50-step fixtures differ by at most about `3.6e-7`. The expert switch
  is observed in real sampling.
- Every task has a locally saved smoke-test output and an API workflow that only
  depends on Core plus these nodes.
- Cold-process low-VRAM/offload execution passes on a 24 GB GPU / 64 GB host.
  Full-resolution video needs an activation-memory estimate at expert switch
  plus `--reserve-vram 3 --disable-smart-memory` on the verified legacy
  PyTorch 2.7 stack. The balanced-INT8 path now passes two uncached
  640x368/33-frame jobs in one process and returns host and device memory to
  baseline. Repeat BF16, current-CUDA/Linux lifecycle, interrupted download,
  and interrupted repack paths still need dedicated tests.
- Code is based on current ComfyUI master and submitted as one self-contained
  review commit. The focused Core suite passes 18 tests and Ruff; real INT8
  Planner, Core Wan UMT5, and both 14.288B Wan standalone files load without
  missing or meta parameters. PR CI remains authoritative.

## Known upstream-facing compatibility items

- ComfyUI's shared Qwen vision RoPE previously computed Q/K in FP32 without
  restoring their original dtype before SDPA. PR #16019 restores the input
  dtypes in the shared implementation, so the Core port needs no private bridge.
- PyTorch 2.7 on the verified Windows host uses Comfy's legacy ModelPatcher.
  Repeat lifecycle and memory tests on the current supported PyTorch/CUDA stack.
  The compatibility lane now has balanced-INT8 repeat evidence plus a
  production-step T2V quality pass, but that does not replace the upstream lane.
- The original `Bernini-v2-bf16` directory was actually FP32 and occupied about
  166 GiB. It is not a valid BF16 memory baseline. The corrected BF16 standalone
  set is 83.04 GiB, and the architecture-aware balanced INT8 ConvRot set is
  45.63 GiB. Core submission still needs repeat-prompt lifecycle evidence for
  both packages on current ComfyUI.
- Combined RV2V VAE streams must remain in reference-image-then-source-video
  order. Wan source ids and RoPE depend on stream position; reversing them
  passed execution smoke but failed the official garment-edit semantics.
- Public Core examples should retain the released 81-frame presets.
  This custom-node package deliberately defaults video examples and regression
  runners to 33 frames (2.0625 seconds) with a 640-pixel long edge so 24 GB / 64
  GB development hosts can exercise the complete path. The distinction must be
  called out when preparing an upstream diff.

PR #16019 is the authoritative upstream review. Further changes should respond
to review or CI evidence there instead of reopening a parallel implementation.
