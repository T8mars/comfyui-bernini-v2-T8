# ComfyUI Core merge plan

This repository is deliberately structured as a proving ground, not as the
final Core diff. The model runs through Comfy model patchers, the native Wan
implementation, standard `CLIP`, `VAE`, `GUIDER`, `SIGMAS`, `SAMPLER`, and
`LATENT` types. No Diffusers pipeline is used at runtime.

## Proposed upstream slices

1. **Model plumbing and fixtures**
   - Add Bernini v2 Diffusers-to-Comfy key detection/mapping fixtures.
   - Reuse the existing Wan `context_latents` and source-id RoPE path introduced
     for Bernini-R; avoid a second Wan implementation.
   - Add tiny/meta state-dict tests for both high- and low-noise experts.

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
   - Document official task presets, model placement, memory expectations, and
     the raw-checkpoint repack step.

## Review gates before opening a PR

- Minimal end-to-end generation and production-setting visual baselines pass on
  the pinned official BF16 checkpoint for all six tasks. Conditional-video
  baselines use an explicit 33-frame (2.0625-second) duration override; the
  latest RV2V case also uses a long-edge-640 resource budget.
- Planner and renderer tensors match the official implementation at agreed
  checkpoints within BF16 tolerances.
- The flow sigma fixture and the dedicated UniPC trajectory match Diffusers;
  1/2/3/8/50-step fixtures differ by at most about `3.6e-7`. The expert switch
  is observed in real sampling.
- Every task has a locally saved smoke-test output and an API workflow that only
  depends on Core plus these nodes.
- Cold-process low-VRAM/offload execution passes on a 24 GB GPU / 64 GB host.
  Full-resolution video needs an activation-memory estimate at expert switch
  plus `--reserve-vram 3 --disable-smart-memory` on the verified legacy
  PyTorch 2.7 stack. Multi-prompt model lifecycle, interrupted download, and
  interrupted repack paths still need dedicated tests.
- Code is rebased onto current ComfyUI and split into reviewable commits.

## Known upstream-facing compatibility items

- ComfyUI 0.33.0's Qwen vision RoPE computes Q/K in FP32 but does not restore
  their original dtype before SDPA. The plugin carries an instance-local bridge
  matching ByteDance's implementation. The Core patch should make that cast in
  the shared Qwen vision implementation and delete the bridge.
- PyTorch 2.7 on the verified Windows host uses Comfy's legacy ModelPatcher.
  Repeat lifecycle and memory tests on the current supported PyTorch/CUDA stack.
- A completed 166 GiB checkpoint graph can leave enough host-side state alive that a
  second fresh loader graph exhausts a 64 GB process. Core submission needs a
  deliberate cache/offload policy and a repeated-prompt test.
- Combined RV2V VAE streams must remain in reference-image-then-source-video
  order. Wan source ids and RoPE depend on stream position; reversing them
  passed execution smoke but failed the official garment-edit semantics.
- The public Core examples should retain the released 81-frame defaults. The
  33-frame quality fixtures are a deliberate two-second test budget and must
  not silently redefine task presets.

No PR, issue comment, or upstream notification should be made until these gates
are complete and the owner explicitly approves submission.
