# Real-weight smoke-test record

## Scope

This record proves that every released Bernini v2 task can execute through
native Comfy model loading, planning, guidance, sampling, VAE decode, and Core
save nodes. It is intentionally a low-cost execution smoke, not a visual-quality
or numerical-parity claim.

The source checkpoint was pinned to Hugging Face revision
`399cf6a18a4c523b367b2b1ac25a2a61009e7df3`. The repack validator checked all
seven component indexes, 62 shards, 3314 runtime tensors, and 166.08 GiB of
indexed tensor data. Both Wan experts map exactly to 1095 native Comfy keys.

## Environment

- ComfyUI `0.33.0`, commit `76135e557da1ec7dcb270160f01e597565e3e003`
- Windows, Python 3.10.10, PyTorch `2.7.0+cu128`, xFormers 0.0.30
- NVIDIA RTX 5090 Laptop GPU, 24 GB VRAM; 64 GB system RAM
- Server flags: `--lowvram --cache-none --preview-method none`
- Only this custom node package was whitelisted during validation

Every case used a clean Comfy process because the verified legacy
`ModelPatcher` environment did not reliably reclaim enough host memory to build
a second full loader graph.

## Settings and results

All outputs used 256×256 geometry, seed 42, one MaskGIT planning step, one VIT
denoising step, and two renderer steps. Video outputs used five frames at 16 fps.

| Task | Source path exercised | Guidance path | Result | Wall time | Output SHA-256 |
|---|---|---|---|---:|---|
| t2i | none | standard APG | PNG success | 10:00 | `c2ba455e3b92473a2f101ef32a2faaf2e7ee66dc1d3e9a594b6d4306d6832f37` |
| i2i | one image | standard APG | PNG success | 8:17 | `402c6761da185b3007a8827a5effb4597b1d32a660097e6ac1b6b72535197639` |
| t2v | none | standard APG | 5-frame H.264 success | 8:39 | `6264469c043ebda58e40e8aee1355049b3db781310c5779bab6db38a168d852c` |
| v2v | one video | standard APG | 5-frame H.264 success | 8:26 | `6b8650dec2c07a6ff99db22fabfe0cd41a290d59247f5d796b2196c9696f8924` |
| r2v | one reference image | standard APG | 5-frame H.264 success | 8:20 | `152cadf1af7fd53a17ec0077cc203572e1ffc7cefb8c7cc284fab6cb6f4a5406` |
| rv2v | video plus reference image | five-arm direct | 5-frame H.264 success | 8:40 | `472313da3f36ffd66b40cd9aaf83da65d53f1b823b723fb33eed88c0efa28fb9` |

The video files were independently read with `ffprobe`; each reports H.264,
256×256, 16/1 fps, duration 0.3125 seconds, and exactly five decoded frames.
Generated media is local evidence and is not committed to the repository.

The request setup is reproducible with `tools/run_comfy_smoke.py`; run one task
per clean Comfy process in this legacy-PyTorch environment. The script derives
its graph from the versioned `examples/api/<task>.json` file and does not modify
the example.

## Bugs found by real execution

1. NumPy `RandomState.permutation` produces an `int32` array on Windows. PyTorch
   scatter requires `int64`; `maskgit_order` now normalizes the dtype and has a
   regression test.
2. ComfyUI 0.33.0's Qwen vision RoPE leaves Q/K in FP32 while V remains BF16.
   The plugin applies ByteDance's FP32-then-cast-back behavior to its own vision
   tower instance, with dtype and numerical regression tests.
3. Reusing the first completed process to construct a second full graph caused
   a native Windows access violation while allocating T5 layers. Clean-process
   runs with `--cache-none` pass; repeated-prompt memory reclamation remains an
   explicit Core-readiness gate.
4. The smoke workflows originally reused Comfy's generic VP UniPC sampler and
   allowed automatic FP16 selection. The production examples now use the
   Diffusers-compatible Bernini flow UniPC sampler and explicit BF16. Existing
   1/1/2-step artifacts remain execution evidence only and are not quality
   baselines.

## Not yet proved by smoke tests

- Visual quality is deliberately outside this 1/1/2-step suite. Separate
  production-setting baselines now cover T2I, I2I, T2V, official-case V2V,
  five-reference R2V, and official-case RV2V; see `QUALITY_TESTS.md`.
- Planner hidden-state, VIT target, Wan prediction, and UniPC step tensors
  against the official Diffusers implementation within agreed tolerances
- Repeated long-video regression beyond the completed 81-frame T2V baseline
- Repeated prompts in one process on the current supported PyTorch/CUDA stack
- Interrupted download/repack recovery and Linux execution
