# Real-weight smoke-test record

## Balanced-INT8 repeat lifecycle acceptance

The new low-memory path passed two identical jobs consecutively in one ComfyUI
server process without restarting or reusing cached node outputs:

| Run | Geometry / duration | Planner / renderer steps | Wall time | SHA-256 |
|---|---|---:|---:|---|
| 1 | 640x368, 33 frames, 2.0625 s | 1 MaskGIT + 1 VIT / 2 | 343.68 s | `f8763a4f932640043fdf4db9f16d27dda9d3dd8358fadf8529411f389a822d0d` |
| 2 | 640x368, 33 frames, 2.0625 s | 1 MaskGIT + 1 VIT / 2 | 227.51 s | `4dd9f6d1327a5d419ce0460b147412b66e1b150d4e391064a02e369ef7a1529d` |

The MP4 container hashes differ because the embedded workflow/output prefix
differs. Independent `framemd5` checks are identical for all 33 decoded frames.
Both files report H.264, 640x368, 16 fps, and 2.0625 seconds through `ffprobe`.

This used the 45.63 GiB balanced weights with 1,300 stock-Comfy
`int8_tensorwise` + ConvRot layers, BF16 compute, sequential guidance arms, and
the same running server. Observed peak use was about 22.3 GiB VRAM while at
least 33 GiB of 64 GB host RAM remained free. After run 2, ComfyUI returned to
about 1.47 GiB VRAM, 2.68 GiB physical process memory, and 5.28 GiB process
private commit; host free RAM recovered to about 47 GiB. This closes the
previous repeat-job crash for the supported low-memory package on the tested
compatibility environment.

These 1/1/2-step videos are intentionally blurred execution smokes. They are
not accepted visual-quality evidence; the balanced-INT8 production-step result
is recorded separately in `QUALITY_TESTS.md`.

## Scope

This record proves that every released Bernini v2 task can execute through
native Comfy model loading, planning, guidance, sampling, VAE decode, and Core
save nodes. It is intentionally a low-cost execution smoke, not a visual-quality
or numerical-parity claim.

The original historical smoke used a preserve-dtype package pinned to Hugging
Face revision
`399cf6a18a4c523b367b2b1ac25a2a61009e7df3`. The repack validator checked all
seven component indexes, 62 shards, 3314 runtime tensors, and 166.08 GiB of
indexed tensor data. That directory was later proved to contain FP32 storage,
not BF16, and is superseded by the 83.04 GiB true-BF16 and 45.63 GiB balanced
INT8 packages. Both Wan experts still map exactly to 1095 native Comfy keys
before quantization side tensors are added.

## Environment

- ComfyUI `0.33.0`, commit `76135e557da1ec7dcb270160f01e597565e3e003`
- Windows, Python 3.10.10, PyTorch `2.7.0+cu128`, xFormers 0.0.30
- NVIDIA RTX 5090 Laptop GPU, 24 GB VRAM; 64 GB system RAM
- Server flags: `--lowvram --cache-none --preview-method none`
- Only this custom node package was whitelisted during validation

Every historical six-task case used a clean Comfy process because the FP32
package did not reliably reclaim enough host memory to build a second full
loader graph. The balanced-INT8 lifecycle acceptance above deliberately uses
two fresh loader graphs in one process with `--cache-none`.

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

The historical 256x256/5-frame request setup remains reproducible with explicit
`tools/run_comfy_smoke.py --width 256 --height 256 --length 5` overrides. The
runner now defaults to 640x368/33 frames and can run consecutive balanced-INT8
jobs in the same legacy-PyTorch server. It derives its graph from the versioned
`examples/api/<task>.json` file and does not modify the example on disk.

## Bugs found by real execution

1. NumPy `RandomState.permutation` produces an `int32` array on Windows. PyTorch
   scatter requires `int64`; `maskgit_order` now normalizes the dtype and has a
   regression test.
2. ComfyUI 0.33.0's Qwen vision RoPE leaves Q/K in FP32 while V remains BF16.
   The plugin applies ByteDance's FP32-then-cast-back behavior to its own vision
   tower instance, with dtype and numerical regression tests.
3. Reusing the first completed **FP32-storage** graph to construct a second full
   graph caused a native Windows access violation while allocating T5 layers.
   True BF16 fixed the mislabeled storage baseline; the balanced-INT8 package,
   sequential branches, and low-memory guidance now pass two uncached jobs in
   one process. Current-CUDA and Linux lifecycle lanes remain Core-readiness
   gates.
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
- Repeated prompts in one process on a current PyTorch/CUDA 13 stack; the
  PyTorch 2.7/CUDA 12.8 compatibility lane now passes
- Interrupted download/repack recovery and Linux execution
