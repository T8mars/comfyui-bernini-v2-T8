# Production-setting quality-test record

## Accepted results

The native pipeline has usable, real-weight visual baselines for all six
released tasks. These were queued through ComfyUI's `/prompt` API and used the
released task presets for planning, renderer steps, guidance, and flow shift.
Conditional-video cases use 33 frames at 16 fps (2.0625 seconds) as an explicit
duration override. V2V and R2V use 848x480; the final RV2V rerun preserves its
portrait aspect at 368x640 to follow the later long-edge-640 test budget. T2V
was completed earlier at the released 848x480, 81-frame duration. A separate
balanced-INT8 T2V acceptance uses the current 640x368, 33-frame test budget.

| Task | Geometry / duration | Wall time | Visual acceptance | SHA-256 |
|---|---|---:|---|---|
| T2I | 512x512 PNG | 588.34 s | sharp, prompt-consistent fox in snow | `b246ea6a6249922d91a96e69a22db662a7f098b7175968adb58f746f6f1a5479` |
| I2I | 512x512 PNG | 659.97 s | reference identity/composition preserved with requested change | `997d2359db10a8776e3b86d926b88c9e6c5c52ed8255872df22226555b29d0ce` |
| T2V | 848x480, 81 frames, 5.0625 s | 4160.44 s | coherent fox walk with continuous motion | `b9b1446ac9c50516397646579b60ff696df7ecb178818945a084eadbf66c9571` |
| T2V balanced INT8 | 640x368, 33 frames, 2.0625 s | 1651.98 s | sharp, coherent fox walk; no visible quantization collapse | `061c3380c7ace7f5d7c74f51fdbef45cfe89084c1523ae7baeb0dd131132442f` |
| T2V standalone INT8 format gate | 640x368, 33 frames, 2.0625 s | 342.31 s | 5/2/10-step gate; recognizable coherent fox motion, 33 unique frames; not a production-quality preset | `3d1d1a96343f46314f85d03ee03b2f2ffcbac9cd2d05fb09a44664d28603c6e4` |
| T2V external NVFP4 renderers | 640x368, 33 frames, 2.0625 s | 1518.03 s | coherent fox walk; no visible quantization collapse; CUDA 12.8 eager quality gate only | `4fb35e38321f19ad069ac7f7c6e0b5a414adcb466a0158398cfcbb3d5068786e` |
| T2V GGUF Q4_K_S renderers | 640x368, 33 frames, 2.0625 s | 1319.91 s | coherent motion and 33 unique frames; mild facial softness and leg/tail merging; experimental | `8348dcc3baa37c782bd88867cf93dbd35622d9aef4b65454187d70724725f838` |
| T2V current Core candidate | 640x368, 33 frames, 2.063 s | 1756.39 s | official 50/1/50 preset; sharp coherent fox walk, 33 unique frames | `df443d0e577af8e8ef36bf81b2d694cd683b7058ec44350d2dbf5c55d45c80d2` |
| V2V | 848x480, 33 frames, 2.0625 s | 3769.25 s | official dog scene and motion preserved; stable snowman added | `853d5ad0cd8d2043184655bde551f6b6320a3c8b3b715c71e973c9a3829895f6` |
| R2V | 848x480, 33 frames, 2.0625 s | 4080.58 s | all five official references integrated into one stable scene | `c98aafe34bf3ae78e71db9d84cdca2392af623cade9bbda7f4374dc51434a462` |
| RV2V | 368x640, 33 frames, 2.0625 s | 7133.09 s | reference shirt applied; inner shirt, identity, scene, and motion preserved | `201fdfeb018b709374cad7aa2f601072036dc0721a84e10a8b5cb790437b2557` |

All accepted videos decode at 16 fps and have a unique MD5 for every frame.
Quality artifacts and contact sheets remain local under `artifacts/quality` and
`artifacts/qa`; they are test evidence, not files intended for a Core patch.

Accepted video deliverables:

- `artifacts/quality/bernini_v2_t2v_full_848x480_81f.mp4`
- `artifacts/quality/bernini_v2_t2v_balanced_int8_long640_33f.mp4`
- `artifacts/quality/bernini_v2_t2v_nvfp4_long640_33f.mp4`
- `artifacts/quality/bernini_v2_t2v_gguf_q4ks_long640_33f.mp4`
- `artifacts/quality/bernini_v2_v2v_official_case1_848x480_33f.mp4`
- `artifacts/quality/bernini_v2_r2v_official_5ref_848x480_33f.mp4`
- `artifacts/quality/bernini_v2_rv2v_official_case1_long640_33f.mp4`

The current-Core candidate result and its inspection images are published with
the model package:

- <https://huggingface.co/t8star/Bernini-V2-Comfy/blob/main/samples/core-pr-t2v-640x368-33f.mp4>
- <https://huggingface.co/t8star/Bernini-V2-Comfy/blob/main/samples/core-pr-t2v-contact.png>
- <https://huggingface.co/t8star/Bernini-V2-Comfy/blob/main/samples/core-pr-t2v-frame16.png>

The external-NVFP4 result and inspection images are also published:

- <https://huggingface.co/t8star/Bernini-V2-Comfy/blob/main/samples/nvfp4-t2v-640x368-33f.mp4>
- <https://huggingface.co/t8star/Bernini-V2-Comfy/blob/main/samples/nvfp4-t2v-9frame-grid.png>
- <https://huggingface.co/t8star/Bernini-V2-Comfy/blob/main/samples/nvfp4-t2v-frame16.png>

The GGUF Q4_K_S result and inspection images are published alongside the
renderer pair:

- <https://huggingface.co/t8star/Bernini-V2-Comfy/blob/main/samples/gguf-q4ks-t2v-640x368-33f.mp4>
- <https://huggingface.co/t8star/Bernini-V2-Comfy/blob/main/samples/gguf-q4ks-t2v-9frame-grid.png>
- <https://huggingface.co/t8star/Bernini-V2-Comfy/blob/main/samples/gguf-q4ks-t2v-frame16.png>

## T2I baseline details

The T2I result is a recognizable, sharp red fox in snow at sunrise with
coherent anatomy, detailed fur, photographic depth of field, and
prompt-consistent lighting. It is a usable generation rather than the blurred
color mass produced by the original reduced smoke. Minor synthetic texture
remains around the rear/tail, so this is a baseline acceptance result rather
than a claim of pixel parity with the official pipeline.

- Output: `Bernini-v2/quality/t2i_full_bf16_flow_unipc_00001_.png`
- Prompt: `A cinematic photograph of a red fox standing in fresh snow at sunrise.`
- Seed: 42
- Planner: 25 MaskGIT steps, 5 VIT denoising steps, text CFG 1.2,
  image CFG 1.0
- Renderer: 50 steps, flow shift 5, UniPC order 2 / BH2, expert boundary
  0.875
- Guidance: image 1.0, text 4.0, VIT target 0.5, post-switch scale 1.0
- Precision: Qwen/planner, UMT5, both Wan experts, and VAE ran in BF16
- Text conditioning: official T2I system prefix and standard Wan negative

## Balanced-INT8 T2V acceptance

The recommended 45.63 GiB standalone set quantizes both Wan experts, Qwen, and UMT5
with 1,300 stock-Comfy `int8_tensorwise` + ConvRot layers while retaining the
connector, MaskGIT tokens, and VIT decoder in BF16. Per-layer reconstruction
checks produced mean/min cosine 0.999954/0.999938 and mean/max relative error
0.967%/1.116%, with zero quality fallbacks.

The production run uses the versioned fox prompt and seed 42, 25 MaskGIT steps,
5 VIT denoising steps, the T2V task guidance preset, and 40 flow-UniPC renderer
steps. The fox remains recognizable and anatomically coherent across the clip,
walks continuously up the snow slope, and keeps stable fur, ears, legs, and tail
detail through the sunrise backlight. Camera/subject motion is continuous; no
blocky quantization noise, gross color shift, frozen duplication, or temporal
flicker is visible in the five-frame contact sheet and three full-resolution
inspection frames. All 33 decoded frame MD5 values are unique.

The container reports H.264, yuv420p, 640x368, 16 fps, 33 frames, and 2.0625
seconds. Resource sampling across 810 polls recorded 16.511 GiB peak Comfy
VRAM, 29.885 GiB peak process RSS, and 50.149 GiB peak process VMS/commit on the
24 GB / 64 GB test host. Wall time was 1,651.984 seconds.

This is a visual usability gate, not a pixelwise BF16 comparison: the accepted
BF16 baseline used 848x480 and 81 frames, so its planned target and camera path
differ. A same-geometry latent/prediction oracle remains a Core-readiness item.

## Standalone-file acceptance

The public runtime format was replaced with four standalone files in ComfyUI's
standard model directories: Planner and UMT5 in `text_encoders`, plus one high-
and one low-noise Wan file in `diffusion_models`. The exporter copies every
weight and stock-Comfy quantization tensor byte-for-byte from the validated
conversion workspace; only the safetensors header and embedded tokenizer/config
payloads are rebuilt.

Both INT8 and BF16 sets pass the strict single-file validator. Real construction
also passes for Qwen language/vision, the Bernini auxiliary heads, Core Wan
UMT5, and both 14.288B Wan experts, with no meta parameters left behind. The
final INT8 set then completed a 640x368, 33-frame, 16-fps T2V run using 5
MaskGIT, 2 VIT, and 10 renderer steps in 342.31 seconds. All 33 decoded frames
are unique; the fox, snow scene, and running motion are recognizable and
temporally coherent. Detail softness and tail/limb distortion remain at this
reduced step budget, so this result is a format/runtime gate rather than a
replacement for the production-step acceptance above.

A preceding 1/1/2-step run produced only a blurred orange subject and was
explicitly rejected as a quality result. It remains useful only as proof that
the graph can encode a complete 2.0625-second video.

## External-NVFP4 T2V acceptance

The external renderer pair comes from immutable revision
`d677618e260be0f6ec934d6c9f72876f89cffe62` of
`rzgar/Bernini-v2-ComfyUI`. Both single-file checkpoints are 8,045,131,882
bytes. Header inspection finds 2,313 tensors, all 40 Wan blocks, 406 `nvfp4`
quantization markers, the stock `model.diffusion_model.` prefix, and identical
normalized high/low key sets. The planner, Qwen, and UMT5 remain the validated
Balanced-INT8 package; only the two renderer `MODEL` inputs change.

The production run uses the same versioned fox prompt and seed 42 as the INT8
gate, with 25 MaskGIT steps, 5 VIT denoising steps, 40 renderer steps, 640x368,
33 frames, 16 fps, and a 2.0625-second duration. The result is a coherent fox
walk with 33/33 unique decoded-frame hashes. Subject anatomy, snow texture, and
motion remain usable with no block noise, frozen duplication, or gross temporal
flicker. Backlit fur color varies more than in the Balanced-INT8 sample, and the
known mild tail/rear-leg geometry issue remains, so this is a quality acceptance
rather than a parity claim.

This run deliberately does **not** establish an NVFP4 speed or isolated-memory
win. The Windows host uses driver 576.28 (CUDA 12.9 maximum), PyTorch
2.11.0+cu128, ComfyUI 0.34.0, and comfy-kitchen 0.2.31, so NVFP4 uses the eager
backend. Wall time is 1,518.031 seconds. The 22.701 GiB device peak includes an
unrelated IndexTTS process holding about 8.5 GiB, and the first version of the
monitor sampled the venv launcher rather than its compute child; its recorded
RSS/VMS values are invalid. A follow-up fixes process-tree sampling. An
optimized CUDA-13 run with no competing GPU process remains required before
publishing performance or memory comparisons.

Compatibility failures are recorded rather than hidden: PyTorch 2.7+cu128
lacks `torch.float4_e2m1fn_x2`, while PyTorch 2.12.1+cu130 cannot initialize on
driver 576.28 (`cudaErrorNotSupported`). Neither failure is caused by the
Bernini node graph or the checkpoint structure.

## GGUF Q4_K_S T2V acceptance

The high- and low-noise renderer files are each 8,756,353,664 bytes. Each has
1,095 tensors: 356 Q4_K, 44 Q5_K, 6 BF16, and 689 F32, including all 40 Wan
blocks and the repaired F32 5-D `patch_embedding.weight`. The normalized
high/low name, shape, and type contracts match exactly and have SHA-256
`e779b82a06707b08f85d03e812293e4a14a96eccf00f189f239443e366351b76`.
The high file SHA-256 is
`b72df0b32d305b7acade0a7245edde37716f7202f5fbc99fda43aedf5d1ebc87`;
the low file SHA-256 is
`d396d3dcf935deb5bb1d8e6c6735e644adfa23d277efe0c2865bba03d6b7c92b`.

The production run keeps the validated Balanced-INT8 planner, Qwen, and UMT5
and swaps only the high/low Wan renderer `MODEL` inputs to ComfyUI-GGUF. It
uses the same fox prompt and seed 42, 25 MaskGIT steps, 5 VIT denoising steps,
40 renderer steps, 640x368, 33 frames, 16 fps, and 2.0625 seconds. All 33
decoded-frame hashes are unique. The fox walk remains coherent with no frozen
duplication, gross block noise, or frame collapse. Compared with Balanced INT8,
the face is softer and isolated frames show an elongated lifted front paw or
mild rear-leg/tail-root merging. This is an experimental low-storage
acceptance, not a default-quality recommendation.

ComfyUI reports 21 minutes 58 seconds; runner wall time is 1,319.907 seconds.
Sampling recorded a 21.048 GiB peak Comfy-visible allocation, 33.801 GiB
process-tree RSS, and 44.996 GiB process-tree VMS/commit. The host used ComfyUI
0.33.0, PyTorch 2.7.0+cu128, `--lowvram`, and ComfyUI-GGUF's partial-compile
compatibility path. Another GPU process may have been active, so these are
diagnostic values rather than isolated-memory claims. CUDA-13 optimized speed,
repeat-lifecycle behavior, and the other five task gates remain pending.

## Cause of the unusable reduced T2I image

The original image combined several non-production choices:

1. It used 256x256 geometry, one MaskGIT step, one VIT denoising step, and only
   two renderer steps. The official T2I preset is 512x512 with 25, 5, and 50
   steps respectively. The reduced run was only intended to prove graph
   execution.
2. It selected Comfy's generic `uni_pc_bh2`. That implementation transforms
   samples through a VP/EDM noise schedule; Bernini uses Diffusers UniPC with
   flow prediction, where `alpha = 1 - sigma`. The update equations are not
   interchangeable.
3. Comfy's automatic dtype selection loaded the Wan experts and UMT5 in FP16,
   while the released pipeline explicitly uses BF16.
4. The example encoded only the raw user text in UMT5 and left the negative
   prompt empty. The official UI prefixes a task-specific system sentence and
   uses the standard Wan negative prompt.

The investigation fixed all four items together. It did not run a full
factorial ablation, so no percentage contribution is assigned to an individual
cause; the two-step budget and non-equivalent sampler were the largest
mathematical departures.

## Conditional-video quality methodology

The first V2V attempt used a synthetic almost-static zoom of the T2I fox and a
vague instruction to restyle it. It generated a clear, temporally unique video
but changed the subject into a flying whale. That artifact is rejected: codec
success and frame uniqueness do not establish conditional fidelity.

The rerun uses ByteDance's official case-1 source and exact edit instruction.
It preserves the dog, path, trees, framing, and motion while adding a stable
snowman. R2V likewise uses all five released case references rather than the
single crude smoke placeholder. It preserves the marble-statue identity while
integrating the headphones, shirt, shorts, bench, and sunset environment.

RV2V required two rejected runs before acceptance. The first test fixture
incorrectly stretched ByteDance's 1080x1920 portrait source to 848x480; the
model preserved that distortion, so the artifact was rejected rather than
counted as a pass. A corrected 480x848 run then exposed a native implementation
bug: the combined renderer context packed source-video tokens before
reference-image tokens. The released renderer packs reference images first,
then source video, then the target. Since Wan assigns source ids and RoPE by
stream position, the reversed order caused the model to close the replacement
shirt and erase the required yellow-and-white inner T-shirt.

The renderer now uses the official image-then-video order and has a regression
test for it. The final 368x640, long-edge-640 run correctly applies the white
pinstriped outer shirt while preserving the striped inner shirt, person, pants,
studio background, framing, and arm motion. Both rejected videos and their QA
frames remain under `artifacts/rejected` so the failure history is auditable.

These cases demonstrate that the native media-conditioning paths work when
tested with a meaningful source and concrete edit prompt. They do not claim
pixel equality with the released output.

## Numerical checks supporting the runs

- The dedicated flow UniPC sampler was compared step by step with Diffusers
  `UniPCMultistepScheduler` at 1, 2, 3, 8, and 50 steps. The largest absolute
  trajectory difference in the deterministic fixture was about `3.6e-7`.
- The native Qwen2.5-VL MRoPE indices matched the official function element for
  element on representative image and video token sequences.
- Both Wan experts switch in one continuous sampler history. The loader now
  reserves estimated activation memory before each switch; this prevents the
  24 GB Windows OOM observed in the first full-resolution video attempt.
- The combined RV2V VAE context is regression-tested in the released
  reference-image-then-source-video order.
- The complete Python suite passes 148 tests. Ruff lint passes and all Python
  files pass `ruff format --check`.

## Remaining Core-readiness gates

- Official-vs-native planner hidden state, VIT target, and Wan prediction tensor
  comparisons within agreed BF16 tolerances
- Repeated prompts in one process on a current supported PyTorch/CUDA stack
- Interrupted download/repack recovery and Linux execution
- A final upstream-friendly split and review against current ComfyUI Core
