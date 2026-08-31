# Production-setting quality-test record

## Accepted results

The native pipeline has usable, real-weight visual baselines for all six
released tasks. These were queued through ComfyUI's `/prompt` API and used the
released task presets for planning, renderer steps, guidance, and flow shift.
Conditional-video cases use 33 frames at 16 fps (2.0625 seconds) as an explicit
duration override. V2V and R2V use 848x480; the final RV2V rerun preserves its
portrait aspect at 368x640 to follow the later long-edge-640 test budget. T2V
was completed earlier at the released 848x480, 81-frame duration.

| Task | Geometry / duration | Wall time | Visual acceptance | SHA-256 |
|---|---|---:|---|---|
| T2I | 512x512 PNG | 588.34 s | sharp, prompt-consistent fox in snow | `b246ea6a6249922d91a96e69a22db662a7f098b7175968adb58f746f6f1a5479` |
| I2I | 512x512 PNG | 659.97 s | reference identity/composition preserved with requested change | `997d2359db10a8776e3b86d926b88c9e6c5c52ed8255872df22226555b29d0ce` |
| T2V | 848x480, 81 frames, 5.0625 s | 4160.44 s | coherent fox walk with continuous motion | `b9b1446ac9c50516397646579b60ff696df7ecb178818945a084eadbf66c9571` |
| V2V | 848x480, 33 frames, 2.0625 s | 3769.25 s | official dog scene and motion preserved; stable snowman added | `853d5ad0cd8d2043184655bde551f6b6320a3c8b3b715c71e973c9a3829895f6` |
| R2V | 848x480, 33 frames, 2.0625 s | 4080.58 s | all five official references integrated into one stable scene | `c98aafe34bf3ae78e71db9d84cdca2392af623cade9bbda7f4374dc51434a462` |
| RV2V | 368x640, 33 frames, 2.0625 s | 7133.09 s | reference shirt applied; inner shirt, identity, scene, and motion preserved | `201fdfeb018b709374cad7aa2f601072036dc0721a84e10a8b5cb790437b2557` |

All accepted videos decode at 16 fps and have a unique MD5 for every frame.
Quality artifacts and contact sheets remain local under `artifacts/quality` and
`artifacts/qa`; they are test evidence, not files intended for a Core patch.

Accepted video deliverables:

- `artifacts/quality/bernini_v2_t2v_full_848x480_81f.mp4`
- `artifacts/quality/bernini_v2_v2v_official_case1_848x480_33f.mp4`
- `artifacts/quality/bernini_v2_r2v_official_5ref_848x480_33f.mp4`
- `artifacts/quality/bernini_v2_rv2v_official_case1_long640_33f.mp4`

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
- The complete Python suite passes 76 tests. Ruff lint passes and all 57 Python
  files pass `ruff format --check`.

## Remaining Core-readiness gates

- Official-vs-native planner hidden state, VIT target, and Wan prediction tensor
  comparisons within agreed BF16 tolerances
- Repeated prompts in one process on a current supported PyTorch/CUDA stack
- Interrupted download/repack recovery and Linux execution
- A final upstream-friendly split and review against current ComfyUI Core
