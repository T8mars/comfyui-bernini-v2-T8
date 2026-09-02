# Bernini v2 parity matrix

| Area | Official behavior | Native implementation | Gate |
|---|---|---|---|
| Renderer models | Wan2.2 high/low experts | Native Comfy Wan models | 1095/1095 keys each; official/native 40-block prediction tensor parity passes for both experts |
| Expert switch | Timestep `< 875` uses low expert | One dual-expert guider at sigma `< 0.875` | Switch observed in real sampling, including full 50-step T2I |
| Scheduler | UniPC order 2, BH2, flow prediction | Dedicated flow UniPC sampler plus exact sigma node | Diffusers stepwise fixture at 1/2/3/8/50 steps; max error about `3.6e-7` |
| Planner LM | Qwen2.5-VL hidden state `[-2]` | Native Comfy Qwen layer 26 output | Real BF16 oracle passes: NRMSE `0.002838`, cosine `1.0` |
| Planner branches | cond / uncond / text-only | Three native branches | Structural tests and six-task real forward pass |
| VIT generation | MaskGIT + flow decoder | Native VIT decoder | Real BF16 oracle target passes elementwise; cosine `0.999907` |
| Renderer text | BF16 UMT5 with task system prefix, official negative; cropped T5 then Qwen | Explicit BF16 Wan T5, official example text, cropped concat/padding | Production-setting T2I/I2I/video passes plus unit tests |
| Standard guidance | source / text / target APG chain | Flow-velocity APG | Unit tests plus t2i/i2i/t2v/v2v/r2v real sampling |
| rv2v guidance | separate video/image direct chain | Five-arm flow-velocity chain | Unit tests plus official-case long-edge-640 quality pass |
| Source media | Image VAE streams, then video streams, then target | Standard Comfy VAE and ordered `context_latents` | Image/video paths pass; combined ordering regression and RV2V quality pass |
| Low-memory weights | Official BF16 is the quality oracle | True-BF16 package plus architecture-aware stock-Comfy INT8 ConvRot; external standard `MODEL` inputs accept NVFP4/FP8/GGUF renderers | Balanced INT8 and CUDA-12.8-eager NVFP4 production visual gates pass; NVFP4 optimized performance and GGUF end-to-end acceptance are pending |
| Lifecycle | Repeated calls reuse one pipeline | Comfy model patchers/offload with sequential planner and guidance arms | Two uncached 640x368, 33-frame balanced-INT8 jobs pass in one server; current-CUDA lane pending |

## Task coverage

| Task | Required inputs | Official preset encoded | Workflow | Smoke test | Production-setting quality |
|---|---|---:|---:|---:|---:|
| t2i | text | yes | API + editable frontend graph validated | real-weight minimal pass | 512x512, 25/5/50 pass |
| i2i | one image | yes | API + editable frontend graph validated | real-weight minimal pass | 512x512 official-preset pass |
| t2v | text | yes | API + editable frontend graph validated | real-weight 5-frame pass | 848x480, 81-frame official-preset pass |
| v2v | one video | yes | API + editable frontend graph validated | real-weight 5-frame pass | official case, 848x480, 33-frame pass |
| r2v | one or more reference images | yes | API + editable frontend graph validated | real-weight 5-frame pass | official five-reference case, 848x480, 33-frame pass |
| rv2v | video plus reference images | yes | API + editable frontend graph validated | real-weight 5-frame pass | official case, 368x640, 33-frame pass |

The historical six-task smoke passes used one MaskGIT planning step, one VIT
denoising step, and two renderer steps at 256 px. The new repeat-lifecycle smoke
uses the same reduced steps at 640x368 and 33 frames. They validate execution
paths, not published-preset quality. See `SMOKE_TESTS.md` for evidence and
remaining gates.
Accepted production-setting baselines and rejected quality cases are recorded
separately in `QUALITY_TESTS.md`. The 33-frame conditional-video cases are an
explicit 2.0625-second test override; the latest RV2V case also uses a
long-edge-640 test budget. This custom-node package's versioned video workflows
now default to 33 frames and a 640-pixel long edge for practical testing; a
public Core examples should explicitly restore the released 81-frame preset.
