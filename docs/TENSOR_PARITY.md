# Official/native tensor parity

`tools/run_tensor_parity.py` captures deterministic tensors from ByteDance's
official Bernini v2 implementation and this native ComfyUI implementation in
separate processes, then writes an inspectable JSON comparison report. The
separate-process design prevents the two large implementations from occupying
GPU memory at the same time.

## Acceptance contract

Every tensor must have the same name, shape, dtype, and finite-value mask. A
numerical tensor passes by either of these BF16 paths:

- elementwise: `abs(native - official) <= 0.125 + 0.05 * abs(official)`;
- accumulated: normalized RMSE `<= 0.015` and cosine similarity `>= 0.999`.

The accumulated path is needed for deep residual stacks, where small BF16
rounding differences compound without indicating a different computation.
Reports retain both elementwise violation counts and aggregate metrics, and
record which path accepted each tensor.

## Pinned oracle

- Official source: ByteDance Bernini commit `e6c2cf1`.
- Renderer source revision: `399cf6a18a4c523b367b2b1ac25a2a61009e7df3`.
- Planner BF16 SHA-256:
  `686437fda8400ca1ee69f8436c2d546334f781360a8bc1416467845436877f3f`.
- Runtime used for the recorded run: PyTorch `2.7.0+cu128`, RTX 5090 Laptop
  GPU, official SDPA fallback.

The official Wan oracle and native Wan path are streamed one Transformer block
at a time. This executes all 40 real-weight blocks while keeping peak model
residency compatible with a 24 GB GPU; it is not a reduced-width or synthetic
model.

## Recorded results

| Stage | Fixture | Result | Key metrics |
|---|---|---|---|
| Planner hidden state `[-2]` | seed 42, 16 real embedding-table token IDs, causal mask | pass (accumulated) | max abs `2.0`; NRMSE `0.002838`; cosine `1.0`; 201 elementwise violations out of 57,344 |
| VIT target | seed 42, three branches, two target tokens, one released T2V decoder step | pass (elementwise) | max abs `0.25`; NRMSE `0.013799`; cosine `0.999907`; zero violations |
| Wan high-noise prediction | timestep `999`, latent `[1,16,1,4,4]`, context `[1,8,4096]`, 40 blocks | pass (elementwise) | prediction max abs `0.030518`; cosine `0.999493`; block 39 NRMSE `0.011606` |
| Wan low-noise prediction | timestep `500`, same shapes, 40 blocks | pass (elementwise) | prediction max abs `0.019531`; cosine `0.999787`; block 39 NRMSE `0.010399` |

The Planner investigation found and fixed three concrete differences in the
native path: explicit Qwen GQA K/V-head expansion, BF16 RoPE inputs, and the
official unfused RMSNorm arithmetic. First-layer Q/K/V, attention, projection,
and MLP probes now pass the elementwise gate before the full-stack accumulated
gate is applied.

## Reproduction

The official checkout and pinned VeOmni checkout may be supplied explicitly
with `--official-repo` and `--veomni-repo`. The default paths target this
repository's local validation workspace.

```powershell
python tools/run_tensor_parity.py run --stage planner_hidden
python tools/run_tensor_parity.py run --stage vit_target --token-count 2 --vit-steps 1
python tools/run_tensor_parity.py run --stage wan_prediction --renderer-expert high
python tools/run_tensor_parity.py run --stage wan_prediction --renderer-expert low
```

Artifacts and reports are written to `artifacts/parity/`, which is intentionally
git-ignored because the captures are generated evidence tied to local model
files. The durable result and exact acceptance policy are recorded here.
