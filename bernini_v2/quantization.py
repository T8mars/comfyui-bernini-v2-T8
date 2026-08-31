"""Architecture-aware selection and metadata for stock Comfy INT8 ConvRot."""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch

VALID_CONVROT_GROUPS = (256, 64, 16)
QUANT_COMPONENT_PROFILES = {
    "renderer": frozenset({"wan_high", "wan_low"}),
    "balanced": frozenset({"wan_high", "wan_low", "mllm", "t5_text_encoder"}),
    "full": frozenset({"wan_high", "wan_low", "mllm", "t5_text_encoder", "connector", "vit_decoder"}),
}

# These are embeddings, conditioning adapters, modulation paths, or output
# projections where small weight error is disproportionately visible.
HIGH_RISK_NAMES = (
    "embed_tokens",
    "shared.weight",
    "token_embedding",
    "position_embedding",
    "patch_embedding",
    "patch_embed",
    "time_embed",
    "time_embedding",
    "adaln",
    "modulation",
    "final_layer",
    "proj_out",
    "output_projection",
    "lm_head",
)


@dataclass(frozen=True)
class QuantDecision:
    quantize: bool
    reason: str
    group_size: int | None = None


def best_convrot_group(in_features: int) -> int | None:
    return next((group for group in VALID_CONVROT_GROUPS if in_features % group == 0), None)


def classify_int8_weight(
    component: str,
    key: str,
    shape: tuple[int, ...],
    *,
    profile: str = "balanced",
    min_gemm: int = 256,
) -> QuantDecision:
    if profile not in QUANT_COMPONENT_PROFILES:
        raise ValueError(f"unknown quantization profile: {profile}")
    if component not in QUANT_COMPONENT_PROFILES[profile]:
        return QuantDecision(False, "component-passthrough")
    if not key.endswith(".weight") or len(shape) != 2:
        return QuantDecision(False, "not-linear-weight")
    lowered = key.lower()
    if any(name in lowered for name in HIGH_RISK_NAMES):
        return QuantDecision(False, "quality-sensitive")
    if min_gemm and min(shape) < min_gemm:
        return QuantDecision(False, f"below-min-gemm-{min_gemm}")
    group_size = best_convrot_group(shape[1])
    if group_size is None:
        return QuantDecision(False, "convrot-incompatible")
    return QuantDecision(True, "int8-convrot", group_size)


def comfy_quant_marker(group_size: int) -> torch.Tensor:
    payload = {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": group_size,
    }
    return torch.tensor(list(json.dumps(payload).encode("utf-8")), dtype=torch.uint8)


def parse_comfy_quant_marker(marker: torch.Tensor) -> dict[str, object]:
    if marker.dtype != torch.uint8 or marker.ndim != 1:
        raise ValueError("comfy_quant marker must be a one-dimensional uint8 tensor")
    return json.loads(marker.numpy().tobytes())
