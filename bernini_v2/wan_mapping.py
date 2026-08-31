"""Map Diffusers Wan2.2 tensor names to ComfyUI's native Wan names."""

from __future__ import annotations

import re

_BLOCK = re.compile(r"^blocks\.(?P<index>\d+)\.(?P<suffix>.+)$")

_GLOBAL_NAMES = {
    "patch_embedding.weight": "patch_embedding.weight",
    "patch_embedding.bias": "patch_embedding.bias",
    "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
    "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
    "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
    "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
    "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
    "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
    "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
    "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
    "condition_embedder.time_proj.weight": "time_projection.1.weight",
    "condition_embedder.time_proj.bias": "time_projection.1.bias",
    "scale_shift_table": "head.modulation",
    "proj_out.weight": "head.head.weight",
    "proj_out.bias": "head.head.bias",
}

_BLOCK_NAMES = {
    "scale_shift_table": "modulation",
    "attn1.to_q.weight": "self_attn.q.weight",
    "attn1.to_q.bias": "self_attn.q.bias",
    "attn1.to_k.weight": "self_attn.k.weight",
    "attn1.to_k.bias": "self_attn.k.bias",
    "attn1.to_v.weight": "self_attn.v.weight",
    "attn1.to_v.bias": "self_attn.v.bias",
    "attn1.to_out.0.weight": "self_attn.o.weight",
    "attn1.to_out.0.bias": "self_attn.o.bias",
    "attn1.norm_q.weight": "self_attn.norm_q.weight",
    "attn1.norm_k.weight": "self_attn.norm_k.weight",
    "attn2.to_q.weight": "cross_attn.q.weight",
    "attn2.to_q.bias": "cross_attn.q.bias",
    "attn2.to_k.weight": "cross_attn.k.weight",
    "attn2.to_k.bias": "cross_attn.k.bias",
    "attn2.to_v.weight": "cross_attn.v.weight",
    "attn2.to_v.bias": "cross_attn.v.bias",
    "attn2.to_out.0.weight": "cross_attn.o.weight",
    "attn2.to_out.0.bias": "cross_attn.o.bias",
    "attn2.norm_q.weight": "cross_attn.norm_q.weight",
    "attn2.norm_k.weight": "cross_attn.norm_k.weight",
    # Diffusers calls the affine cross-attention pre-norm norm2. Comfy's Wan
    # model calls the same module norm3; norm1/norm2 are non-affine.
    "norm2.weight": "norm3.weight",
    "norm2.bias": "norm3.bias",
    "ffn.net.0.proj.weight": "ffn.0.weight",
    "ffn.net.0.proj.bias": "ffn.0.bias",
    "ffn.net.2.weight": "ffn.2.weight",
    "ffn.net.2.bias": "ffn.2.bias",
}


def wan_diffusers_to_comfy(key: str) -> str:
    """Convert one bare Diffusers ``WanTransformer3DModel`` parameter name."""

    global_name = _GLOBAL_NAMES.get(key)
    if global_name is not None:
        return global_name

    match = _BLOCK.match(key)
    if match is None:
        raise KeyError(f"unknown Diffusers Wan key: {key}")
    target_suffix = _BLOCK_NAMES.get(match.group("suffix"))
    if target_suffix is None:
        raise KeyError(f"unknown Diffusers Wan block key: {key}")
    return f"blocks.{match.group('index')}.{target_suffix}"


def expected_comfy_wan_keys(num_layers: int) -> set[str]:
    """Return the native key set without importing or instantiating ComfyUI."""

    keys = set(_GLOBAL_NAMES.values())
    for index in range(num_layers):
        keys.update(f"blocks.{index}.{suffix}" for suffix in _BLOCK_NAMES.values())
    return keys
