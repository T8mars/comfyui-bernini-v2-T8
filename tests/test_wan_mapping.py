from __future__ import annotations

import json
from pathlib import Path

from bernini_v2.state_dict import Component, component_key
from bernini_v2.wan_mapping import expected_comfy_wan_keys, wan_diffusers_to_comfy


def test_global_wan_mapping() -> None:
    assert wan_diffusers_to_comfy("scale_shift_table") == "head.modulation"
    assert wan_diffusers_to_comfy("proj_out.weight") == "head.head.weight"
    assert wan_diffusers_to_comfy("condition_embedder.time_embedder.linear_1.weight") == "time_embedding.0.weight"


def test_block_wan_mapping() -> None:
    assert wan_diffusers_to_comfy("blocks.7.attn1.to_q.weight") == "blocks.7.self_attn.q.weight"
    assert wan_diffusers_to_comfy("blocks.7.attn2.norm_k.weight") == "blocks.7.cross_attn.norm_k.weight"
    assert wan_diffusers_to_comfy("blocks.7.ffn.net.0.proj.bias") == "blocks.7.ffn.0.bias"
    assert wan_diffusers_to_comfy("blocks.7.norm2.weight") == "blocks.7.norm3.weight"


def test_expected_native_key_count() -> None:
    assert len(expected_comfy_wan_keys(40)) == 1095


def test_official_metadata_has_total_wan_coverage() -> None:
    index = Path("models/ByteDance/Bernini-Diffusers-v2/bernini/model.safetensors.index.json")
    if not index.is_file():
        return
    weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    for component in (Component.WAN_HIGH, Component.WAN_LOW):
        mapped = {
            wan_diffusers_to_comfy(component_key(key)[1]) for key in weight_map if component_key(key)[0] is component
        }
        assert mapped == expected_comfy_wan_keys(40)
