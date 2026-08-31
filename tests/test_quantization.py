import pytest
import torch

from bernini_v2.quantization import (
    best_convrot_group,
    classify_int8_weight,
    comfy_quant_marker,
    parse_comfy_quant_marker,
)


def test_convrot_group_prefers_largest_supported_divisor():
    assert best_convrot_group(4096) == 256
    assert best_convrot_group(320) == 64
    assert best_convrot_group(144) == 16
    assert best_convrot_group(130) is None


def test_balanced_profile_quantizes_large_linear_but_not_embedding():
    decision = classify_int8_weight("mllm", "model.layers.0.mlp.up_proj.weight", (4096, 3584))
    assert decision.quantize and decision.group_size == 256
    embedding = classify_int8_weight("mllm", "model.embed_tokens.weight", (152064, 3584))
    assert not embedding.quantize
    assert embedding.reason == "quality-sensitive"


def test_renderer_profile_leaves_planner_in_bf16():
    decision = classify_int8_weight(
        "mllm",
        "model.layers.0.self_attn.q_proj.weight",
        (3584, 3584),
        profile="renderer",
    )
    assert decision.reason == "component-passthrough"


def test_small_and_unaligned_layers_are_not_quantized():
    assert not classify_int8_weight("wan_high", "blocks.0.ffn.weight", (128, 4096)).quantize
    assert not classify_int8_weight("wan_high", "blocks.0.ffn.weight", (512, 130)).quantize


def test_stock_comfy_marker_round_trip():
    marker = comfy_quant_marker(256)
    assert marker.dtype == torch.uint8
    assert parse_comfy_quant_marker(marker) == {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": 256,
    }


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="unknown quantization profile"):
        classify_int8_weight("wan_high", "x.weight", (512, 512), profile="gguf")
