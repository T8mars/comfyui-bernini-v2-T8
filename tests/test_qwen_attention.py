import pytest
import torch

from bernini_v2.qwen import (
    _explicit_gqa_attention,
    _official_qwen_layer_forward,
    _official_qwen_rms_norm,
)


def test_explicit_gqa_attention_repeats_kv_heads_in_official_order():
    captured = {}

    def attention(query, key, value, heads, **kwargs):
        captured.update(query=query, key=key, value=value, heads=heads, kwargs=kwargs)
        return query

    query = torch.zeros(1, 4, 2, 3)
    key = torch.arange(12).reshape(1, 2, 2, 3)
    value = key + 100
    result = _explicit_gqa_attention(attention)(
        query,
        key,
        value,
        4,
        enable_gqa=True,
        mask="mask",
    )

    assert result is query
    assert captured["key"].shape == query.shape
    assert captured["key"][0, :, 0, 0].tolist() == [0, 0, 6, 6]
    assert captured["value"][0, :, 0, 0].tolist() == [100, 100, 106, 106]
    assert captured["kwargs"] == {"mask": "mask"}


def test_explicit_gqa_attention_rejects_non_divisible_heads():
    query = torch.zeros(1, 3, 2, 4)
    key = torch.zeros(1, 2, 2, 4)

    with pytest.raises(ValueError, match="divisible"):
        _explicit_gqa_attention(lambda *args, **kwargs: None)(query, key, key, 3, enable_gqa=True)


def test_explicit_gqa_attention_preserves_non_gqa_call():
    captured = {}

    def attention(query, key, value, heads, **kwargs):
        captured.update(key=key, value=value, heads=heads, kwargs=kwargs)
        return query

    query = torch.zeros(1, 2, 1, 4)
    key = torch.ones_like(query)
    _explicit_gqa_attention(attention)(query, key, key, 2, scale=0.5)

    assert captured == {"key": key, "value": key, "heads": 2, "kwargs": {"scale": 0.5}}


def test_official_qwen_rms_norm_matches_reference_arithmetic():
    class Norm:
        eps = 1e-6
        weight = torch.linspace(0.5, 1.5, 16, dtype=torch.bfloat16)

    hidden = torch.randn(2, 3, 16, dtype=torch.bfloat16)
    expected = hidden.float()
    variance = expected.pow(2).mean(-1, keepdim=True)
    expected = Norm.weight * (expected * torch.rsqrt(variance + Norm.eps)).to(hidden.dtype)

    actual = _official_qwen_rms_norm(Norm, hidden)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_official_qwen_layer_forward_matches_reference_residual_order():
    class Norm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.eps = 1e-6
            self.weight = torch.nn.Parameter(torch.ones(4))

    class Attention(torch.nn.Module):
        def forward(self, hidden_states, **kwargs):
            assert kwargs["attention_mask"] == "mask"
            assert kwargs["freqs_cis"] == "rope"
            assert kwargs["optimized_attention"] == "attention"
            assert kwargs["past_key_value"] is None
            return hidden_states + 0.25, "cache"

    class Layer:
        input_layernorm = Norm()
        post_attention_layernorm = Norm()
        self_attn = Attention()
        mlp = staticmethod(lambda value: value * 0.5)

    hidden = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    normalized = _official_qwen_rms_norm(Layer.input_layernorm, hidden)
    after_attention = hidden + normalized + 0.25
    expected = after_attention + 0.5 * _official_qwen_rms_norm(Layer.post_attention_layernorm, after_attention)

    actual, cache = _official_qwen_layer_forward(Layer, hidden, "mask", "rope", "attention")

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert cache == "cache"
