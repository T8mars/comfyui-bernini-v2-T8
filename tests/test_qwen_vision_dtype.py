import torch

from bernini_v2.qwen import apply_qwen_vision_rope


def test_qwen_vision_rope_restores_bfloat16_dtype():
    query = torch.randn(8, 2, 4, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    cos = torch.randn(8, 4, dtype=torch.float32)
    sin = torch.randn(8, 4, dtype=torch.float32)

    rotated_query, rotated_key = apply_qwen_vision_rope(query, key, cos, sin)

    assert rotated_query.dtype == torch.bfloat16
    assert rotated_key.dtype == torch.bfloat16


def test_qwen_vision_rope_matches_official_fp32_then_cast_behavior():
    query = torch.randn(8, 2, 4, dtype=torch.float16)
    key = torch.randn_like(query)
    cos = torch.randn(8, 4)
    sin = torch.randn(8, 4)

    actual_query, actual_key = apply_qwen_vision_rope(query, key, cos, sin)
    cos_expanded = cos.unsqueeze(-2).float()
    sin_expanded = sin.unsqueeze(-2).float()

    def rotate_half(value):
        return torch.cat((-value[..., 2:], value[..., :2]), dim=-1)

    expected_query = (query.float() * cos_expanded + rotate_half(query.float()) * sin_expanded).to(query.dtype)
    expected_key = (key.float() * cos_expanded + rotate_half(key.float()) * sin_expanded).to(key.dtype)

    torch.testing.assert_close(actual_query, expected_query)
    torch.testing.assert_close(actual_key, expected_key)
