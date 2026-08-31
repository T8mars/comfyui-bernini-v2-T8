import torch

from bernini_v2.planner import _conditioning_tensor, _pad_renderer_context


def test_comfy_t5_padding_is_trimmed_before_qwen_concat():
    tensor = torch.zeros(1, 512, 4096)
    tensor[:, :7] = 1
    result = _conditioning_tensor([[tensor, {}]], "positive")
    assert result.shape == (1, 7, 4096)


def test_renderer_context_is_padded_after_concat():
    context = torch.ones(1, 10, 4096)
    result = _pad_renderer_context(context)
    assert result.shape == (1, 512, 4096)
    torch.testing.assert_close(result[:, :10], context)
    assert torch.count_nonzero(result[:, 10:]) == 0


def test_long_renderer_context_is_not_truncated_by_default():
    context = torch.ones(1, 600, 4096)
    assert _pad_renderer_context(context) is context
