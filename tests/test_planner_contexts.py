import torch

from bernini_v2.planner import _conditioning_tensor, _pad_renderer_context, split_reference_images


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


def test_reference_autogrow_inputs_keep_numeric_order_after_nine():
    references = {
        "reference_image_10": torch.full((1, 1, 1, 1), 10),
        "reference_image_2": torch.full((1, 1, 1, 1), 2),
        "reference_image_1": torch.full((1, 1, 1, 1), 1),
    }
    ordered = split_reference_images(references)
    assert [int(item.item()) for item in ordered] == [1, 2, 10]
