from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from bernini_v2.template import BerniniTemplate, build_conversation


def _tokenizer():
    path = Path(__file__).parents[1] / "models" / "ByteDance" / "Bernini-Diffusers-v2" / "mllm"
    return AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)


def test_template_builds_cond_uncond_and_text_only_branches() -> None:
    template = BerniniTemplate(_tokenizer())
    conversation = build_conversation(
        "replace the sky",
        source_videos=1,
        source_images=2,
        output_is_image=False,
    )
    counts = {"video": [6, 8], "image": [4, 4]}
    cond = template.encode(
        conversation,
        num_tokens=counts,
        task="rv2v",
        mask_dtype=torch.bfloat16,
    )
    uncond = template.encode(
        conversation,
        num_tokens=counts,
        task="rv2v",
        drop_text=True,
        drop_images=True,
        drop_videos=True,
        negative_prompt="low quality",
    )
    text_only = template.encode(
        conversation,
        num_tokens=counts,
        task="rv2v",
        drop_images=True,
        drop_videos=True,
    )
    assert int(cond["visual_input_token_mask"].sum()) == 14
    assert int(cond["visual_output_token_mask"].sum()) == 8
    assert int(uncond["visual_input_token_mask"].sum()) == 0
    assert int(text_only["visual_input_token_mask"].sum()) == 0
    assert int(uncond["visual_output_token_mask"].sum()) == 8
    assert int(text_only["visual_output_token_mask"].sum()) == 8
    assert cond["vit_type_list"].tolist() == [1, 0, 0, 1]
    mask = cond["attention_mask_4d"]
    assert mask.dtype == torch.bfloat16
    assert mask.shape == (1, len(cond["input_ids"]), len(cond["input_ids"]))
    assert torch.isfinite(mask.diagonal(dim1=-2, dim2=-1)).all()


def test_template_preserves_one_character_negative_prompt() -> None:
    template = BerniniTemplate(_tokenizer())
    conversation = build_conversation(
        "replace the sky",
        source_videos=0,
        source_images=0,
        output_is_image=True,
    )
    encoded = template.encode(
        conversation,
        num_tokens={"video": [], "image": [4]},
        task="t2i",
        drop_text=True,
        negative_prompt="坏",
    )
    decoded = template.tokenizer.decode(encoded["input_ids"])
    assert "坏" in decoded
    assert "replace the sky" not in decoded


def test_template_reports_visual_item_limit() -> None:
    template = BerniniTemplate(_tokenizer())
    with pytest.raises(ValueError, match="at most 64 visual items, got 65"):
        template._visual_pattern(1, 64, output=False)
