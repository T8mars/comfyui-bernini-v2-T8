from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from bernini_v2.checkpoint_info import inspect_renderer_checkpoint, inspect_renderer_pair


def _renderer(path: Path, *, blocks: int = 40, prefix: str = "", legacy_fp8: bool = False) -> None:
    tensors = {
        f"{prefix}patch_embedding.weight": torch.ones(1, 1, 1, 1, 1, dtype=torch.bfloat16),
        f"{prefix}time_embedding.0.weight": torch.ones(1, 1, dtype=torch.bfloat16),
        f"{prefix}text_embedding.0.weight": torch.ones(1, 1, dtype=torch.bfloat16),
        f"{prefix}head.head.weight": torch.ones(1, 1, dtype=torch.bfloat16),
    }
    for block in range(blocks):
        tensors[f"{prefix}blocks.{block}.self_attn.q.weight"] = torch.ones(1, 1, dtype=torch.int8)
    base = f"{prefix}blocks.0.self_attn.q"
    if legacy_fp8:
        tensors["scaled_fp8"] = torch.ones(1, dtype=torch.float8_e4m3fn)
        tensors[f"{base}.scale_weight"] = torch.ones(1)
    else:
        tensors[f"{base}.weight_scale"] = torch.ones(1, 1)
        marker = json.dumps({"format": "int8_tensorwise"}).encode("utf-8")
        tensors[f"{base}.comfy_quant"] = torch.tensor(list(marker), dtype=torch.uint8)
    save_file(tensors, path)


def test_inspects_stock_prefixed_single_file_and_pair(tmp_path: Path) -> None:
    high = tmp_path / "high.safetensors"
    low = tmp_path / "low.safetensors"
    _renderer(high, prefix="model.diffusion_model.")
    _renderer(low, prefix="model.diffusion_model.")

    report = inspect_renderer_checkpoint(high)
    assert report["renderer_blocks"] == 40
    assert report["quant_formats"] == {"int8_tensorwise": 1}
    assert report["key_prefix"] == "model.diffusion_model."
    assert inspect_renderer_pair(high, low)["pair_keys_match"] is True


def test_rejects_incomplete_renderer(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.safetensors"
    _renderer(path, blocks=39)
    with pytest.raises(ValueError, match=r"missing|0\.\.39"):
        inspect_renderer_checkpoint(path)


def test_reports_legacy_scaled_fp8(tmp_path: Path) -> None:
    path = tmp_path / "legacy.safetensors"
    _renderer(path, legacy_fp8=True)
    report = inspect_renderer_checkpoint(path)
    assert report["quant_formats"] == {"legacy_scaled_fp8": 1}
    assert report["quantized_layers"] == 1
