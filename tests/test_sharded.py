from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from bernini_v2.sharded import component_checkpoint, load_checkpoint_state_dict, load_sharded_state_dict


def test_load_sharded_state_dict(tmp_path: Path) -> None:
    save_file({"a": torch.arange(4), "b": torch.ones(2)}, tmp_path / "one.safetensors")
    save_file({"c": torch.zeros(3)}, tmp_path / "two.safetensors")
    index = {
        "metadata": {"total_size": 72},
        "weight_map": {"a": "one.safetensors", "b": "one.safetensors", "c": "two.safetensors"},
    }
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    state_dict = load_sharded_state_dict(index_path)
    assert set(state_dict) == {"a", "b", "c"}
    torch.testing.assert_close(state_dict["a"], torch.arange(4))


def test_missing_shard_is_rejected(tmp_path: Path) -> None:
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": {"a": "missing.safetensors"}}), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_sharded_state_dict(index_path)


def test_shard_path_escape_is_rejected(tmp_path: Path) -> None:
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": {"a": "../outside.safetensors"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        load_sharded_state_dict(index_path)


def test_load_checkpoint_state_dict_accepts_single_safetensors(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    save_file({"weight": torch.arange(3)}, checkpoint)
    assert torch.equal(load_checkpoint_state_dict(checkpoint)["weight"], torch.arange(3))


def test_component_checkpoint_prefers_sharded_index(tmp_path: Path) -> None:
    component = tmp_path / "wan_high"
    component.mkdir()
    index = component / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a": "one.safetensors"}}), encoding="utf-8")
    save_file({"a": torch.ones(1)}, component / "one.safetensors")
    assert component_checkpoint(tmp_path, "wan_high") == index
