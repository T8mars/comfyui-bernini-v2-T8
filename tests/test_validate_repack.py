import json

import pytest
import torch
from safetensors.torch import save_file

from tools.validate_repack import file_sha256, validate_index, validate_repack


def _write_tiny_repack(root):
    component = root / "component"
    component.mkdir(parents=True)
    shard = component / "model-00001-of-00001.safetensors"
    save_file({"weight": torch.ones(2, 3, dtype=torch.bfloat16)}, shard)
    index = {
        "metadata": {"total_size": 12},
        "weight_map": {"weight": shard.name},
    }
    (component / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    manifest = {
        "outputs": {
            "component": {
                "tensors": 1,
                "total_size": 12,
                "sha256": {shard.name: file_sha256(shard)},
            }
        }
    }
    (root / "repack-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_validates_index_tensor_bytes_and_manifest_hash(tmp_path):
    _write_tiny_repack(tmp_path)
    assert validate_index(tmp_path / "component" / "model.safetensors.index.json") == {
        "tensors": 1,
        "shards": 1,
        "total_size": 12,
    }
    assert validate_repack(tmp_path, verify_hashes=True)["component"]["tensors"] == 1


def test_rejects_index_total_size_mismatch(tmp_path):
    _write_tiny_repack(tmp_path)
    index_path = tmp_path / "component" / "model.safetensors.index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["metadata"]["total_size"] = 13
    index_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="declared total_size"):
        validate_index(index_path)
