import json

import pytest

from tools.convert_sharded_gguf import (
    TensorDescriptor,
    convert_sharded_checkpoint,
    load_weight_map,
    resolve_index,
    target_qtype_name,
)


def descriptor(name, shape, dtype="BF16"):
    return TensorDescriptor(name=name, shard="model-00001-of-00001.safetensors", dtype=dtype, shape=shape)


def test_resolve_index_accepts_component_directory(tmp_path):
    index = tmp_path / "model.safetensors.index.json"
    index.write_text('{"weight_map":{"x":"part.safetensors"}}', encoding="utf-8")
    (tmp_path / "part.safetensors").write_bytes(b"fixture")
    assert resolve_index(tmp_path) == index.resolve()
    weight_map, _ = load_weight_map(index)
    assert weight_map == {"x": "part.safetensors"}


def test_load_weight_map_rejects_missing_shards(tmp_path):
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"x": "missing.safetensors"}}), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing source shards"):
        load_weight_map(index)


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        (descriptor("blocks.0.ffn.0.weight", (5120, 5120)), "BF16"),
        (descriptor("blocks.0.norm.weight", (5120,)), "F32"),
        (descriptor("head.modulation", (1, 6, 5120)), "F32"),
        (descriptor("tiny.weight", (16, 16)), "F32"),
        (descriptor("fp32.weight", (2048, 2048), dtype="F32"), "F16"),
    ],
)
def test_target_qtype_matches_comfyui_gguf_wan_policy(item, expected):
    assert target_qtype_name(item) == expected


def test_tiny_sharded_wan_converts_with_disk_spool_and_5d_fix(tmp_path):
    gguf = pytest.importorskip("gguf")
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")

    component = tmp_path / "wan_high"
    component.mkdir()
    first = {
        "blocks.0.self_attn.norm_q.weight": torch.ones(16, dtype=torch.bfloat16),
        "text_embedding.2.weight": torch.ones(16, 16, dtype=torch.bfloat16),
    }
    second = {
        "head.modulation": torch.ones(1, 6, 16, dtype=torch.bfloat16),
        "patch_embedding.weight": torch.ones(16, 4, 1, 2, 2, dtype=torch.bfloat16),
    }
    first_name = "model-00001-of-00002.safetensors"
    second_name = "model-00002-of-00002.safetensors"
    safetensors_torch.save_file(first, component / first_name)
    safetensors_torch.save_file(second, component / second_name)
    weight_map = {name: first_name for name in first} | {name: second_name for name in second}
    total_size = sum(tensor.numel() * tensor.element_size() for tensor in [*first.values(), *second.values()])
    (component / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}),
        encoding="utf-8",
    )

    output = tmp_path / "wan-high-BF16.gguf"
    fix = tmp_path / "wan-high-5d.safetensors"
    report = convert_sharded_checkpoint(component, output, fix_path=fix, temp_root=tmp_path)

    assert output.is_file()
    assert fix.is_file()
    assert report["tensors"] == 4
    assert report["gguf_tensors"] == 3
    assert report["quantization_counts"] == {"5D_FIX": 1, "F32": 3}
    assert not list(tmp_path.glob("*.partial"))
    reader = gguf.GGUFReader(str(output))
    assert {tensor.name for tensor in reader.tensors} == {
        "blocks.0.self_attn.norm_q.weight",
        "head.modulation",
        "text_embedding.2.weight",
    }
    fixed = safetensors_torch.load_file(fix)
    assert fixed["patch_embedding.weight"].dtype == torch.float32
