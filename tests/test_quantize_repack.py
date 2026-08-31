import json

import torch
from safetensors.torch import save_file

from tools.quantize_repack import quantize_repack
from tools.validate_repack import validate_repack


def test_quantize_repack_writes_stock_int8_shards_and_manifest(tmp_path, monkeypatch):
    source = tmp_path / "source"
    component = source / "wan_high"
    component.mkdir(parents=True)
    shard = component / "model-00001-of-00001.safetensors"
    save_file({"blocks.0.ffn.weight": torch.ones(16, 16)}, shard)
    index = {
        "metadata": {"total_size": 16 * 16 * 4},
        "weight_map": {"blocks.0.ffn.weight": shard.name},
    }
    (component / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    manifest = {
        "storage_dtype": "preserve",
        "metadata_files": [],
        "outputs": {"wan_high": {"tensors": 1, "total_size": 16 * 16 * 4, "sha256": {}}},
    }
    (source / "repack-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def fake_quantize(weight, *, group_size, device):
        del group_size, device
        return (
            torch.zeros_like(weight, dtype=torch.int8),
            torch.ones(weight.shape[0], 1, dtype=torch.float32),
            0.999,
            1.0,
        )

    monkeypatch.setattr("tools.quantize_repack._quantize_weight", fake_quantize)
    output = tmp_path / "output"
    report = quantize_repack(source, output, profile="renderer", min_gemm=0, device="cpu")
    assert report["outputs"]["wan_high"]["quantized_layers"] == 1
    assert report["quantization"]["runtime"]["device"] == "cpu"
    progress = json.loads((output / ".quantize-progress.json").read_text(encoding="utf-8"))
    assert progress["schema_version"] == 2
    assert progress["quantizer_runtime"]["torch"] == str(torch.__version__)
    validated = validate_repack(output, verify_hashes=True)
    assert validated["wan_high"]["quantized_layers"] == 1


def test_quantize_repack_quality_gate_falls_back_to_bf16(tmp_path, monkeypatch):
    source = tmp_path / "source"
    component = source / "wan_high"
    component.mkdir(parents=True)
    shard = component / "model.safetensors"
    save_file({"blocks.0.ffn.weight": torch.ones(16, 16)}, shard)
    manifest = {"metadata_files": [], "outputs": {"wan_high": {"tensors": 1}}}
    (source / "repack-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        "tools.quantize_repack._quantize_weight",
        lambda weight, **kwargs: (
            torch.zeros_like(weight, dtype=torch.int8),
            torch.ones(weight.shape[0], 1),
            0.8,
            20.0,
        ),
    )
    report = quantize_repack(source, tmp_path / "output", profile="renderer", min_gemm=0, device="cpu")
    assert report["outputs"]["wan_high"]["quantized_layers"] == 0
    assert report["outputs"]["wan_high"]["quality_fallbacks"] == 1
