import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tools.export_single_files import export_single_files, read_safetensors_header


def _component(root: Path, name: str, shards: list[dict[str, torch.Tensor]]) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    weight_map = {}
    total = 0
    for number, tensors in enumerate(shards, start=1):
        filename = f"model-{number:05d}-of-{len(shards):05d}.safetensors"
        save_file(tensors, directory / filename)
        for key, tensor in tensors.items():
            weight_map[key] = filename
            total += tensor.numel() * tensor.element_size()
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}), encoding="utf-8"
    )


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "repack"
    root.mkdir()
    _component(
        root,
        "mllm",
        [
            {"model.embed_tokens.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4)},
            {"visual.patch_embed.proj.weight": torch.arange(8, dtype=torch.float16).reshape(2, 4)},
        ],
    )
    _component(root, "connector", [{"fc1.weight": torch.ones(2, 2, dtype=torch.bfloat16)}])
    _component(root, "vit_decoder", [{"blocks.0.weight": torch.zeros(2, 2)}])
    _component(root, "mask_tokens", [{"mask_tokens": torch.ones(1, 2, 4)}])
    _component(root, "t5_text_encoder", [{"shared.weight": torch.eye(3)}])
    _component(root, "wan_high", [{"blocks.0.weight": torch.arange(4, dtype=torch.int8).reshape(2, 2)}])
    _component(root, "wan_low", [{"blocks.0.weight": torch.arange(4, dtype=torch.int8).reshape(2, 2) + 1}])
    (root / "mllm" / "config.json").write_text('{"hidden_size":4}', encoding="utf-8")
    (root / "mllm" / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")
    (root / "mllm" / "tokenizer_config.json").write_text('{"model_max_length":8}', encoding="utf-8")
    (root / "t5_tokenizer").mkdir()
    (root / "t5_tokenizer" / "spiece.model").write_bytes(b"spiece-fixture")
    components = ("mllm", "connector", "vit_decoder", "mask_tokens", "t5_text_encoder", "wan_high", "wan_low")
    (root / "repack-manifest.json").write_text(
        json.dumps({"source_revision": "fixture", "outputs": {name: {} for name in components}}),
        encoding="utf-8",
    )
    return root


def test_export_single_files_preserves_and_prefixes_tensors(tmp_path):
    source = _fixture(tmp_path)
    output = tmp_path / "single"
    report = export_single_files(source, output, profile="int8")

    assert report["format"] == "bernini_v2_comfyui_single_files"
    assert set(report["outputs"]) == {"planner", "t5", "wan_high", "wan_low"}
    planner = output / "bernini_v2_planner_int8.safetensors"
    with safe_open(planner, framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == {
            "model.embed_tokens.weight",
            "visual.patch_embed.proj.weight",
            "connector.fc1.weight",
            "vit_decoder.blocks.0.weight",
            "mask_tokens",
            "config_json",
            "tokenizer_json",
            "tokenizer_config",
        }
        assert torch.equal(handle.get_tensor("connector.fc1.weight"), torch.ones(2, 2, dtype=torch.bfloat16))
        assert bytes(handle.get_tensor("tokenizer_json").tolist()) == b'{"version":"1.0"}'

    with safe_open(output / "umt5_xxl_bernini_v2_int8.safetensors", framework="pt", device="cpu") as handle:
        assert bytes(handle.get_tensor("spiece_model").tolist()) == b"spiece-fixture"
    with safe_open(output / "bernini_v2_high_noise_int8.safetensors", framework="pt", device="cpu") as handle:
        assert torch.equal(handle.get_tensor("blocks.0.weight"), torch.arange(4, dtype=torch.int8).reshape(2, 2))


def test_export_is_atomic_and_refuses_overwrite(tmp_path):
    source = _fixture(tmp_path)
    output = tmp_path / "single"
    export_single_files(source, output, profile="bf16")
    try:
        export_single_files(source, output, profile="bf16")
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing standalone models must not be overwritten implicitly")
    assert not list(output.glob("*.partial"))


def test_header_reader_rejects_non_safetensors(tmp_path):
    invalid = tmp_path / "bad.safetensors"
    invalid.write_bytes(b"short")
    try:
        read_safetensors_header(invalid)
    except ValueError as error:
        assert "truncated" in str(error)
    else:
        raise AssertionError("invalid safetensors file must be rejected")
