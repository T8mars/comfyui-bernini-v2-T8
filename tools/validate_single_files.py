#!/usr/bin/env python3
"""Validate the four Core-compatible Bernini v2 standalone model files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bernini_v2.checkpoint_info import SUPPORTED_QUANT_FORMATS, inspect_renderer_pair  # noqa: E402
from bernini_v2.runtime import _build_qwen_tokenizer  # noqa: E402
from tools.export_single_files import SafeTensorFile, read_safetensors_header  # noqa: E402


def _single_header(path: Path, component: str) -> SafeTensorFile:
    checkpoint = path.resolve()
    if checkpoint.suffix != ".safetensors":
        raise ValueError(f"{component} must be one .safetensors file: {checkpoint}")
    header = read_safetensors_header(checkpoint)
    if header.metadata.get("architecture") != "bernini_v2":
        raise ValueError(f"{component} has invalid architecture metadata: {header.metadata}")
    if header.metadata.get("component") != component:
        raise ValueError(f"{component} has invalid component metadata: {header.metadata}")
    return header


def _tensor_bytes(handle, header: SafeTensorFile, key: str) -> bytes:
    try:
        descriptor = header.tensors[key]
    except KeyError as error:
        raise ValueError(f"{header.path.name} is missing embedded tensor {key!r}") from error
    if descriptor["dtype"] != "U8" or len(descriptor["shape"]) != 1:
        raise ValueError(f"embedded tensor {key!r} must be one-dimensional U8")
    tensor = handle.get_tensor(key)
    if tensor.dtype != torch.uint8:
        raise AssertionError(f"safetensors dtype mismatch for {key!r}")
    return tensor.numpy().tobytes()


def _json_object(payload: bytes, key: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"embedded tensor {key!r} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"embedded tensor {key!r} must contain a JSON object")
    return value


def _quant_formats(path: Path, keys: set[str]) -> dict[str, int]:
    formats: Counter[str] = Counter()
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in sorted(item for item in keys if item.endswith(".comfy_quant")):
            try:
                marker = json.loads(handle.get_tensor(key).numpy().tobytes())
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid comfy_quant marker: {key}") from error
            quant_format = marker.get("format")
            if quant_format not in SUPPORTED_QUANT_FORMATS:
                raise ValueError(f"unsupported quantization format {quant_format!r}: {key}")
            base = key.removesuffix(".comfy_quant")
            if f"{base}.weight" not in keys:
                raise ValueError(f"quantization marker has no weight tensor: {key}")
            formats[str(quant_format)] += 1
    return dict(sorted(formats.items()))


def inspect_planner(path: Path) -> dict[str, Any]:
    header = _single_header(path, "planner")
    keys = set(header.tensors)
    required = {
        "model.embed_tokens.weight",
        "visual.patch_embed.proj.weight",
        "connector.proj_gen.0.weight",
        "vit_decoder.net.input_proj.weight",
        "mask_tokens",
        "config_json",
        "tokenizer_json",
        "tokenizer_config",
    }
    missing = required - keys
    if missing:
        raise ValueError(f"planner is incomplete; missing {sorted(missing)}")
    if not any(key.startswith("model.layers.27.") for key in keys):
        raise ValueError("planner does not contain all 28 Qwen language layers")
    if not any(key.startswith("visual.blocks.31.") for key in keys):
        raise ValueError("planner does not contain all 32 Qwen vision layers")
    with safe_open(header.path, framework="pt", device="cpu") as handle:
        config_payload = _tensor_bytes(handle, header, "config_json")
        tokenizer_payload = _tensor_bytes(handle, header, "tokenizer_json")
        tokenizer_config_payload = _tensor_bytes(handle, header, "tokenizer_config")
    config = _json_object(config_payload, "config_json")
    if config.get("model_type") != "qwen2_5_vl" or config.get("hidden_size") != 3584:
        raise ValueError("planner contains an incompatible Qwen model config")
    tokenizer_config = _json_object(tokenizer_config_payload, "tokenizer_config")
    tokenizer = _build_qwen_tokenizer(tokenizer_payload, tokenizer_config)
    expected_ids = {
        "<|vision_start|>": 151652,
        "<|vision_end|>": 151653,
        "<|image_pad|>": 151655,
        "<|video_pad|>": 151656,
    }
    actual_ids = {token: tokenizer.convert_tokens_to_ids(token) for token in expected_ids}
    if actual_ids != expected_ids:
        raise ValueError(f"planner tokenizer special-token mismatch: {actual_ids}")
    return {
        "path": str(header.path),
        "bytes": header.path.stat().st_size,
        "tensors": len(keys),
        "profile": header.metadata.get("profile"),
        "quant_formats": _quant_formats(header.path, keys),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab": len(tokenizer),
    }


def inspect_t5(path: Path) -> dict[str, Any]:
    header = _single_header(path, "t5")
    keys = set(header.tensors)
    required = {
        "shared.weight",
        "encoder.embed_tokens.weight",
        "encoder.final_layer_norm.weight",
        "spiece_model",
    }
    missing = required - keys
    if missing:
        raise ValueError(f"UMT5 is incomplete; missing {sorted(missing)}")
    if not any(key.startswith("encoder.block.23.") for key in keys):
        raise ValueError("UMT5 does not contain all 24 encoder blocks")
    with safe_open(header.path, framework="pt", device="cpu") as handle:
        spiece = _tensor_bytes(handle, header, "spiece_model")
    if len(spiece) < 1_000_000:
        raise ValueError("embedded UMT5 SentencePiece model is unexpectedly small")
    return {
        "path": str(header.path),
        "bytes": header.path.stat().st_size,
        "tensors": len(keys),
        "profile": header.metadata.get("profile"),
        "quant_formats": _quant_formats(header.path, keys),
        "spiece_bytes": len(spiece),
    }


def inspect_single_file_set(planner: Path, t5: Path, high: Path, low: Path) -> dict[str, Any]:
    planner_report = inspect_planner(planner)
    t5_report = inspect_t5(t5)
    profiles = {planner_report["profile"], t5_report["profile"]}
    high_header = _single_header(high, "wan_high")
    low_header = _single_header(low, "wan_low")
    profiles.update((high_header.metadata.get("profile"), low_header.metadata.get("profile")))
    if len(profiles) != 1:
        raise ValueError(f"standalone files mix profiles: {sorted(str(item) for item in profiles)}")
    renderer_report = inspect_renderer_pair(high, low)
    if renderer_report["high"]["files"] != 1 or renderer_report["low"]["files"] != 1:
        raise ValueError("renderer experts must each be one standalone file")
    return {
        "format": "bernini_v2_comfyui_single_files",
        "profile": next(iter(profiles)),
        "planner": planner_report,
        "t5": t5_report,
        "renderers": renderer_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--t5", type=Path, required=True)
    parser.add_argument("--high", type=Path, required=True)
    parser.add_argument("--low", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_single_file_set(args.planner, args.t5, args.high, args.low)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
