#!/usr/bin/env python3
"""Validate Bernini sharded indexes and, optionally, manifest hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
FLOAT_DTYPES = {"F8_E4M3", "F8_E5M2", "F16", "BF16", "F32", "F64"}


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_index(index_path: Path, *, expected_float_dtype: str | None = None) -> dict[str, Any]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid or empty weight_map: {index_path}")
    by_shard: dict[str, set[str]] = defaultdict(set)
    for key, shard in weight_map.items():
        if not isinstance(key, str) or not isinstance(shard, str):
            raise ValueError(f"non-string weight map entry: {index_path}")
        by_shard[shard].add(key)

    total_size = 0
    dtype_counts: dict[str, int] = defaultdict(int)
    tensor_info: dict[str, tuple[str, list[int]]] = {}
    quant_markers: dict[str, dict[str, Any]] = {}
    for shard, expected_keys in sorted(by_shard.items()):
        shard_path = (index_path.parent / shard).resolve()
        if shard_path.parent != index_path.parent.resolve():
            raise ValueError(f"shard escapes index directory: {shard}")
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            actual_keys = set(handle.keys())
            if actual_keys != expected_keys:
                missing = sorted(expected_keys - actual_keys)[:8]
                extra = sorted(actual_keys - expected_keys)[:8]
                raise ValueError(f"{shard_path}: missing={missing}, extra={extra}")
            for key in actual_keys:
                tensor = handle.get_slice(key)
                dtype = tensor.get_dtype()
                shape = tensor.get_shape()
                if dtype not in DTYPE_BYTES:
                    raise ValueError(f"unsupported safetensors dtype {dtype!r} in {shard_path}")
                total_size += math.prod(shape) * DTYPE_BYTES[dtype]
                dtype_counts[dtype] += 1
                tensor_info[key] = (dtype, shape)
                if expected_float_dtype is not None and dtype in FLOAT_DTYPES and not key.endswith(".weight_scale"):
                    if dtype not in {expected_float_dtype, "F8_E4M3", "F8_E5M2"}:
                        raise ValueError(f"{key}: expected {expected_float_dtype} passthrough storage, found {dtype}")
                if key.endswith(".comfy_quant"):
                    marker = handle.get_tensor(key)
                    if marker.dtype != torch.uint8 or marker.ndim != 1:
                        raise ValueError(f"{key}: comfy_quant marker must be one-dimensional uint8")
                    try:
                        quant_markers[key.removesuffix(".comfy_quant")] = json.loads(marker.numpy().tobytes())
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise ValueError(f"{key}: invalid comfy_quant JSON") from error

    for base, marker in quant_markers.items():
        if marker.get("format") != "int8_tensorwise":
            raise ValueError(f"{base}: unsupported comfy_quant format {marker.get('format')!r}")
        weight = tensor_info.get(f"{base}.weight")
        scale = tensor_info.get(f"{base}.weight_scale")
        if weight is None or scale is None:
            raise ValueError(f"{base}: INT8 layer is missing weight or weight_scale")
        if weight[0] != "I8":
            raise ValueError(f"{base}.weight: expected I8, found {weight[0]}")
        if scale[0] != "F32" or scale[1] != [weight[1][0], 1]:
            raise ValueError(f"{base}.weight_scale: expected F32 [{weight[1][0]}, 1], found {scale}")
        if marker.get("convrot"):
            group_size = int(marker.get("convrot_groupsize", 0))
            if group_size not in {16, 64, 256} or weight[1][1] % group_size:
                raise ValueError(f"{base}: invalid ConvRot group size {group_size} for shape {weight[1]}")

    declared_size = int(payload.get("metadata", {}).get("total_size", 0))
    if declared_size and declared_size != total_size:
        raise ValueError(f"{index_path}: declared total_size={declared_size}, actual={total_size}")
    return {
        "tensors": len(weight_map),
        "shards": len(by_shard),
        "total_size": total_size,
        "dtypes": dict(sorted(dtype_counts.items())),
        "quantized_layers": len(quant_markers),
    }


def validate_repack(root: Path, *, verify_hashes: bool = False) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "repack-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    storage_dtype = manifest.get("storage_dtype", "preserve")
    expected_float_dtype = {"bfloat16": "BF16", "float16": "F16", "preserve": None}.get(storage_dtype)
    if storage_dtype not in {"bfloat16", "float16", "preserve"}:
        raise ValueError(f"unsupported manifest storage_dtype: {storage_dtype!r}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError(f"invalid or empty outputs in {manifest_path}")

    report = {}
    for component, expected in sorted(outputs.items()):
        index_report = validate_index(
            root / component / "model.safetensors.index.json",
            expected_float_dtype=expected_float_dtype,
        )
        if index_report["tensors"] != int(expected["tensors"]):
            raise ValueError(f"{component}: tensor count does not match manifest")
        if index_report["total_size"] != int(expected["total_size"]):
            raise ValueError(f"{component}: tensor bytes do not match manifest")
        if expected.get("dtypes") is not None and index_report["dtypes"] != expected["dtypes"]:
            raise ValueError(f"{component}: dtype counts do not match manifest")
        if expected.get("quantized_layers") is not None:
            if index_report["quantized_layers"] != int(expected["quantized_layers"]):
                raise ValueError(f"{component}: quantized layer count does not match manifest")
        if verify_hashes:
            for shard, digest in expected["sha256"].items():
                actual = file_sha256(root / component / shard)
                if actual != digest:
                    raise ValueError(f"{component}/{shard}: sha256 mismatch")
        report[component] = index_report
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate_repack(args.root, verify_hashes=args.verify_hashes), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
