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


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_index(index_path: Path) -> dict[str, Any]:
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
                if dtype not in DTYPE_BYTES:
                    raise ValueError(f"unsupported safetensors dtype {dtype!r} in {shard_path}")
                total_size += math.prod(tensor.get_shape()) * DTYPE_BYTES[dtype]

    declared_size = int(payload.get("metadata", {}).get("total_size", 0))
    if declared_size and declared_size != total_size:
        raise ValueError(f"{index_path}: declared total_size={declared_size}, actual={total_size}")
    return {"tensors": len(weight_map), "shards": len(by_shard), "total_size": total_size}


def validate_repack(root: Path, *, verify_hashes: bool = False) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "repack-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError(f"invalid or empty outputs in {manifest_path}")

    report = {}
    for component, expected in sorted(outputs.items()):
        index_report = validate_index(root / component / "model.safetensors.index.json")
        if index_report["tensors"] != int(expected["tensors"]):
            raise ValueError(f"{component}: tensor count does not match manifest")
        if index_report["total_size"] != int(expected["total_size"]):
            raise ValueError(f"{component}: tensor bytes do not match manifest")
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
