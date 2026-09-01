#!/usr/bin/env python3
"""Validate a Bernini v2 high/low GGUF renderer pair from GGUF metadata."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.quantize_gguf import _architecture, _file_type, _load_runtime  # noqa: E402

REQUIRED_KEYS = {
    "blocks.0.self_attn.norm_q.weight",
    "text_embedding.2.weight",
    "head.modulation",
    "patch_embedding.weight",
}


def validate_contracts(
    high: dict[str, tuple[tuple[int, ...], str]],
    low: dict[str, tuple[tuple[int, ...], str]],
) -> None:
    high_names = set(high)
    low_names = set(low)
    if high_names != low_names:
        raise ValueError(
            f"GGUF tensor names differ: high_only={sorted(high_names - low_names)[:5]}, "
            f"low_only={sorted(low_names - high_names)[:5]}"
        )
    mismatches = [name for name in sorted(high) if high[name] != low[name]]
    if mismatches:
        details = {name: {"high": high[name], "low": low[name]} for name in mismatches[:5]}
        raise ValueError(f"GGUF tensor shape/type contracts differ: {details}")


def inspect_gguf(path: Path) -> tuple[dict[str, Any], dict[str, tuple[tuple[int, ...], str]]]:
    gguf, _, numpy = _load_runtime()
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GGUF not found: {path}")
    reader = gguf.GGUFReader(str(path))
    try:
        arch = _architecture(reader, numpy)
        if arch != "wan":
            raise ValueError(f"expected Wan GGUF, got architecture {arch!r}: {path}")
        file_type = _file_type(reader, gguf, numpy).name
        contract = {
            tensor.name: (tuple(int(value) for value in tensor.shape), tensor.tensor_type.name)
            for tensor in reader.tensors
        }
    finally:
        reader = None
        gc.collect()

    missing = sorted(REQUIRED_KEYS - set(contract))
    if missing:
        raise ValueError(f"missing required Bernini/Wan tensors: {missing}")
    blocks = sorted({int(name.split(".")[1]) for name in contract if name.startswith("blocks.")})
    if blocks != list(range(40)):
        raise ValueError(f"expected Wan blocks 0..39, got {blocks}")
    patch_shape, patch_type = contract["patch_embedding.weight"]
    if len(patch_shape) != 5 or patch_type != "F32":
        raise ValueError(f"invalid restored patch_embedding.weight: shape={patch_shape}, type={patch_type}")
    type_counts = Counter(item[1] for item in contract.values())
    if not any(name.startswith("Q") for name in type_counts):
        raise ValueError("GGUF contains no quantized tensors")

    serialized = json.dumps(
        sorted((name, shape, dtype) for name, (shape, dtype) in contract.items()), separators=(",", ":")
    )
    report = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "architecture": arch,
        "file_type": file_type,
        "tensors": len(contract),
        "blocks": len(blocks),
        "tensor_types": dict(sorted(type_counts.items())),
        "contract_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "patch_embedding": {"shape": patch_shape, "type": patch_type},
    }
    return report, contract


def validate_pair(high_path: Path, low_path: Path) -> dict[str, Any]:
    high_report, high_contract = inspect_gguf(high_path)
    low_report, low_contract = inspect_gguf(low_path)
    validate_contracts(high_contract, low_contract)
    return {
        "status": "ok",
        "same_contract": True,
        "contract_sha256": high_report["contract_sha256"],
        "high": high_report,
        "low": low_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--high", type=Path, required=True)
    parser.add_argument("--low", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_pair(args.high, args.low), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
