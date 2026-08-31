"""Cheap structural inspection for Bernini v2 Wan renderer checkpoints."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from safetensors import safe_open

EXPECTED_BLOCKS = tuple(range(40))
REQUIRED_RENDERER_KEYS = {
    "patch_embedding.weight",
    "time_embedding.0.weight",
    "text_embedding.0.weight",
    "blocks.0.self_attn.q.weight",
    "blocks.39.self_attn.q.weight",
    "head.head.weight",
}
SUPPORTED_QUANT_FORMATS = {
    "int8_tensorwise",
    "nvfp4",
    "mxfp8",
    "fp8_e4m3fn",
    "fp8_e5m2",
}
_BLOCK_PATTERN = re.compile(r"^blocks\.(\d+)\.")
_PREFIXES = ("model.diffusion_model.", "diffusion_model.")
_DTYPE_BYTES = {
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


def normalize_renderer_key(key: str) -> str:
    """Remove the optional prefixes used by stock Comfy diffusion checkpoints."""

    for prefix in _PREFIXES:
        if key.startswith(prefix):
            return key.removeprefix(prefix)
    return key


def _checkpoint_files(path: Path) -> dict[Path, tuple[str, ...] | None]:
    if path.name.endswith(".safetensors.index.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid or empty weight_map in {path}")
        by_shard: dict[Path, list[str]] = defaultdict(list)
        for key, shard_name in weight_map.items():
            if not isinstance(key, str) or not isinstance(shard_name, str):
                raise ValueError(f"non-string weight_map entry in {path}")
            shard_path = (path.parent / shard_name).resolve()
            if shard_path.parent != path.parent.resolve():
                raise ValueError(f"shard escapes checkpoint directory: {shard_name}")
            by_shard[shard_path].append(key)
        return {shard: tuple(sorted(keys)) for shard, keys in sorted(by_shard.items())}
    if path.suffix == ".safetensors":
        return {path: None}
    raise ValueError(f"expected .safetensors or .safetensors.index.json, got {path}")


def inspect_renderer_checkpoint(path: str | Path) -> dict[str, Any]:
    """Validate one Wan expert without materializing its large weight tensors."""

    checkpoint = Path(path).resolve()
    files = _checkpoint_files(checkpoint)
    raw_keys: set[str] = set()
    normalized_keys: set[str] = set()
    dtypes: Counter[str] = Counter()
    quant_formats: Counter[str] = Counter()
    total_size = 0
    marker_count = 0
    legacy_scale_weights = 0
    legacy_scaled_fp8 = False

    for shard_path, expected_keys in files.items():
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            keys = available if expected_keys is None else set(expected_keys)
            missing = keys - available
            if missing:
                raise ValueError(f"{shard_path} is missing indexed tensors: {sorted(missing)[:8]}")
            for key in keys:
                if key in raw_keys:
                    raise ValueError(f"duplicate tensor across shards: {key}")
                raw_keys.add(key)
                normalized = normalize_renderer_key(key)
                if normalized in normalized_keys:
                    raise ValueError(f"duplicate normalized renderer key: {normalized}")
                normalized_keys.add(normalized)
                tensor_slice = handle.get_slice(key)
                dtype = str(tensor_slice.get_dtype())
                shape = tuple(tensor_slice.get_shape())
                if dtype not in _DTYPE_BYTES:
                    raise ValueError(f"unsupported safetensors dtype {dtype!r} for {key}")
                dtypes[dtype] += 1
                total_size += math.prod(shape) * _DTYPE_BYTES[dtype]
                if normalized == "scaled_fp8":
                    legacy_scaled_fp8 = True
                elif normalized.endswith(".scale_weight"):
                    legacy_scale_weights += 1
                if normalized.endswith(".comfy_quant"):
                    marker = handle.get_tensor(key)
                    try:
                        payload = json.loads(marker.numpy().tobytes())
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise ValueError(f"invalid comfy_quant marker: {key}") from error
                    quant_format = payload.get("format")
                    if quant_format not in SUPPORTED_QUANT_FORMATS:
                        raise ValueError(f"unsupported quantization format {quant_format!r}: {key}")
                    base = normalized.removesuffix(".comfy_quant")
                    if f"{base}.weight" not in normalized_keys and normalize_renderer_key(
                        key.removesuffix(".comfy_quant") + ".weight"
                    ) not in {normalize_renderer_key(item) for item in available}:
                        raise ValueError(f"quantization marker has no weight tensor: {key}")
                    quant_formats[str(quant_format)] += 1
                    marker_count += 1

    if legacy_scaled_fp8:
        if marker_count:
            raise ValueError("checkpoint mixes legacy scaled_fp8 and comfy_quant metadata")
        if legacy_scale_weights < 1:
            raise ValueError("legacy scaled_fp8 checkpoint contains no per-layer weight scales")
        quant_formats["legacy_scaled_fp8"] = legacy_scale_weights
        marker_count = legacy_scale_weights

    missing_required = REQUIRED_RENDERER_KEYS - normalized_keys
    if missing_required:
        raise ValueError(f"checkpoint is not a complete Bernini v2 Wan renderer; missing {sorted(missing_required)}")
    blocks = sorted(
        {int(match.group(1)) for key in normalized_keys if (match := _BLOCK_PATTERN.match(key)) is not None}
    )
    if blocks != list(EXPECTED_BLOCKS):
        raise ValueError(f"expected renderer blocks 0..39, found {blocks}")
    prefix = "model.diffusion_model." if any(key.startswith("model.diffusion_model.") for key in raw_keys) else ""
    return {
        "path": str(checkpoint),
        "files": len(files),
        "tensors": len(raw_keys),
        "tensor_bytes": total_size,
        "tensor_gib": round(total_size / 2**30, 3),
        "dtypes": dict(sorted(dtypes.items())),
        "quantized_layers": marker_count,
        "quant_formats": dict(sorted(quant_formats.items())),
        "renderer_blocks": len(blocks),
        "key_prefix": prefix,
        "normalized_keys": sorted(normalized_keys),
    }


def inspect_renderer_pair(high: str | Path, low: str | Path) -> dict[str, Any]:
    """Inspect both experts and require an identical normalized tensor contract."""

    high_report = inspect_renderer_checkpoint(high)
    low_report = inspect_renderer_checkpoint(low)
    high_keys = set(high_report.pop("normalized_keys"))
    low_keys = set(low_report.pop("normalized_keys"))
    if high_keys != low_keys:
        only_high = sorted(high_keys - low_keys)[:8]
        only_low = sorted(low_keys - high_keys)[:8]
        raise ValueError(f"renderer expert key mismatch: only_high={only_high}, only_low={only_low}")
    if high_report["quant_formats"] != low_report["quant_formats"]:
        raise ValueError(
            "renderer expert quantization mismatch: "
            f"high={high_report['quant_formats']}, low={low_report['quant_formats']}"
        )
    return {"high": high_report, "low": low_report, "pair_keys_match": True}
