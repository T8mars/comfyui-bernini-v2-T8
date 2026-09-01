#!/usr/bin/env python3
"""Convert a sharded native Wan checkpoint to a base GGUF without a full-model RAM load.

The tensor precision policy and the post-quantization 5D-tensor workflow are
compatible with City96/ComfyUI-GGUF.  This entry point is specialized for the
native Wan renderer layout used by Bernini v2.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_TENSOR_NAME_LENGTH = 127
MAX_TENSOR_DIMS = 4
QUANTIZATION_THRESHOLD = 1024
WAN_DETECTION_KEYS = frozenset(
    {
        "blocks.0.self_attn.norm_q.weight",
        "text_embedding.2.weight",
        "head.modulation",
    }
)
WAN_HIGH_PRECISION_FRAGMENTS = (".modulation",)


@dataclass(frozen=True)
class TensorDescriptor:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def elements(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    @property
    def source_bytes(self) -> int:
        bytes_per_element = {"BF16": 2, "F16": 2, "F32": 4}.get(self.dtype)
        if bytes_per_element is None:
            raise ValueError(f"unsupported source dtype {self.dtype!r} for {self.name}")
        return self.elements * bytes_per_element


def _load_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        import gguf
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError("conversion requires torch, safetensors, and the gguf package from ComfyUI-GGUF") from error
    return gguf, torch, safe_open, save_file


def resolve_index(path: Path) -> Path:
    path = path.resolve()
    if path.is_dir():
        path = path / "model.safetensors.index.json"
    if not path.is_file():
        raise FileNotFoundError(f"shard index not found: {path}")
    return path


def load_weight_map(index_path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    document = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid or empty weight_map in {index_path}")
    if not all(isinstance(name, str) and isinstance(shard, str) for name, shard in weight_map.items()):
        raise ValueError(f"weight_map must contain string tensor and shard names: {index_path}")

    missing_shards = sorted({shard for shard in weight_map.values() if not (index_path.parent / shard).is_file()})
    if missing_shards:
        raise FileNotFoundError(f"missing source shards: {', '.join(missing_shards)}")
    return weight_map, document


def inspect_sharded_checkpoint(index_path: Path) -> tuple[list[TensorDescriptor], dict[str, Any]]:
    _, _, safe_open, _ = _load_runtime()
    weight_map, document = load_weight_map(index_path)
    expected_by_shard: dict[str, set[str]] = defaultdict(set)
    for name, shard in weight_map.items():
        expected_by_shard[shard].add(name)

    descriptors: list[TensorDescriptor] = []
    for shard_name in sorted(expected_by_shard):
        shard_path = index_path.parent / shard_name
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            actual = set(handle.keys())
            expected = expected_by_shard[shard_name]
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            if missing or unexpected:
                raise ValueError(
                    f"index/header mismatch in {shard_name}: missing={missing[:5]}, unexpected={unexpected[:5]}"
                )
            for name in sorted(expected):
                tensor_slice = handle.get_slice(name)
                descriptors.append(
                    TensorDescriptor(
                        name=name,
                        shard=shard_name,
                        dtype=str(tensor_slice.get_dtype()),
                        shape=tuple(int(value) for value in tensor_slice.get_shape()),
                    )
                )

    names = {descriptor.name for descriptor in descriptors}
    missing_detection_keys = sorted(WAN_DETECTION_KEYS - names)
    if missing_detection_keys:
        raise ValueError(f"checkpoint is not a native Wan renderer; missing keys: {missing_detection_keys}")

    long_names = sorted((item.name, len(item.name)) for item in descriptors if len(item.name) > MAX_TENSOR_NAME_LENGTH)
    if long_names:
        raise ValueError(f"GGUF tensor-name limit exceeded: {long_names[:5]}")

    dtype_counts = Counter(item.dtype for item in descriptors)
    dimension_counts = Counter(len(item.shape) for item in descriptors)
    source_bytes = sum(item.source_bytes for item in descriptors)
    metadata_size = document.get("metadata", {}).get("total_size")
    if metadata_size is not None and int(metadata_size) != source_bytes:
        raise ValueError(f"index total_size is {metadata_size}, but tensor headers total {source_bytes}")

    report = {
        "index": str(index_path),
        "tensors": len(descriptors),
        "shards": len(expected_by_shard),
        "source_bytes": source_bytes,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "dimension_counts": {str(key): value for key, value in sorted(dimension_counts.items())},
        "largest_tensor": max(
            (
                {"name": item.name, "shape": item.shape, "dtype": item.dtype, "source_bytes": item.source_bytes}
                for item in descriptors
            ),
            key=lambda item: item["source_bytes"],
        ),
        "five_dimensional_tensors": [item.name for item in descriptors if len(item.shape) > MAX_TENSOR_DIMS],
    }
    return descriptors, report


def target_qtype_name(descriptor: TensorDescriptor) -> str:
    if descriptor.dtype == "BF16":
        result = "BF16"
    else:
        result = "F16"
    if descriptor.dtype in {"BF16", "F32"}:
        if len(descriptor.shape) == 1:
            result = "F32"
        elif descriptor.elements <= QUANTIZATION_THRESHOLD:
            result = "F32"
        elif any(fragment in descriptor.name for fragment in WAN_HIGH_PRECISION_FRAGMENTS):
            result = "F32"
    return result


def iter_tensors(index_path: Path, descriptors: list[TensorDescriptor]) -> Iterator[tuple[TensorDescriptor, Any]]:
    _, _, safe_open, _ = _load_runtime()
    descriptors_by_shard: dict[str, list[TensorDescriptor]] = defaultdict(list)
    for descriptor in descriptors:
        descriptors_by_shard[descriptor.shard].append(descriptor)
    for shard_name in sorted(descriptors_by_shard):
        with safe_open(index_path.parent / shard_name, framework="pt", device="cpu") as handle:
            for descriptor in descriptors_by_shard[shard_name]:
                yield descriptor, handle.get_tensor(descriptor.name)


def _to_numpy(tensor: Any, source_dtype: str, torch: Any) -> Any:
    if source_dtype == "BF16":
        return tensor.to(torch.float32).numpy()
    float8_types = {
        value for value in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)) if value
    }
    if tensor.dtype in float8_types:
        return tensor.to(torch.float16).numpy()
    return tensor.numpy()


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def convert_sharded_checkpoint(
    source: Path,
    destination: Path,
    *,
    fix_path: Path | None = None,
    temp_root: Path | None = None,
    overwrite: bool = False,
    verify_hash: bool = False,
) -> dict[str, Any]:
    gguf, torch, _, save_file = _load_runtime()
    index_path = resolve_index(source)
    destination = destination.resolve()
    fix_path = (fix_path or destination.with_name(f"{destination.stem}-5d.safetensors")).resolve()
    temp_root = (temp_root or destination.parent).resolve()
    temp_root.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)

    for path in (destination, fix_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"output exists; pass --overwrite to replace it: {path}")

    descriptors, report = inspect_sharded_checkpoint(index_path)
    dtype_counts = Counter(item.dtype for item in descriptors)
    main_dtype = dtype_counts.most_common(1)[0][0]
    if main_dtype == "BF16":
        file_type = gguf.LlamaFileType.MOSTLY_BF16
    else:
        file_type = gguf.LlamaFileType.MOSTLY_F16

    output_partial = destination.with_name(f"{destination.name}.partial")
    fix_partial = fix_path.with_name(f"{fix_path.name}.partial")
    for partial in (output_partial, fix_partial):
        partial.unlink(missing_ok=True)

    writer = None
    saved_tempdir = tempfile.tempdir
    quantization_counts: Counter[str] = Counter()
    five_dimensional: dict[str, Any] = {}
    try:
        with tempfile.TemporaryDirectory(prefix="bernini-gguf-", dir=temp_root) as run_temp:
            tempfile.tempdir = run_temp
            writer = gguf.GGUFWriter(path=None, arch="wan", use_temp_file=True)
            writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
            writer.add_file_type(file_type)

            total = len(descriptors)
            for position, (descriptor, tensor) in enumerate(iter_tensors(index_path, descriptors), start=1):
                array = _to_numpy(tensor, descriptor.dtype, torch)
                if len(descriptor.shape) > MAX_TENSOR_DIMS:
                    five_dimensional[descriptor.name] = torch.from_numpy(array.copy())
                    quantization_counts["5D_FIX"] += 1
                else:
                    qtype_name = target_qtype_name(descriptor)
                    qtype = getattr(gguf.GGMLQuantizationType, qtype_name)
                    try:
                        quantized = gguf.quants.quantize(array, qtype)
                    except (AttributeError, gguf.QuantError) as error:
                        print(
                            f"[{position}/{total}] {descriptor.name}: {qtype_name} failed ({error}); using F16",
                            file=sys.stderr,
                        )
                        qtype_name = "F16"
                        qtype = gguf.GGMLQuantizationType.F16
                        quantized = gguf.quants.quantize(array, qtype)
                    writer.add_tensor(descriptor.name, quantized, raw_dtype=qtype)
                    quantization_counts[qtype_name] += 1
                if position == 1 or position % 25 == 0 or position == total:
                    print(f"[{position}/{total}] {descriptor.name}", flush=True)
                del tensor, array
                if "quantized" in locals():
                    del quantized
                if position % 25 == 0:
                    gc.collect()

            if not five_dimensional:
                raise ValueError("expected Wan patch_embedding 5D tensor was not found")
            save_file(five_dimensional, fix_partial)

            writer.write_header_to_file(path=output_partial)
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file(progress=True)
            writer.close()
            writer = None
            os.replace(fix_partial, fix_path)
            os.replace(output_partial, destination)
    except BaseException:
        if writer is not None:
            if writer.temp_file is not None:
                writer.temp_file.close()
            writer.close()
        output_partial.unlink(missing_ok=True)
        fix_partial.unlink(missing_ok=True)
        raise
    finally:
        tempfile.tempdir = saved_tempdir

    reader = gguf.GGUFReader(str(destination))
    try:
        output_tensor_count = len(reader.tensors)
    finally:
        del reader
    expected_output_tensors = len(descriptors) - len(five_dimensional)
    if output_tensor_count != expected_output_tensors:
        raise ValueError(f"GGUF contains {output_tensor_count} tensors; expected {expected_output_tensors}")

    report.update(
        {
            "destination": str(destination),
            "destination_bytes": destination.stat().st_size,
            "destination_sha256": _sha256(destination) if verify_hash else None,
            "fix_path": str(fix_path),
            "fix_bytes": fix_path.stat().st_size,
            "gguf_tensors": output_tensor_count,
            "quantization_counts": dict(sorted(quantization_counts.items())),
        }
    )
    report_path = destination.with_name(f"{destination.name}.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Wan directory or model.safetensors.index.json")
    parser.add_argument("--dst", type=Path, help="Base BF16/F16 GGUF output")
    parser.add_argument("--fix", type=Path, help="5D safetensors output for post-quantization repair")
    parser.add_argument("--temp-dir", type=Path, help="Directory for the disk-backed GGUF spool")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-hash", action="store_true", help="SHA-256 the large output after conversion")
    parser.add_argument("--inspect", action="store_true", help="Validate headers and print a report without converting")
    args = parser.parse_args()

    index_path = resolve_index(args.src)
    if args.inspect:
        _, report = inspect_sharded_checkpoint(index_path)
    else:
        if args.dst is None:
            parser.error("--dst is required unless --inspect is used")
        report = convert_sharded_checkpoint(
            index_path,
            args.dst,
            fix_path=args.fix,
            temp_root=args.temp_dir,
            overwrite=args.overwrite,
            verify_hash=args.verify_hash,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
