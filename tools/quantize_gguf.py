#!/usr/bin/env python3
"""Quantize a base Wan GGUF, restore its 5D tensor, and validate the final file."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

QTYPE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _load_runtime() -> tuple[Any, Any, Any]:
    try:
        import gguf
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError("GGUF finalization requires gguf and safetensors") from error
    return gguf, load_file, __import__("numpy")


def normalize_qtype(value: str) -> str:
    value = value.strip().upper()
    if not QTYPE_PATTERN.fullmatch(value):
        raise ValueError(f"invalid GGUF quantization type: {value!r}")
    return value


def _field_value(field: Any, numpy: Any) -> Any:
    value = field.parts[field.data[-1]]
    array = numpy.asarray(value)
    return array.item() if array.size == 1 else value


def _architecture(reader: Any, numpy: Any) -> str:
    value = _field_value(reader.get_field("general.architecture"), numpy)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    array = numpy.asarray(value)
    if array.ndim == 1 and array.dtype.kind in {"i", "u"}:
        return bytes(array.tolist()).decode("utf-8")
    return str(value)


def _file_type(reader: Any, gguf: Any, numpy: Any) -> Any:
    return gguf.LlamaFileType(int(_field_value(reader.get_field("general.file_type"), numpy)))


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def restore_5d_tensor(
    source: Path,
    destination: Path,
    fix_path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Copy a quantized GGUF and append the float32 Wan Conv3D tensor."""
    gguf, load_file, numpy = _load_runtime()
    source = source.resolve()
    destination = destination.resolve()
    fix_path = fix_path.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"quantized GGUF not found: {source}")
    if not fix_path.is_file():
        raise FileNotFoundError(f"5D fix file not found: {fix_path}")
    if source == destination:
        raise ValueError("source and destination GGUF paths must differ")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace it: {destination}")

    partial = destination.with_name(f"{destination.name}.partial")
    partial.unlink(missing_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reader = gguf.GGUFReader(str(source))
    writer = None
    try:
        arch = _architecture(reader, numpy)
        if arch != "wan":
            raise ValueError(f"expected Wan GGUF, got architecture {arch!r}")
        file_type = _file_type(reader, gguf, numpy)
        existing_names = {tensor.name for tensor in reader.tensors}
        fixed = load_file(fix_path)
        duplicates = sorted(existing_names & set(fixed))
        if duplicates:
            raise ValueError(f"5D tensors already exist in source GGUF: {duplicates}")
        if not fixed or any(tensor.ndim <= 4 for tensor in fixed.values()):
            raise ValueError("fix file must contain only 5D-or-higher tensors")

        writer = gguf.GGUFWriter(path=None, arch=arch)
        writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
        writer.add_file_type(file_type)
        for tensor in reader.tensors:
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
        for name, tensor in fixed.items():
            array = tensor.float().numpy()
            writer.add_tensor(
                name,
                gguf.quants.quantize(array, gguf.GGMLQuantizationType.F32),
                raw_dtype=gguf.GGMLQuantizationType.F32,
            )

        writer.write_header_to_file(path=partial)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()
        writer = None
        reader = None
        gc.collect()
        os.replace(partial, destination)
    except BaseException:
        if writer is not None:
            writer.close()
        reader = None
        gc.collect()
        partial.unlink(missing_ok=True)
        raise

    final_reader = gguf.GGUFReader(str(destination))
    try:
        type_counts = Counter(tensor.tensor_type.name for tensor in final_reader.tensors)
        output_names = {tensor.name for tensor in final_reader.tensors}
        missing_fixed = sorted(set(fixed) - output_names)
        if missing_fixed:
            raise ValueError(f"final GGUF is missing restored tensors: {missing_fixed}")
        tensor_count = len(final_reader.tensors)
    finally:
        del final_reader
    return {
        "architecture": arch,
        "file_type": file_type.name,
        "tensors": tensor_count,
        "restored_5d": sorted(fixed),
        "tensor_types": dict(sorted(type_counts.items())),
    }


def quantize_gguf(
    source: Path,
    destination: Path,
    fix_path: Path,
    quantizer: Path,
    *,
    qtype: str = "Q4_K_S",
    overwrite: bool = False,
    keep_intermediate: bool = False,
    reuse_intermediate: bool = False,
    verify_hash: bool = False,
) -> dict[str, Any]:
    source = source.resolve()
    destination = destination.resolve()
    fix_path = fix_path.resolve()
    quantizer = quantizer.resolve()
    qtype = normalize_qtype(qtype)
    for path, label in ((source, "base GGUF"), (fix_path, "5D fix file"), (quantizer, "llama-quantize")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace it: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    intermediate = destination.with_name(f"{destination.stem}-no5d.gguf.partial")
    if reuse_intermediate:
        if not intermediate.is_file():
            raise FileNotFoundError(f"quantized intermediate not found: {intermediate}")
    else:
        intermediate.unlink(missing_ok=True)
    try:
        if not reuse_intermediate:
            subprocess.run([str(quantizer), str(source), str(intermediate), qtype], check=True)
        report = restore_5d_tensor(intermediate, destination, fix_path, overwrite=overwrite)
    except BaseException:
        print(f"quantized intermediate preserved for recovery: {intermediate}", flush=True)
        raise
    if not keep_intermediate:
        intermediate.unlink(missing_ok=True)

    report.update(
        {
            "source": str(source),
            "destination": str(destination),
            "destination_bytes": destination.stat().st_size,
            "destination_sha256": _sha256(destination) if verify_hash else None,
            "quantizer": str(quantizer),
            "reused_intermediate": reuse_intermediate,
            "requested_qtype": qtype,
        }
    )
    report_path = destination.with_name(f"{destination.name}.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Base BF16/F16 GGUF without the 5D tensor")
    parser.add_argument("--dst", type=Path, required=True, help="Final quantized GGUF")
    parser.add_argument("--fix", type=Path, required=True, help="5D safetensors emitted by convert_sharded_gguf.py")
    parser.add_argument("--quantizer", type=Path, required=True, help="Patched llama-quantize executable")
    parser.add_argument("--type", default="Q4_K_S", dest="qtype")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument(
        "--reuse-intermediate", action="store_true", help="Skip quantization and finalize an existing no5d partial"
    )
    parser.add_argument("--verify-hash", action="store_true")
    args = parser.parse_args()
    report = quantize_gguf(
        args.src,
        args.dst,
        args.fix,
        args.quantizer,
        qtype=args.qtype,
        overwrite=args.overwrite,
        keep_intermediate=args.keep_intermediate,
        reuse_intermediate=args.reuse_intermediate,
        verify_hash=args.verify_hash,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
