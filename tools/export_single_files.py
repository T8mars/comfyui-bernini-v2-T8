#!/usr/bin/env python3
"""Export a Bernini v2 component repack as four standalone ComfyUI models.

The input may be the existing sharded BF16 or stock-Comfy quantized conversion
workspace.  The output is the user-facing format: one planner, one UMT5, and
one file for each Wan expert.  Tensor payloads are copied byte-for-byte from
their source safetensors files, so peak memory is bounded by a small copy
buffer rather than model size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
from collections import Counter
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HEADER_ALIGNMENT = 8
COPY_BUFFER_SIZE = 16 * 1024 * 1024
PROFILE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
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


@dataclass(frozen=True)
class TensorSource:
    target_name: str
    dtype: str
    shape: tuple[int, ...]
    source_path: Path | None
    source_offset: int
    byte_length: int
    literal: bytes | None = None


@dataclass(frozen=True)
class SafeTensorFile:
    path: Path
    data_start: int
    tensors: dict[str, dict[str, Any]]
    metadata: dict[str, str]


def _sha256(path: Path, chunk_size: int = COPY_BUFFER_SIZE) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_bytes(dtype: str, shape: tuple[int, ...]) -> int:
    try:
        size = DTYPE_BYTES[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported safetensors dtype: {dtype}") from error
    elements = 1
    for dimension in shape:
        if dimension < 0:
            raise ValueError(f"negative tensor dimension: {shape}")
        elements *= dimension
    return elements * size


def read_safetensors_header(path: Path) -> SafeTensorFile:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"truncated safetensors header length: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length <= 0 or 8 + header_length > file_size:
            raise ValueError(f"invalid safetensors header length {header_length}: {path}")
        try:
            header = json.loads(handle.read(header_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid safetensors JSON header: {path}") from error

    raw_metadata = header.get("__metadata__", {})
    if not isinstance(raw_metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in raw_metadata.items()
    ):
        raise ValueError(f"invalid safetensors metadata in {path}")
    tensors: dict[str, dict[str, Any]] = {}
    ranges: list[tuple[int, int, str]] = []
    data_size = file_size - 8 - header_length
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise ValueError(f"invalid tensor header entry in {path}: {name!r}")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or not isinstance(offsets, list)
            or len(offsets) != 2
        ):
            raise ValueError(f"invalid tensor descriptor for {name!r}: {path}")
        tensor_shape = tuple(int(value) for value in shape)
        start, end = (int(offsets[0]), int(offsets[1]))
        if start < 0 or end < start or end > data_size:
            raise ValueError(f"invalid data offsets for {name!r}: {path}")
        expected = _tensor_bytes(dtype, tensor_shape)
        if end - start != expected:
            raise ValueError(f"tensor byte length mismatch for {name!r}: expected {expected}, got {end - start}")
        tensors[name] = {"dtype": dtype, "shape": tensor_shape, "start": start, "end": end}
        ranges.append((start, end, name))

    for previous, current in zip(sorted(ranges), sorted(ranges)[1:], strict=False):
        if current[0] < previous[1]:
            raise ValueError(f"overlapping tensor payloads {previous[2]!r}/{current[2]!r}: {path}")
    return SafeTensorFile(
        path=path,
        data_start=8 + header_length,
        tensors=tensors,
        metadata=dict(raw_metadata),
    )


def component_sources(
    root: Path,
    component: str,
    rename: Callable[[str], str] = lambda name: name,
) -> list[TensorSource]:
    component_dir = root.resolve() / component
    index_path = component_dir / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid or empty weight_map: {index_path}")
    else:
        candidates = sorted(component_dir.glob("*.safetensors"))
        if len(candidates) != 1:
            raise ValueError(f"expected one safetensors file or an index in {component_dir}")
        header = read_safetensors_header(candidates[0])
        weight_map = {name: candidates[0].name for name in header.tensors}

    header_cache: dict[Path, SafeTensorFile] = {}
    sources: list[TensorSource] = []
    target_names: set[str] = set()
    for source_name, relative_file in sorted(weight_map.items()):
        if not isinstance(source_name, str) or not isinstance(relative_file, str):
            raise ValueError(f"non-string weight-map entry: {index_path}")
        source_path = (component_dir / relative_file).resolve()
        if source_path.parent != component_dir.resolve():
            raise ValueError(f"component shard escapes its directory: {relative_file}")
        header = header_cache.get(source_path)
        if header is None:
            header = read_safetensors_header(source_path)
            header_cache[source_path] = header
        try:
            descriptor = header.tensors[source_name]
        except KeyError as error:
            raise ValueError(f"{relative_file} is missing indexed tensor {source_name!r}") from error
        target_name = rename(source_name)
        if target_name in target_names:
            raise ValueError(f"duplicate target tensor name: {target_name}")
        target_names.add(target_name)
        sources.append(
            TensorSource(
                target_name=target_name,
                dtype=descriptor["dtype"],
                shape=descriptor["shape"],
                source_path=source_path,
                source_offset=header.data_start + descriptor["start"],
                byte_length=descriptor["end"] - descriptor["start"],
            )
        )
    return sources


def byte_tensor(name: str, payload: bytes) -> TensorSource:
    return TensorSource(
        target_name=name,
        dtype="U8",
        shape=(len(payload),),
        source_path=None,
        source_offset=0,
        byte_length=len(payload),
        literal=payload,
    )


def _source_revision(source: Path, manifest: dict[str, Any]) -> str:
    revision = manifest.get("source_revision")
    if isinstance(revision, str) and revision:
        return revision
    parent_manifest = manifest.get("source_manifest")
    if isinstance(parent_manifest, str) and parent_manifest:
        parent_path = Path(parent_manifest)
        if not parent_path.is_absolute():
            parent_path = source / parent_path
        if parent_path.is_file():
            parent = json.loads(parent_path.read_text(encoding="utf-8"))
            revision = parent.get("source_revision")
            if isinstance(revision, str) and revision:
                return revision
    return "unknown"


def _encode_header(sources: list[TensorSource], metadata: dict[str, str]) -> bytes:
    header: dict[str, Any] = {"__metadata__": metadata}
    offset = 0
    for source in sources:
        header[source.target_name] = {
            "dtype": source.dtype,
            "shape": list(source.shape),
            "data_offsets": [offset, offset + source.byte_length],
        }
        offset += source.byte_length
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    padding = (-len(raw)) % HEADER_ALIGNMENT
    return raw + (b" " * padding)


def write_standalone_model(
    destination: Path,
    sources: list[TensorSource],
    *,
    metadata: dict[str, str],
    overwrite: bool = False,
) -> dict[str, Any]:
    destination = destination.resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace it: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(sources, key=lambda item: item.target_name)
    names = [item.target_name for item in ordered]
    if len(names) != len(set(names)):
        raise ValueError("standalone model contains duplicate tensor names")
    header = _encode_header(ordered, metadata)
    partial = destination.with_name(f"{destination.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        with ExitStack() as stack, partial.open("wb") as output:
            handles: dict[Path, Any] = {}
            output.write(struct.pack("<Q", len(header)))
            output.write(header)
            for position, source in enumerate(ordered, start=1):
                if source.literal is not None:
                    output.write(source.literal)
                else:
                    assert source.source_path is not None
                    handle = handles.get(source.source_path)
                    if handle is None:
                        handle = stack.enter_context(source.source_path.open("rb"))
                        handles[source.source_path] = handle
                    handle.seek(source.source_offset)
                    remaining = source.byte_length
                    while remaining:
                        chunk = handle.read(min(COPY_BUFFER_SIZE, remaining))
                        if not chunk:
                            raise ValueError(f"truncated tensor payload while copying {source.target_name!r}")
                        output.write(chunk)
                        remaining -= len(chunk)
                if position == 1 or position % 100 == 0 or position == len(ordered):
                    print(f"[{position}/{len(ordered)}] {destination.name}: {source.target_name}", flush=True)
            output.flush()
            os.fsync(output.fileno())
        os.replace(partial, destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    verified = read_safetensors_header(destination)
    if set(verified.tensors) != set(names):
        raise ValueError(f"standalone output tensor contract mismatch: {destination}")
    type_counts = Counter(item.dtype for item in ordered)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
        "tensors": len(ordered),
        "dtypes": dict(sorted(type_counts.items())),
    }


def export_single_files(
    source: Path,
    output: Path,
    *,
    profile: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if not PROFILE_PATTERN.fullmatch(profile):
        raise ValueError(f"invalid profile name: {profile!r}")
    source_manifest_path = source / "repack-manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_revision = _source_revision(source, source_manifest)

    planner_sources = component_sources(source, "mllm")
    planner_sources += component_sources(source, "connector", lambda name: f"connector.{name}")
    planner_sources += component_sources(source, "vit_decoder", lambda name: f"vit_decoder.{name}")
    planner_sources += component_sources(source, "mask_tokens")
    model_config = source / "mllm" / "config.json"
    tokenizer_json = source / "mllm" / "tokenizer.json"
    tokenizer_config = source / "mllm" / "tokenizer_config.json"
    for path in (model_config, tokenizer_json, tokenizer_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    planner_sources += [
        byte_tensor("config_json", model_config.read_bytes()),
        byte_tensor("tokenizer_json", tokenizer_json.read_bytes()),
        byte_tensor("tokenizer_config", tokenizer_config.read_bytes()),
    ]

    t5_sources = component_sources(source, "t5_text_encoder")
    spiece = source / "t5_tokenizer" / "spiece.model"
    if not spiece.is_file():
        raise FileNotFoundError(spiece)
    t5_sources.append(byte_tensor("spiece_model", spiece.read_bytes()))

    jobs = {
        "planner": (f"bernini_v2_planner_{profile}.safetensors", planner_sources),
        "t5": (f"umt5_xxl_bernini_v2_{profile}.safetensors", t5_sources),
        "wan_high": (
            f"bernini_v2_high_noise_{profile}.safetensors",
            component_sources(source, "wan_high"),
        ),
        "wan_low": (
            f"bernini_v2_low_noise_{profile}.safetensors",
            component_sources(source, "wan_low"),
        ),
    }
    report: dict[str, Any] = {
        "format": "bernini_v2_comfyui_single_files",
        "schema_version": 1,
        "profile": profile,
        "source": str(source),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "source_revision": source_revision,
        "outputs": {},
    }
    for component, (filename, sources) in jobs.items():
        report["outputs"][component] = write_standalone_model(
            output / filename,
            sources,
            metadata={
                "format": "pt",
                "architecture": "bernini_v2",
                "component": component,
                "profile": profile,
                "source_revision": source_revision,
            },
            overwrite=overwrite,
        )
    manifest_path = output / f"bernini_v2_{profile}_manifest.json"
    temporary = manifest_path.with_name(f"{manifest_path.name}.partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="BF16 or quantized component repack")
    parser.add_argument("--output", type=Path, required=True, help="Directory for four standalone models")
    parser.add_argument("--profile", required=True, help="Filename suffix, for example int8 or bf16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            export_single_files(args.source, args.output, profile=args.profile, overwrite=args.overwrite),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
