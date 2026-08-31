#!/usr/bin/env python3
"""Stream the combined Bernini checkpoint into component safetensor shards.

Each source shard is opened once. Tensors are grouped by runtime component and
written immediately, so peak host memory is bounded by one source shard rather
than the roughly 180 GB combined checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bernini_v2.index import load_index, summarize  # noqa: E402
from bernini_v2.qwen import qwen_checkpoint_to_comfy  # noqa: E402
from bernini_v2.state_dict import Component, component_key  # noqa: E402
from bernini_v2.wan_mapping import wan_diffusers_to_comfy  # noqa: E402

INDEX_RELATIVE = Path("bernini") / "model.safetensors.index.json"
SOURCE_RELATIVE = Path("bernini")
METADATA_FILES = (
    Path("config.json"),
    Path("mllm") / "chat_template.json",
    Path("mllm") / "config.json",
    Path("mllm") / "merges.txt",
    Path("mllm") / "preprocessor_config.json",
    Path("mllm") / "tokenizer.json",
    Path("mllm") / "tokenizer_config.json",
    Path("mllm") / "vocab.json",
    Path("t5_tokenizer") / "special_tokens_map.json",
    Path("t5_tokenizer") / "spiece.model",
    Path("t5_tokenizer") / "tokenizer_config.json",
)
MANIFEST_SCHEMA_VERSION = 2
PROGRESS_FILE = ".repack-progress.json"
STORAGE_DTYPES = {
    "preserve": None,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
SAFETENSOR_DTYPE_BYTES = {
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
FLOAT_SAFETENSOR_DTYPES = {"F8_E4M3", "F8_E5M2", "F16", "BF16", "F32", "F64"}


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    """Write metadata without leaving a valid-looking partial JSON file."""

    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _copy_file_atomic(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def convert_storage_tensor(tensor: torch.Tensor, storage_dtype: str) -> torch.Tensor:
    """Downcast floating checkpoint tensors while preserving integer metadata."""

    try:
        target_dtype = STORAGE_DTYPES[storage_dtype]
    except KeyError as error:
        raise ValueError(f"unsupported storage dtype: {storage_dtype}") from error
    if target_dtype is None or not tensor.is_floating_point():
        return tensor
    return tensor.to(dtype=target_dtype)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _shard_report(path: Path, expected_keys: set[str], *, storage_dtype: str) -> dict[str, object]:
    """Validate a completed shard before accepting it as resumable output."""

    total_size = 0
    dtype_counts: dict[str, int] = defaultdict(int)
    with safe_open(path, framework="pt", device="cpu") as handle:
        actual_keys = set(handle.keys())
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)[:8]
            extra = sorted(actual_keys - expected_keys)[:8]
            raise ValueError(f"{path}: resumable shard key mismatch; missing={missing}, extra={extra}")
        for key in sorted(actual_keys):
            tensor = handle.get_slice(key)
            dtype = tensor.get_dtype()
            shape = tensor.get_shape()
            if storage_dtype != "preserve" and dtype in FLOAT_SAFETENSOR_DTYPES and dtype not in {"F8_E4M3", "F8_E5M2"}:
                expected_dtype = "BF16" if storage_dtype == "bfloat16" else "F16"
                if dtype != expected_dtype:
                    raise ValueError(f"{path}:{key} has {dtype}, expected {expected_dtype}")
            if dtype not in SAFETENSOR_DTYPE_BYTES:
                raise ValueError(f"{path}:{key} has unsupported dtype {dtype}")
            total_size += math.prod(shape) * SAFETENSOR_DTYPE_BYTES[dtype]
            dtype_counts[dtype] += 1
    return {
        "total_size": total_size,
        "dtypes": dict(sorted(dtype_counts.items())),
        "sha256": file_sha256(path),
    }


def native_target_key(component: Component, source_key: str, *, wan_format: str) -> str | None:
    _, target_key = component_key(source_key)
    if component in (Component.WAN_HIGH, Component.WAN_LOW) and wan_format == "comfy":
        return wan_diffusers_to_comfy(target_key)
    if component is Component.MLLM:
        return qwen_checkpoint_to_comfy(target_key)
    return target_key


def repack(
    source: Path,
    output: Path,
    *,
    dry_run: bool = False,
    wan_format: str = "comfy",
    storage_dtype: str = "bfloat16",
    resume: bool = True,
    source_revision: str | None = None,
) -> dict[str, object]:
    if storage_dtype not in STORAGE_DTYPES:
        raise ValueError(f"unsupported storage dtype: {storage_dtype}")
    plan = load_index(source / INDEX_RELATIVE)
    report: dict[str, object] = summarize(plan)
    report["schema_version"] = MANIFEST_SCHEMA_VERSION
    report["format"] = "bernini_v2_safetensors_sharded"
    report["wan_format"] = wan_format
    report["storage_dtype"] = storage_dtype
    report["source_index_sha256"] = file_sha256(source / INDEX_RELATIVE)
    report["source_revision"] = source_revision
    report["excluded"] = {
        source_key: "unused Qwen language-model head"
        for source_key in plan.weight_map
        if native_target_key(component_key(source_key)[0], source_key, wan_format=wan_format) is None
    }
    report["outputs"] = {}
    # Validate every renderer name before touching a multi-gigabyte shard.
    renderer_targets: dict[Component, set[str]] = defaultdict(set)
    for source_key in plan.weight_map:
        component, _ = component_key(source_key)
        if component in (Component.WAN_HIGH, Component.WAN_LOW):
            target_key = native_target_key(component, source_key, wan_format=wan_format)
            if target_key is None:
                continue
            if target_key in renderer_targets[component]:
                raise ValueError(f"duplicate renderer target key {component.value}:{target_key}")
            renderer_targets[component].add(target_key)
    if dry_run:
        return report

    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / PROGRESS_FILE
    progress_identity = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_index_sha256": report["source_index_sha256"],
        "wan_format": wan_format,
        "storage_dtype": storage_dtype,
        "source_revision": source_revision,
    }
    progress: dict[str, object] = {**progress_identity, "completed": {}}
    if resume and progress_path.is_file():
        existing_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        mismatches = {
            key: (existing_progress.get(key), value)
            for key, value in progress_identity.items()
            if existing_progress.get(key) != value
        }
        if mismatches:
            raise ValueError(f"cannot resume repack with different inputs/options: {mismatches}")
        progress = existing_progress
    elif not resume:
        _write_json_atomic(progress_path, progress)
    copied_metadata = []
    for relative_path in METADATA_FILES:
        source_metadata = source / relative_path
        if not source_metadata.is_file():
            continue
            target_metadata = output / relative_path
            target_metadata.parent.mkdir(parents=True, exist_ok=True)
            _copy_file_atomic(source_metadata, target_metadata)
        copied_metadata.append(relative_path.as_posix())
    report["metadata_files"] = copied_metadata
    output_indexes: dict[Component, dict[str, str]] = defaultdict(dict)
    output_sizes: dict[Component, int] = defaultdict(int)
    output_hashes: dict[Component, dict[str, str]] = defaultdict(dict)
    output_dtypes: dict[Component, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    ordered_shards = sorted(plan.by_shard)
    shard_count = len(ordered_shards)
    for source_number, source_name in enumerate(ordered_shards, start=1):
        print(
            f"[{source_number:02d}/{shard_count:02d}] reading {source_name}",
            file=sys.stderr,
            flush=True,
        )
        selected: dict[Component, list[str]] = defaultdict(list)
        for source_key in plan.by_shard[source_name]:
            component, _ = component_key(source_key)
            if native_target_key(component, source_key, wan_format=wan_format) is None:
                continue
            selected[component].append(source_key)

        source_path = source / SOURCE_RELATIVE / source_name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        with safe_open(source_path, framework="pt", device="cpu") as handle:
            for component, source_keys in sorted(selected.items(), key=lambda item: item[0].value):
                component_dir = output / component.value
                component_dir.mkdir(parents=True, exist_ok=True)
                output_name = f"model-{source_number:05d}-of-{shard_count:05d}.safetensors"
                output_path = component_dir / output_name
                target_keys = {
                    target_key
                    for source_key in source_keys
                    if (target_key := native_target_key(component, source_key, wan_format=wan_format)) is not None
                }
                progress_key = f"{component.value}/{output_name}"
                completed = progress.setdefault("completed", {})
                can_resume = resume and progress_key in completed and output_path.is_file()
                if can_resume:
                    shard_info = _shard_report(output_path, target_keys, storage_dtype=storage_dtype)
                    if shard_info["sha256"] != completed[progress_key].get("sha256"):
                        raise ValueError(f"{output_path}: resumable shard hash mismatch")
                    print(f"  resumed {progress_key}", file=sys.stderr, flush=True)
                else:
                    tensors = {}
                    for source_key in source_keys:
                        target_key = native_target_key(component, source_key, wan_format=wan_format)
                        if target_key is None:
                            continue
                        tensors[target_key] = convert_storage_tensor(handle.get_tensor(source_key), storage_dtype)

                    temporary = output_path.with_name(f".{output_name}.tmp-{os.getpid()}")
                    save_file(
                        tensors,
                        temporary,
                        metadata={
                            "format": "pt",
                            "source": source_name,
                            "storage_dtype": storage_dtype,
                        },
                    )
                    os.replace(temporary, output_path)
                    shard_info = _shard_report(output_path, target_keys, storage_dtype=storage_dtype)
                    completed[progress_key] = shard_info
                    _write_json_atomic(progress_path, progress)

                output_hashes[component][output_name] = str(shard_info["sha256"])
                output_sizes[component] += int(shard_info["total_size"])
                for dtype_name, count in shard_info["dtypes"].items():
                    output_dtypes[component][dtype_name] += int(count)
                for target_key in target_keys:
                    output_indexes[component][target_key] = output_name
                print(
                    f"  wrote {component.value}/{output_name} "
                    f"({len(target_keys)} tensors, {output_path.stat().st_size / 2**30:.2f} GiB)",
                    file=sys.stderr,
                    flush=True,
                )

    outputs: dict[str, object] = {}
    for component in sorted(output_indexes, key=lambda item: item.value):
        component_dir = output / component.value
        payload = {
            "metadata": {"total_size": output_sizes[component]},
            "weight_map": dict(sorted(output_indexes[component].items())),
        }
        _write_json_atomic(component_dir / "model.safetensors.index.json", payload)
        outputs[component.value] = {
            "tensors": len(output_indexes[component]),
            "total_size": output_sizes[component],
            "dtypes": dict(sorted(output_dtypes[component].items())),
            "sha256": dict(sorted(output_hashes[component].items())),
        }

    report["outputs"] = outputs
    _write_json_atomic(output / "repack-manifest.json", report)
    progress["finished"] = True
    _write_json_atomic(progress_path, progress)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("repacked") / "bernini-v2-bf16")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wan-format", choices=("comfy", "diffusers"), default="comfy")
    parser.add_argument("--storage-dtype", choices=tuple(STORAGE_DTYPES), default="bfloat16")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source-revision")
    args = parser.parse_args()
    report = repack(
        args.source.resolve(),
        args.output.resolve(),
        dry_run=args.dry_run,
        wan_format=args.wan_format,
        storage_dtype=args.storage_dtype,
        resume=args.resume,
        source_revision=args.source_revision,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
