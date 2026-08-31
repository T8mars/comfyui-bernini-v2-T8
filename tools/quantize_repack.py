#!/usr/bin/env python3
"""Convert a Bernini v2 repack to stock-Comfy INT8 ConvRot shards.

The converter keeps quality-sensitive and tensor-core-inefficient layers in
BF16, quantizes one source shard at a time, and writes resumable atomic output.
It intentionally uses comfy-kitchen's public layout API so the artifact follows
the same storage contract as stock ComfyUI.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bernini_v2.quantization import (  # noqa: E402
    QUANT_COMPONENT_PROFILES,
    classify_int8_weight,
    comfy_quant_marker,
)
from tools.repack_diffusers import file_sha256  # noqa: E402

SCHEMA_VERSION = 2
PROGRESS_FILE = ".quantize-progress.json"


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _dtype_name(tensor: torch.Tensor) -> str:
    names = {
        torch.bfloat16: "BF16",
        torch.float16: "F16",
        torch.float32: "F32",
        torch.float64: "F64",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.int16: "I16",
        torch.int32: "I32",
        torch.int64: "I64",
        torch.bool: "BOOL",
    }
    try:
        return names[tensor.dtype]
    except KeyError as error:
        raise ValueError(f"unsupported output dtype: {tensor.dtype}") from error


def quantizer_runtime_fingerprint(device: str) -> dict[str, str | None]:
    """Capture recipe-affecting runtime versions so resumed shards cannot silently mix math."""

    try:
        kitchen_version = importlib.metadata.version("comfy-kitchen")
    except importlib.metadata.PackageNotFoundError:
        kitchen_version = None
    target = torch.device(device)
    device_name = None
    if target.type == "cuda" and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(target)
    return {
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "comfy_kitchen": kitchen_version,
        "device": str(target),
        "device_name": device_name,
    }


def _component_shards(source: Path, component: str) -> dict[str, tuple[str, ...]]:
    component_dir = source / component
    index_path = component_dir / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        by_shard: dict[str, list[str]] = defaultdict(list)
        for key, shard in payload.get("weight_map", {}).items():
            by_shard[shard].append(key)
        if not by_shard:
            raise ValueError(f"empty component index: {index_path}")
        return {name: tuple(sorted(keys)) for name, keys in sorted(by_shard.items())}
    single = component_dir / "model.safetensors"
    if not single.is_file():
        candidates = sorted(component_dir.glob("*.safetensors"))
        if len(candidates) != 1:
            raise FileNotFoundError(index_path)
        single = candidates[0]
    with safe_open(single, framework="pt", device="cpu") as handle:
        return {single.name: tuple(sorted(handle.keys()))}


def _quantize_weight(
    weight: torch.Tensor,
    *,
    group_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    try:
        import comfy_kitchen
        from comfy_kitchen.tensor import TensorWiseINT8Layout
    except ImportError as error:
        raise RuntimeError("INT8 ConvRot conversion requires comfy-kitchen in this Python environment") from error

    # New comfy-kitchen wheels can contain a CUDA 13 extension even when the
    # active PyTorch/runtime is older. Directly selecting that extension fails
    # with an insufficient-driver error; stock ComfyUI applies the same gate.
    if torch.version.cuda is not None:
        cuda_version = tuple(int(part) for part in torch.version.cuda.split(".")[:2])
        if cuda_version < (13, 0):
            comfy_kitchen.registry.disable("cuda")

    reference = weight.to(device=device, dtype=torch.bfloat16)
    quantized, params = TensorWiseINT8Layout.quantize(
        reference,
        is_weight=True,
        per_channel=True,
        convrot=True,
        convrot_groupsize=group_size,
    )
    reconstructed = TensorWiseINT8Layout.dequantize(quantized, params).float()
    reference_float = reference.float()
    cosine = float(F.cosine_similarity(reconstructed.flatten(), reference_float.flatten(), dim=0).item())
    relative_error = float(
        ((reconstructed - reference_float).norm() / reference_float.norm().clamp_min(1e-30)).item() * 100.0
    )
    return quantized.cpu(), params.scale.float().cpu(), cosine, relative_error


def _inspect_plan(
    source: Path,
    components: list[str],
    *,
    profile: str,
    min_gemm: int,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    parameters = 0
    layers = 0
    for component in components:
        for shard_name, keys in _component_shards(source, component).items():
            with safe_open(source / component / shard_name, framework="pt", device="cpu") as handle:
                if any(key.endswith(".comfy_quant") for key in keys):
                    raise ValueError(f"{component}/{shard_name} is already quantized")
                for key in keys:
                    shape = tuple(handle.get_slice(key).get_shape())
                    decision = classify_int8_weight(component, key, shape, profile=profile, min_gemm=min_gemm)
                    counts[decision.reason] += 1
                    if decision.quantize:
                        layers += 1
                        parameters += math.prod(shape)
    return {
        "profile": profile,
        "components": sorted(QUANT_COMPONENT_PROFILES[profile]),
        "quantized_layers": layers,
        "quantized_parameters": parameters,
        "decisions": dict(sorted(counts.items())),
    }


def quantize_repack(
    source: Path,
    output: Path,
    *,
    profile: str = "balanced",
    min_gemm: int = 256,
    max_relerr: float = 2.0,
    min_cosine: float = 0.99,
    device: str = "cuda",
    resume: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if profile not in QUANT_COMPONENT_PROFILES:
        raise ValueError(f"unknown quantization profile: {profile}")
    source_manifest_path = source / "repack-manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_outputs = source_manifest.get("outputs")
    if not isinstance(source_outputs, dict) or not source_outputs:
        raise ValueError(f"invalid source manifest: {source_manifest_path}")
    components = sorted(source_outputs)
    plan = _inspect_plan(source, components, profile=profile, min_gemm=min_gemm)
    if dry_run:
        return plan

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for quantization but is not available")
    identity = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "profile": profile,
        "min_gemm": min_gemm,
        "max_relerr": max_relerr,
        "min_cosine": min_cosine,
        "quantizer_runtime": quantizer_runtime_fingerprint(device),
    }
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / PROGRESS_FILE
    progress: dict[str, Any] = {**identity, "completed": {}}
    if resume and progress_path.is_file():
        existing = json.loads(progress_path.read_text(encoding="utf-8"))
        mismatches = {key: (existing.get(key), value) for key, value in identity.items() if existing.get(key) != value}
        if mismatches:
            raise ValueError(f"cannot resume quantization with different inputs/options: {mismatches}")
        progress = existing

    copied_metadata = []
    for relative_name in source_manifest.get("metadata_files", []):
        relative = Path(relative_name)
        source_file = source / relative
        if source_file.is_file():
            target_file = output / relative
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target_file)
            copied_metadata.append(relative.as_posix())

    output_summaries: dict[str, Any] = {}
    all_layer_reports: dict[str, Any] = {}
    total_quality_fallbacks = 0
    for component in components:
        component_dir = output / component
        component_dir.mkdir(parents=True, exist_ok=True)
        source_shards = _component_shards(source, component)
        weight_map: dict[str, str] = {}
        component_hashes: dict[str, str] = {}
        component_dtypes: Counter[str] = Counter()
        component_size = 0
        component_quantized = 0
        component_fallbacks = 0
        for shard_number, (source_name, source_keys) in enumerate(source_shards.items(), start=1):
            output_name = f"model-{shard_number:05d}-of-{len(source_shards):05d}.safetensors"
            output_path = component_dir / output_name
            progress_key = f"{component}/{output_name}"
            completed = progress["completed"]
            entry = completed.get(progress_key)
            if resume and entry is not None and output_path.is_file():
                digest = file_sha256(output_path)
                if digest != entry["sha256"]:
                    raise ValueError(f"{output_path}: resumable shard hash mismatch")
                shard_keys = entry["keys"]
                shard_size = int(entry["total_size"])
                shard_dtypes = entry["dtypes"]
                layer_reports = entry.get("layers", {})
                shard_quantized = int(entry.get("quantized_layers", 0))
                shard_fallbacks = int(entry.get("quality_fallbacks", 0))
                print(f"resumed {progress_key}", file=sys.stderr, flush=True)
            else:
                tensors: dict[str, torch.Tensor] = {}
                layer_reports = {}
                shard_quantized = 0
                shard_fallbacks = 0
                with safe_open(source / component / source_name, framework="pt", device="cpu") as handle:
                    for key in source_keys:
                        tensor = handle.get_tensor(key)
                        decision = classify_int8_weight(
                            component,
                            key,
                            tuple(tensor.shape),
                            profile=profile,
                            min_gemm=min_gemm,
                        )
                        if decision.quantize:
                            quantized, scale, cosine, relerr = _quantize_weight(
                                tensor,
                                group_size=int(decision.group_size),
                                device=target_device,
                            )
                            layer_name = key.removesuffix(".weight")
                            report = {
                                "cosine": cosine,
                                "relative_error_percent": relerr,
                                "group_size": decision.group_size,
                            }
                            if cosine >= min_cosine and relerr <= max_relerr:
                                tensors[key] = quantized
                                tensors[f"{layer_name}.weight_scale"] = scale
                                tensors[f"{layer_name}.comfy_quant"] = comfy_quant_marker(int(decision.group_size))
                                report["status"] = "quantized"
                                shard_quantized += 1
                            else:
                                tensors[key] = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
                                report["status"] = "bf16-quality-fallback"
                                shard_fallbacks += 1
                            layer_reports[f"{component}.{layer_name}"] = report
                        else:
                            tensors[key] = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor

                temporary = output_path.with_name(f".{output_name}.tmp-{os.getpid()}")
                save_file(
                    tensors,
                    temporary,
                    metadata={
                        "format": "pt",
                        "quantization": "int8_tensorwise_convrot",
                        "profile": profile,
                        "source": source_name,
                    },
                )
                os.replace(temporary, output_path)
                shard_keys = sorted(tensors)
                shard_size = sum(_nbytes(tensor) for tensor in tensors.values())
                shard_dtypes = dict(sorted(Counter(_dtype_name(tensor) for tensor in tensors.values()).items()))
                digest = file_sha256(output_path)
                entry = {
                    "sha256": digest,
                    "keys": shard_keys,
                    "total_size": shard_size,
                    "dtypes": shard_dtypes,
                    "layers": layer_reports,
                    "quantized_layers": shard_quantized,
                    "quality_fallbacks": shard_fallbacks,
                }
                completed[progress_key] = entry
                _write_json_atomic(progress_path, progress)
                del tensors
                if target_device.type == "cuda":
                    torch.cuda.empty_cache()

            for key in shard_keys:
                if key in weight_map:
                    raise ValueError(f"duplicate output key {component}:{key}")
                weight_map[key] = output_name
            component_hashes[output_name] = digest
            component_size += shard_size
            component_dtypes.update(shard_dtypes)
            component_quantized += shard_quantized
            component_fallbacks += shard_fallbacks
            all_layer_reports.update(layer_reports)

        index_payload = {
            "metadata": {"total_size": component_size},
            "weight_map": dict(sorted(weight_map.items())),
        }
        _write_json_atomic(component_dir / "model.safetensors.index.json", index_payload)
        output_summaries[component] = {
            "tensors": len(weight_map),
            "total_size": component_size,
            "dtypes": dict(sorted(component_dtypes.items())),
            "sha256": dict(sorted(component_hashes.items())),
            "quantized_layers": component_quantized,
            "quality_fallbacks": component_fallbacks,
        }
        total_quality_fallbacks += component_fallbacks

    report = {
        "schema_version": 3,
        "format": "bernini_v2_int8_tensorwise_convrot",
        "storage_dtype": "bfloat16",
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": identity["source_manifest_sha256"],
        "metadata_files": copied_metadata,
        "quantization": {
            **plan,
            "min_gemm": min_gemm,
            "max_relerr": max_relerr,
            "min_cosine": min_cosine,
            "quality_fallbacks": total_quality_fallbacks,
            "marker_format": "int8_tensorwise",
            "convrot": True,
            "runtime": identity["quantizer_runtime"],
        },
        "outputs": output_summaries,
    }
    _write_json_atomic(output / "quantization-report.json", all_layer_reports)
    _write_json_atomic(output / "repack-manifest.json", report)
    progress["finished"] = True
    _write_json_atomic(progress_path, progress)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(QUANT_COMPONENT_PROFILES), default="balanced")
    parser.add_argument("--min-gemm", type=int, default=256)
    parser.add_argument("--max-relerr", type=float, default=2.0)
    parser.add_argument("--min-cosine", type=float, default=0.99)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = quantize_repack(
        args.source,
        args.output,
        profile=args.profile,
        min_gemm=args.min_gemm,
        max_relerr=args.max_relerr,
        min_cosine=args.min_cosine,
        device=args.device,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
