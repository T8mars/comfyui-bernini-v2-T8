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
import shutil
import sys
from collections import defaultdict
from pathlib import Path

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


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


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
) -> dict[str, object]:
    plan = load_index(source / INDEX_RELATIVE)
    report: dict[str, object] = summarize(plan)
    report["wan_format"] = wan_format
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
    copied_metadata = []
    for relative_path in METADATA_FILES:
        source_metadata = source / relative_path
        if not source_metadata.is_file():
            continue
        target_metadata = output / relative_path
        target_metadata.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_metadata, target_metadata)
        copied_metadata.append(relative_path.as_posix())
    report["metadata_files"] = copied_metadata
    output_indexes: dict[Component, dict[str, str]] = defaultdict(dict)
    output_sizes: dict[Component, int] = defaultdict(int)
    output_hashes: dict[Component, dict[str, str]] = defaultdict(dict)

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
                tensors = {}
                for source_key in source_keys:
                    target_key = native_target_key(component, source_key, wan_format=wan_format)
                    if target_key is None:
                        continue
                    tensor = handle.get_tensor(source_key)
                    tensors[target_key] = tensor
                    output_sizes[component] += tensor.numel() * tensor.element_size()

                component_dir = output / component.value
                component_dir.mkdir(parents=True, exist_ok=True)
                output_name = f"model-{source_number:05d}-of-{shard_count:05d}.safetensors"
                output_path = component_dir / output_name
                save_file(tensors, output_path, metadata={"format": "pt", "source": source_name})
                output_hashes[component][output_name] = file_sha256(output_path)
                for target_key in tensors:
                    output_indexes[component][target_key] = output_name
                print(
                    f"  wrote {component.value}/{output_name} "
                    f"({len(tensors)} tensors, {output_path.stat().st_size / 2**30:.2f} GiB)",
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
        (component_dir / "model.safetensors.index.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        outputs[component.value] = {
            "tensors": len(output_indexes[component]),
            "total_size": output_sizes[component],
            "sha256": dict(sorted(output_hashes[component].items())),
        }

    report["outputs"] = outputs
    (output / "repack-manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("repacked") / "bernini-v2-bf16")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wan-format", choices=("comfy", "diffusers"), default="comfy")
    args = parser.parse_args()
    report = repack(
        args.source.resolve(),
        args.output.resolve(),
        dry_run=args.dry_run,
        wan_format=args.wan_format,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
