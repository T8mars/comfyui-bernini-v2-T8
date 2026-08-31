"""Read and validate Hugging Face sharded safetensors indexes."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .state_dict import COMPONENTS, Component, classify_key, component_key


@dataclass(frozen=True)
class IndexPlan:
    index_path: Path
    total_size: int
    weight_map: dict[str, str]
    by_component: dict[Component, tuple[str, ...]]
    by_shard: dict[str, tuple[str, ...]]

    @property
    def tensor_count(self) -> int:
        return len(self.weight_map)


def load_index(index_path: str | Path) -> IndexPlan:
    path = Path(index_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid or empty weight_map in {path}")

    by_component: dict[Component, list[str]] = defaultdict(list)
    by_shard: dict[str, list[str]] = defaultdict(list)
    targets: dict[Component, set[str]] = defaultdict(set)
    for source_key, shard in weight_map.items():
        component, target_key = component_key(source_key)
        if target_key in targets[component]:
            raise ValueError(f"duplicate target key {component.value}:{target_key}")
        targets[component].add(target_key)
        by_component[component].append(source_key)
        by_shard[shard].append(source_key)

    missing_components = set(COMPONENTS) - set(by_component)
    if missing_components:
        raise ValueError(f"index is missing components: {sorted(item.value for item in missing_components)}")

    return IndexPlan(
        index_path=path,
        total_size=int(payload.get("metadata", {}).get("total_size", 0)),
        weight_map=dict(weight_map),
        by_component={key: tuple(sorted(value)) for key, value in by_component.items()},
        by_shard={key: tuple(sorted(value)) for key, value in by_shard.items()},
    )


def summarize(plan: IndexPlan) -> dict[str, object]:
    counts = Counter(classify_key(key).value for key in plan.weight_map)
    shards = Counter(plan.weight_map[key] for key in plan.weight_map)
    return {
        "index": str(plan.index_path),
        "tensors": plan.tensor_count,
        "total_size": plan.total_size,
        "components": dict(sorted(counts.items())),
        "source_shards": len(shards),
    }
