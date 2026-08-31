from __future__ import annotations

from pathlib import Path


def test_low_expert_switch_reserves_estimated_inference_memory() -> None:
    source = (Path(__file__).parents[1] / "nodes" / "rendering.py").read_text(encoding="utf-8")

    assert "comfy.sampler_helpers.estimate_memory(" in source
    assert "memory_required=memory_required" in source
    assert "minimum_memory_required=minimum_memory_required" in source
