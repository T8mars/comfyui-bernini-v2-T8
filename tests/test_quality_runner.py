from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from run_comfy_quality import prepare_graph  # noqa: E402


def test_quality_runner_preserves_official_defaults_and_overrides_media() -> None:
    graph = prepare_graph(
        "rv2v",
        width=640,
        height=384,
        length=81,
        reference_image=["reference.png", "reference-2.png", "reference-3.png"],
        source_video="source.mp4",
        output_prefix="video/Bernini-v2/quality/rv2v_acceptance",
        prompt="Preserve the dog and add a snowman.",
    )

    assert graph["11"]["inputs"]["use_task_defaults"] is True
    assert graph["12"]["inputs"]["use_task_defaults"] is True
    assert graph["13"]["inputs"]["use_task_defaults"] is True
    assert graph["11"]["inputs"]["width"] == 640
    assert graph["11"]["inputs"]["height"] == 384
    assert graph["11"]["inputs"]["length"] == 81
    assert graph["8"]["inputs"]["image"] == "reference.png"
    assert graph["20"]["inputs"]["image"] == "reference-2.png"
    assert graph["21"]["inputs"]["image"] == "reference-3.png"
    assert graph["11"]["inputs"]["reference_images.reference_image_1"] == ["20", 0]
    assert graph["11"]["inputs"]["reference_images.reference_image_2"] == ["21", 0]
    assert graph["9"]["inputs"]["file"] == "source.mp4"
    assert graph["11"]["inputs"]["prompt"] == "Preserve the dog and add a snowman."
    assert graph["3"]["inputs"]["text"].endswith("Preserve the dog and add a snowman.")
    assert graph["19"]["inputs"]["filename_prefix"] == "video/Bernini-v2/quality/rv2v_acceptance"
