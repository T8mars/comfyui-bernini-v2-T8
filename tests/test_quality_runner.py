from __future__ import annotations

import sys
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_comfy_quality  # noqa: E402
from run_comfy_quality import ResourceMonitor, prepare_graph  # noqa: E402


def test_quality_runner_preserves_official_defaults_and_overrides_media() -> None:
    graph = prepare_graph(
        "rv2v",
        width=640,
        height=368,
        length=33,
        reference_image=["reference.png", "reference-2.png", "reference-3.png"],
        source_video="source.mp4",
        output_prefix="video/Bernini-v2/quality/rv2v_acceptance",
        prompt="Preserve the dog and add a snowman.",
    )

    assert graph["11"]["inputs"]["use_task_defaults"] is True
    assert graph["12"]["inputs"]["use_task_defaults"] is True
    assert graph["13"]["inputs"]["use_task_defaults"] is True
    assert graph["11"]["inputs"]["width"] == 640
    assert graph["11"]["inputs"]["height"] == 368
    assert graph["11"]["inputs"]["length"] == 33
    assert graph["8"]["inputs"]["image"] == "reference.png"
    assert graph["20"]["inputs"]["image"] == "reference-2.png"
    assert graph["21"]["inputs"]["image"] == "reference-3.png"
    assert graph["11"]["inputs"]["reference_images.reference_image_1"] == ["20", 0]
    assert graph["11"]["inputs"]["reference_images.reference_image_2"] == ["21", 0]
    assert graph["9"]["inputs"]["file"] == "source.mp4"
    assert graph["11"]["inputs"]["prompt"] == "Preserve the dog and add a snowman."
    assert graph["3"]["inputs"]["text"].endswith("Preserve the dog and add a snowman.")
    assert graph["19"]["inputs"]["filename_prefix"] == "video/Bernini-v2/quality/rv2v_acceptance"


@pytest.mark.parametrize(
    ("loader", "class_type", "input_name"),
    [
        ("native", "UNETLoader", "unet_name"),
        ("gguf", "UnetLoaderGGUF", "unet_name"),
        ("bernini", "BerniniV2WanLoader", "unet_name"),
    ],
)
def test_quality_runner_can_override_only_the_renderer_pair(loader, class_type, input_name) -> None:
    graph = prepare_graph(
        "t2v",
        planner_name="bernini_v2_planner_int8.safetensors",
        t5_name="umt5_xxl_bernini_v2_int8.safetensors",
        high_renderer="renderers/high.model",
        low_renderer="renderers/low.model",
        renderer_loader=loader,
    )

    assert graph["1"]["inputs"]["planner_name"] == "bernini_v2_planner_int8.safetensors"
    assert graph["2"]["inputs"]["clip_name"] == "umt5_xxl_bernini_v2_int8.safetensors"
    assert graph["5"]["class_type"] == class_type
    assert graph["6"]["class_type"] == class_type
    expected_high = "renderers/high.model" if loader == "bernini" else str(Path("renderers/high.model"))
    expected_low = "renderers/low.model" if loader == "bernini" else str(Path("renderers/low.model"))
    assert graph["5"]["inputs"][input_name] == expected_high
    assert graph["6"]["inputs"][input_name] == expected_low


def test_quality_runner_rejects_half_a_renderer_pair() -> None:
    with pytest.raises(ValueError, match="must be supplied together"):
        prepare_graph("t2v", high_renderer="renderers/high.safetensors")


def test_resource_monitor_aggregates_venv_child_processes(monkeypatch) -> None:
    memory = namedtuple("memory", "rss vms")

    class FakeProcess:
        def __init__(self, rss, vms, children=()):
            self._memory = memory(rss, vms)
            self._children = list(children)

        def children(self, recursive=False):
            assert recursive is True
            return self._children

        def memory_info(self):
            return self._memory

    child = FakeProcess(200, 400)
    root = FakeProcess(100, 300, [child])
    fake_psutil = SimpleNamespace(Process=lambda pid: root, Error=RuntimeError)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(run_comfy_quality, "request_json", lambda url: {"devices": []})

    monitor = ResourceMonitor("http://127.0.0.1:8199", server_pid=123)
    monitor.sample()

    assert monitor.peak_rss == 300
    assert monitor.peak_vms == 700
