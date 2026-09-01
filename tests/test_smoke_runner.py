import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "tools" / "run_comfy_smoke.py"
SPEC = importlib.util.spec_from_file_location("run_comfy_smoke", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_prepare_graph_overrides_every_task_without_mutating_examples():
    for task in MODULE.TASKS:
        graph = MODULE.prepare_graph(
            task,
            width=320,
            height=192,
            length=5,
            renderer_steps=3,
            seed=0xFFFFFFFFFFFFFFFF,
        )
        plan = graph["11"]["inputs"]
        assert plan["width"] == 320
        assert plan["height"] == 192
        assert plan["length"] == (1 if task in {"t2i", "i2i"} else 5)
        assert plan["use_task_defaults"] is False
        assert graph["12"]["inputs"]["use_task_defaults"] is False
        preset = MODULE.task_preset(task)
        assert graph["12"]["inputs"]["omega_video"] == preset["omega_video"]
        assert graph["12"]["inputs"]["omega_image"] == preset["omega_image"]
        assert graph["12"]["inputs"]["omega_text"] == preset["omega_text"]
        assert graph["12"]["inputs"]["omega_target"] == preset["omega_target"]
        assert graph["12"]["inputs"]["omega_scale"] == preset["omega_scale"]
        assert graph["13"]["inputs"]["steps"] == 3
        assert plan["seed"] == 0xFFFFFFFFFFFFFFFF
        assert graph["14"]["inputs"]["noise_seed"] == 0xFFFFFFFFFFFFFFFF


def test_prepare_graph_sets_media_and_output_names():
    graph = MODULE.prepare_graph(
        "rv2v",
        reference_image="subject.png",
        source_video="source.mp4",
        output_prefix="checks/rv2v",
    )
    assert graph["8"]["inputs"]["image"] == "subject.png"
    assert graph["9"]["inputs"]["file"] == "source.mp4"
    assert graph["19"]["inputs"]["filename_prefix"] == "checks/rv2v"


def test_prepare_graph_can_switch_single_files():
    graph = MODULE.prepare_graph(
        "t2v",
        planner_name="bernini_v2_planner_bf16.safetensors",
        t5_name="umt5_xxl_bernini_v2_bf16.safetensors",
        high_renderer="bernini_v2_high_noise_bf16.safetensors",
        low_renderer="bernini_v2_low_noise_bf16.safetensors",
    )
    assert graph["1"]["inputs"]["planner_name"] == "bernini_v2_planner_bf16.safetensors"
    assert graph["2"]["inputs"]["clip_name"] == "umt5_xxl_bernini_v2_bf16.safetensors"
    assert graph["5"]["inputs"]["unet_name"] == "bernini_v2_high_noise_bf16.safetensors"
    assert graph["6"]["inputs"]["unet_name"] == "bernini_v2_low_noise_bf16.safetensors"
