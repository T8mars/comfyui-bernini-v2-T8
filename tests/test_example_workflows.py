import json

import pytest

from tools.build_example_workflows import (
    DEFAULT_PROMPT,
    OFFICIAL_DEFAULT_NEGATIVE,
    SYSTEM_PROMPT,
    build_workflow,
    validate_graph,
    write_examples,
)


@pytest.mark.parametrize("task", DEFAULT_PROMPT)
def test_example_workflow_is_connected_and_has_exact_task_media(task):
    graph = build_workflow(task)
    validate_graph(task, graph)


def test_video_save_uses_v3_dynamic_combo_paths():
    inputs = build_workflow("t2v")["19"]["inputs"]
    assert inputs["format"] == "mp4"
    assert inputs["format.codec"] == "h264"


def test_video_examples_use_two_second_long_edge_640_and_low_memory_guidance():
    graph = build_workflow("t2v")
    assert graph["11"]["inputs"]["width"] == 640
    assert graph["11"]["inputs"]["height"] == 368
    assert graph["11"]["inputs"]["length"] == 33
    assert graph["12"]["inputs"]["guidance_batch_size"] == "auto"


def test_t2i_uses_official_text_conditions_bf16_and_flow_solver():
    graph = build_workflow("t2i")
    assert graph["3"]["inputs"]["text"] == SYSTEM_PROMPT["t2i"] + DEFAULT_PROMPT["t2i"]
    assert graph["4"]["inputs"]["text"] == OFFICIAL_DEFAULT_NEGATIVE
    assert graph["11"]["inputs"]["negative_prompt"] == OFFICIAL_DEFAULT_NEGATIVE
    assert graph["5"]["inputs"]["weight_dtype"] == "bfloat16"
    assert graph["6"]["inputs"]["weight_dtype"] == "bfloat16"
    assert graph["2"]["inputs"]["dtype"] == "bfloat16"
    assert graph["15"]["class_type"] == "BerniniV2UniPCSampler"


def test_writes_all_six_json_files(tmp_path):
    write_examples(tmp_path)
    paths = sorted(tmp_path.glob("*.json"))
    assert [path.stem for path in paths] == sorted(DEFAULT_PROMPT)
    for path in paths:
        graph = json.loads(path.read_text(encoding="utf-8"))
        validate_graph(path.stem, graph)
