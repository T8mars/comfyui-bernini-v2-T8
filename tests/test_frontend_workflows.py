import json

import pytest

from tools.build_example_workflows import DEFAULT_PROMPT, OFFICIAL_DEFAULT_NEGATIVE, SYSTEM_PROMPT
from tools.build_frontend_workflows import (
    build_frontend_workflow,
    validate_frontend_workflow,
    write_frontend_workflows,
)


@pytest.mark.parametrize("task", DEFAULT_PROMPT)
def test_frontend_workflow_is_connected_and_task_specific(task):
    workflow = build_frontend_workflow(task)
    validate_frontend_workflow(task, workflow)
    assert workflow["version"] == 0.4
    assert workflow["nodes"]
    assert workflow["links"]


def test_t2i_frontend_uses_quality_defaults():
    workflow = build_frontend_workflow("t2i")
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert nodes[2]["widgets_values"][1] == "bfloat16"
    assert nodes[3]["widgets_values"][0] == SYSTEM_PROMPT["t2i"] + DEFAULT_PROMPT["t2i"]
    assert nodes[4]["widgets_values"][0] == OFFICIAL_DEFAULT_NEGATIVE
    assert nodes[5]["widgets_values"][-1] == "bfloat16"
    assert nodes[6]["widgets_values"][-1] == "bfloat16"
    assert nodes[11]["widgets_values"][2:6] == ["t2i", 512, 512, 1]
    assert nodes[15]["type"] == "BerniniV2UniPCSampler"


def test_video_frontend_uses_two_second_long_edge_640_and_low_memory_guidance():
    workflow = build_frontend_workflow("t2v")
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert nodes[11]["widgets_values"][2:6] == ["t2v", 640, 368, 33]
    assert nodes[12]["widgets_values"][-2:] == ["auto", "auto"]


def test_writes_all_six_frontend_workflows(tmp_path):
    write_frontend_workflows(tmp_path)
    paths = sorted(tmp_path.glob("*.json"))
    assert [path.stem for path in paths] == sorted(DEFAULT_PROMPT)
    for path in paths:
        workflow = json.loads(path.read_text(encoding="utf-8"))
        validate_frontend_workflow(path.stem, workflow)
