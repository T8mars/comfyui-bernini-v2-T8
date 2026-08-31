import pytest

from bernini_v2.presets import task_preset


@pytest.mark.parametrize(
    ("task", "steps", "planning_steps", "omega_text"),
    [
        ("t2i", 50, 25, 4.0),
        ("i2i", 40, 25, 4.0),
        ("t2v", 50, 50, 5.0),
        ("v2v", 40, 50, 4.0),
        ("r2v", 40, 50, 4.5),
        ("rv2v", 40, 50, 3.6),
    ],
)
def test_official_task_presets(task, steps, planning_steps, omega_text):
    preset = task_preset(task)
    assert preset["steps"] == steps
    assert preset["planning_steps"] == planning_steps
    assert preset["omega_text"] == omega_text


def test_task_preset_returns_a_copy():
    preset = task_preset("t2v")
    preset["steps"] = 1
    assert task_preset("t2v")["steps"] == 50
