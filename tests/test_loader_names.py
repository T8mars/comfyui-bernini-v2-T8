from pathlib import Path


def test_portable_model_name_uses_workflow_stable_separators():
    # Read the tiny pure helper without importing ComfyUI-only dependencies.
    source = (Path(__file__).parents[1] / "nodes" / "loaders.py").read_text(encoding="utf-8")
    assert 'return name.replace("\\\\", "/")' in source
