from pathlib import Path


def test_quantized_wan_loader_forces_selected_compute_dtype_through_stock_mixed_ops():
    source = (Path(__file__).parents[1] / "nodes" / "loaders.py").read_text(encoding="utf-8")
    assert 'if "scaled_fp8" in state_dict:' in source
    assert "comfy.utils.convert_old_quants(state_dict)" in source
    assert 'model_options["custom_operations"] = comfy.ops.mixed_precision_ops({}, compute_dtype)' in source
