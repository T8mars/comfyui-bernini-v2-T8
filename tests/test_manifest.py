import json

import pytest

from bernini_v2.manifest import REQUIRED_COMPONENTS, load_repack_manifest


def _manifest(tmp_path, **overrides):
    payload = {
        "schema_version": 3,
        "format": "bernini_v2_int8_tensorwise_convrot",
        "storage_dtype": "bfloat16",
        "outputs": {name: {} for name in REQUIRED_COMPONENTS},
    }
    payload.update(overrides)
    path = tmp_path / "repack-manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_accepts_current_native_quant_manifest(tmp_path):
    assert load_repack_manifest(_manifest(tmp_path))["storage_dtype"] == "bfloat16"


def test_rejects_future_schema(tmp_path):
    with pytest.raises(ValueError, match="schema_version"):
        load_repack_manifest(_manifest(tmp_path, schema_version=999))


def test_rejects_missing_runtime_component(tmp_path):
    with pytest.raises(ValueError, match="missing components"):
        load_repack_manifest(_manifest(tmp_path, outputs={"wan_high": {}}))
