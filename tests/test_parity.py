import json

import pytest
import torch

from bernini_v2.parity import (
    PARITY_SCHEMA,
    TensorTolerance,
    compare_artifacts,
    compare_tensor,
    load_parity_artifact,
    save_parity_artifact,
    write_parity_report,
)


def test_compare_tensor_reports_error_metrics_and_tolerance():
    reference = torch.tensor([1.0, 2.0, 4.0], dtype=torch.bfloat16)
    candidate = torch.tensor([1.0, 2.015625, 4.0], dtype=torch.bfloat16)

    passing = compare_tensor(reference, candidate, TensorTolerance(atol=0.02, rtol=0.0))
    failing = compare_tensor(reference, candidate, TensorTolerance(atol=0.01, rtol=0.0))

    assert passing["passed"] is True
    assert passing["max_abs"] == pytest.approx(0.015625)
    assert passing["violations"] == 0
    assert passing["acceptance"] == "elementwise"
    assert failing["passed"] is False
    assert failing["failure"] == "outside_tolerance"
    assert failing["violations"] == 1


def test_compare_tensor_rejects_shape_and_nonfinite_mismatches():
    shape = compare_tensor(torch.zeros(2), torch.zeros(3), TensorTolerance(atol=0.0, rtol=0.0))
    finite = compare_tensor(
        torch.tensor([1.0, float("nan")]),
        torch.tensor([1.0, 2.0]),
        TensorTolerance(atol=0.0, rtol=0.0),
    )

    assert shape["failure"] == "shape_mismatch"
    assert finite["failure"] == "finite_mask_mismatch"
    assert finite["passed"] is False


def test_compare_tensor_rejects_dtype_mismatch():
    result = compare_tensor(
        torch.ones(2, dtype=torch.float32),
        torch.ones(2, dtype=torch.float16),
        TensorTolerance(atol=0.0, rtol=0.0),
    )

    assert result["passed"] is False
    assert result["failure"] == "dtype_mismatch"


def test_parity_artifact_round_trip_and_comparison(tmp_path):
    reference_path = tmp_path / "official.safetensors"
    candidate_path = tmp_path / "native.safetensors"
    save_parity_artifact(
        reference_path,
        {"hidden": torch.arange(8, dtype=torch.float32).reshape(2, 4)},
        metadata={"implementation": "official", "stage": "planner_hidden"},
    )
    save_parity_artifact(
        candidate_path,
        {"hidden": torch.arange(8, dtype=torch.float32).reshape(2, 4)},
        metadata={"implementation": "native", "stage": "planner_hidden"},
    )

    tensors, metadata = load_parity_artifact(reference_path)
    report = compare_artifacts(
        reference_path,
        candidate_path,
        tolerance=TensorTolerance(atol=0.0, rtol=0.0),
    )

    assert torch.equal(tensors["hidden"], torch.arange(8, dtype=torch.float32).reshape(2, 4))
    assert metadata["schema"] == PARITY_SCHEMA
    assert report["passed"] is True


def test_compare_artifacts_reports_key_contract_and_writes_json(tmp_path):
    reference_path = tmp_path / "official.safetensors"
    candidate_path = tmp_path / "native.safetensors"
    report_path = tmp_path / "report.json"
    save_parity_artifact(
        reference_path,
        {"expected": torch.zeros(1)},
        metadata={"implementation": "official"},
    )
    save_parity_artifact(
        candidate_path,
        {"other": torch.zeros(1)},
        metadata={"implementation": "native"},
    )

    report = compare_artifacts(
        reference_path,
        candidate_path,
        tolerance=TensorTolerance(atol=0.0, rtol=0.0),
    )
    write_parity_report(report_path, report)

    assert report["passed"] is False
    assert report["missing_tensors"] == ["expected"]
    assert report["unexpected_tensors"] == ["other"]
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is False


def test_negative_tolerance_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        TensorTolerance(atol=-1.0, rtol=0.0)


def test_compare_tensor_accepts_bounded_aggregate_error():
    reference = torch.full((100,), 100.0)
    candidate = reference + torch.linspace(-1.0, 1.0, 100)
    result = compare_tensor(
        reference,
        candidate,
        TensorTolerance(
            atol=0.1,
            rtol=0.0,
            max_normalized_rmse=0.015,
            min_cosine_similarity=0.999,
        ),
    )

    assert result["elementwise_passed"] is False
    assert result["aggregate_passed"] is True
    assert result["acceptance"] == "aggregate"
    assert result["passed"] is True


def test_compare_tensor_rejects_aggregate_direction_change():
    reference = torch.arange(1.0, 11.0)
    result = compare_tensor(
        reference,
        -reference,
        TensorTolerance(
            atol=0.1,
            rtol=0.0,
            max_normalized_rmse=3.0,
            min_cosine_similarity=0.999,
        ),
    )

    assert result["aggregate_passed"] is False
    assert result["passed"] is False


def test_invalid_aggregate_tolerances_are_rejected():
    with pytest.raises(ValueError, match="normalized RMSE"):
        TensorTolerance(atol=0.0, rtol=0.0, max_normalized_rmse=-0.1)
    with pytest.raises(ValueError, match="cosine similarity"):
        TensorTolerance(atol=0.0, rtol=0.0, min_cosine_similarity=1.1)
