"""Tensor artifact and numerical comparison helpers for implementation parity."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

PARITY_SCHEMA = "bernini_v2_tensor_parity_v1"


@dataclass(frozen=True)
class TensorTolerance:
    atol: float
    rtol: float
    max_normalized_rmse: float | None = None
    min_cosine_similarity: float | None = None

    def __post_init__(self) -> None:
        if self.atol < 0 or self.rtol < 0:
            raise ValueError("tensor tolerances must be non-negative")
        if self.max_normalized_rmse is not None and self.max_normalized_rmse < 0:
            raise ValueError("normalized RMSE tolerance must be non-negative")
        if self.min_cosine_similarity is not None and not -1 <= self.min_cosine_similarity <= 1:
            raise ValueError("cosine similarity tolerance must be between -1 and 1")


def _safe_float(value: torch.Tensor) -> float | None:
    scalar = float(value.item())
    return scalar if torch.isfinite(value).item() else None


def compare_tensor(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    tolerance: TensorTolerance,
) -> dict[str, object]:
    """Compare one candidate tensor against an authoritative reference tensor."""

    shape_matches = reference.shape == candidate.shape
    dtype_matches = reference.dtype == candidate.dtype
    report: dict[str, object] = {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_dtype": str(reference.dtype).removeprefix("torch."),
        "candidate_dtype": str(candidate.dtype).removeprefix("torch."),
        "shape_matches": shape_matches,
        "dtype_matches": dtype_matches,
        "atol": tolerance.atol,
        "rtol": tolerance.rtol,
    }
    if not shape_matches:
        report.update({"passed": False, "failure": "shape_mismatch"})
        return report

    reference_float = reference.detach().to(device="cpu", dtype=torch.float32)
    candidate_float = candidate.detach().to(device="cpu", dtype=torch.float32)
    reference_finite = torch.isfinite(reference_float)
    candidate_finite = torch.isfinite(candidate_float)
    finite_matches = bool(torch.equal(reference_finite, candidate_finite))
    jointly_finite = reference_finite & candidate_finite
    report["finite_matches"] = finite_matches
    report["finite_values"] = int(jointly_finite.sum().item())
    report["elements"] = reference.numel()

    if not jointly_finite.any():
        report.update(
            {
                "max_abs": None,
                "mean_abs": None,
                "rmse": None,
                "max_rel": None,
                "cosine_similarity": None,
                "violations": reference.numel(),
                "passed": False,
                "failure": "no_jointly_finite_values",
            }
        )
        return report

    reference_values = reference_float[jointly_finite]
    candidate_values = candidate_float[jointly_finite]
    absolute = (candidate_values - reference_values).abs()
    relative = absolute / reference_values.abs().clamp_min(max(tolerance.atol, torch.finfo(torch.float32).eps))
    allowed = tolerance.atol + tolerance.rtol * reference_values.abs()
    violations = int((absolute > allowed).sum().item())
    cosine = torch.nn.functional.cosine_similarity(
        reference_values.reshape(1, -1),
        candidate_values.reshape(1, -1),
        dim=1,
        eps=1e-12,
    )[0]
    cosine = cosine.clamp(-1.0, 1.0)
    reference_rms = reference_values.square().mean().sqrt()
    rmse = absolute.square().mean().sqrt()
    if reference_rms.item() == 0:
        normalized_rmse = torch.zeros_like(rmse) if rmse.item() == 0 else torch.full_like(rmse, float("inf"))
    else:
        normalized_rmse = rmse / reference_rms
    aggregate_checks = []
    if tolerance.max_normalized_rmse is not None:
        aggregate_checks.append(normalized_rmse.item() <= tolerance.max_normalized_rmse)
    if tolerance.min_cosine_similarity is not None:
        aggregate_checks.append(cosine.item() >= tolerance.min_cosine_similarity)
    elementwise_passed = violations == 0
    aggregate_passed = bool(aggregate_checks) and all(aggregate_checks)
    numerical_passed = elementwise_passed or aggregate_passed
    passed = dtype_matches and finite_matches and numerical_passed
    report.update(
        {
            "max_abs": _safe_float(absolute.max()),
            "mean_abs": _safe_float(absolute.mean()),
            "rmse": _safe_float(rmse),
            "reference_rms": _safe_float(reference_rms),
            "normalized_rmse": _safe_float(normalized_rmse),
            "max_rel": _safe_float(relative.max()),
            "cosine_similarity": _safe_float(cosine),
            "violations": violations,
            "elementwise_passed": elementwise_passed,
            "aggregate_passed": aggregate_passed,
            "max_normalized_rmse": tolerance.max_normalized_rmse,
            "min_cosine_similarity": tolerance.min_cosine_similarity,
            "acceptance": "elementwise" if elementwise_passed else "aggregate" if aggregate_passed else None,
            "passed": passed,
        }
    )
    if not dtype_matches:
        report["failure"] = "dtype_mismatch"
    elif not finite_matches:
        report["failure"] = "finite_mask_mismatch"
    elif not numerical_passed:
        report["failure"] = "outside_tolerance"
    return report


def save_parity_artifact(
    path: str | Path,
    tensors: Mapping[str, torch.Tensor],
    *,
    metadata: Mapping[str, str],
) -> None:
    """Atomically save CPU-contiguous tensors with the parity schema metadata."""

    path = Path(path)
    if not tensors:
        raise ValueError("a parity artifact must contain at least one tensor")
    normalized = {}
    for name, tensor in tensors.items():
        if not name or not torch.is_tensor(tensor):
            raise TypeError("parity artifact entries must be named tensors")
        normalized[name] = tensor.detach().to("cpu").contiguous().clone()
    artifact_metadata = {str(key): str(value) for key, value in metadata.items()}
    artifact_metadata["schema"] = PARITY_SCHEMA
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        save_file(normalized, temporary_name, metadata=artifact_metadata)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_parity_artifact(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    path = Path(path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("schema") != PARITY_SCHEMA:
        raise ValueError(f"{path} is not a {PARITY_SCHEMA} artifact")
    return load_file(path, device="cpu"), metadata


def compare_artifacts(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    tolerance: TensorTolerance,
    per_tensor: Mapping[str, TensorTolerance] | None = None,
) -> dict[str, object]:
    """Compare matching tensors in two saved artifacts and return a JSON-ready report."""

    reference, reference_metadata = load_parity_artifact(reference_path)
    candidate, candidate_metadata = load_parity_artifact(candidate_path)
    reference_names = set(reference)
    candidate_names = set(candidate)
    missing = sorted(reference_names - candidate_names)
    unexpected = sorted(candidate_names - reference_names)
    comparisons = {
        name: compare_tensor(
            reference[name],
            candidate[name],
            (per_tensor or {}).get(name, tolerance),
        )
        for name in sorted(reference_names & candidate_names)
    }
    passed = not missing and not unexpected and all(result["passed"] for result in comparisons.values())
    return {
        "schema": PARITY_SCHEMA,
        "reference": str(Path(reference_path).resolve()),
        "candidate": str(Path(candidate_path).resolve()),
        "reference_metadata": reference_metadata,
        "candidate_metadata": candidate_metadata,
        "default_tolerance": asdict(tolerance),
        "missing_tensors": missing,
        "unexpected_tensors": unexpected,
        "tensors": comparisons,
        "passed": passed,
    }


def write_parity_report(path: str | Path, report: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as output:
            output.write(payload)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
