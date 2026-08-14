from __future__ import annotations

import json

import pytest
import torch
from conformance import (
    CONFORMANCE_REPORT_SCHEMA,
    ComparisonPolicy,
    canonical_json_tensor,
    compare_conformance_layers,
    remap_named_tensors,
    write_conformance_report,
)


def test_layered_conformance_requires_complete_exact_named_coverage(tmp_path):
    reference = {
        "parameters": {"input.weight": torch.tensor([[1.0, 2.0]])},
        "forward": {"logits": torch.tensor([[0.25, 0.75]])},
    }
    report = compare_conformance_layers("hand-checkable", reference, reference)
    assert report.passed
    report.require_passed()
    path = write_conformance_report(tmp_path / "conformance.json", report)
    payload = json.loads(path.read_text())
    assert payload["schema"] == CONFORMANCE_REPORT_SCHEMA
    assert payload["summary"] == {"comparisons": 2, "failed": 0, "passed": 2}

    incomplete = compare_conformance_layers(
        "missing", reference, {"parameters": reference["parameters"]}
    )
    assert not incomplete.passed
    assert incomplete.comparisons[0].reason == "missing from candidate"
    with pytest.raises(AssertionError, match="forward.logits"):
        incomplete.require_passed()


def test_numeric_policy_reports_error_and_never_hides_shape_or_dtype_mismatch():
    reference = {"gradients": {"readout.weight": torch.tensor([1.0, 2.0])}}
    close = {"gradients": {"readout.weight": torch.tensor([1.0, 2.00001])}}
    policy = {
        "gradients": {"readout.weight": ComparisonPolicy(mode="numeric", atol=2e-5)}
    }
    report = compare_conformance_layers("tolerant", reference, close, policies=policy)
    assert report.passed
    assert report.comparisons[0].max_abs_error == pytest.approx(1e-5, rel=0.01)

    wrong_dtype = {
        "gradients": {"readout.weight": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    }
    mismatch = compare_conformance_layers(
        "dtype", reference, wrong_dtype, policies=policy
    )
    assert mismatch.comparisons[0].reason == "dtype mismatch"


def test_conformance_rejects_implicit_or_unused_tolerance_rules():
    with pytest.raises(ValueError, match="exact.*tolerances"):
        ComparisonPolicy(atol=1e-6)
    with pytest.raises(ValueError, match="absent fields"):
        compare_conformance_layers(
            "unused",
            {"forward": {"logits": torch.zeros(1)}},
            {"forward": {"logits": torch.zeros(1)}},
            policies={"forward": {"rates": ComparisonPolicy(mode="numeric")}},
        )


def test_canonical_json_tensor_compares_structural_layers_independent_of_key_order():
    first = canonical_json_tensor({"b": [2, 3], "a": 1})
    second = canonical_json_tensor({"a": 1, "b": [2, 3]})
    assert torch.equal(first, second)


def test_explicit_name_remapping_rejects_partial_and_duplicate_maps():
    values = {"graph.a": torch.ones(1), "graph.b": torch.zeros(1)}
    assert set(
        remap_named_tensors(values, {"graph.a": "legacy.a", "graph.b": "legacy.b"})
    ) == {"legacy.a", "legacy.b"}
    with pytest.raises(ValueError, match="must be complete"):
        remap_named_tensors(values, {"graph.a": "legacy.a"})
    with pytest.raises(ValueError, match="duplicate destination"):
        remap_named_tensors(values, {"graph.a": "legacy.a", "graph.b": "legacy.a"})
