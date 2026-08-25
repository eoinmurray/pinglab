from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.runstore.contract import ContractError, inventory_payload, write_json_atomic
from tools.runstore.lifecycle import initialize_run
from tools.snnsim.migration import load_equivalence_policy, migration_preflight

POLICY = Path(__file__).parents[1] / "equivalence-policy-v1.json"
GRAPH = "sha256:" + "a" * 64
TRAINING = "sha256:" + "b" * 64


def _campaign(root: Path, *, executor: str) -> Path:
    run = initialize_run(
        root,
        run_id=f"{executor}-campaign",
        kind="campaign",
        experiment=None,
        collection="snnlang",
        command=["collection", "run"],
        repository=root.parent,
        executor=executor,
        graph_digest=GRAPH if executor == "graph" else None,
        training_digest=TRAINING if executor == "graph" else None,
    )
    if executor == "legacy":
        run["kind"] = "legacy"
        run["status"] = "legacy"
    else:
        run["status"] = "complete"
    write_json_atomic(root / "run.json", run)
    write_json_atomic(
        root / "inventory.json", inventory_payload(root, run_id=run["run_id"])
    )
    return root


def _accelerator(path: Path, policy: dict) -> Path:
    write_json_atomic(
        path,
        {
            "schema": "tools/snnsim.accelerator-conformance/v1",
            "passed": True,
            "device_class": "cuda",
            "policy_digest": policy["policy_digest"],
            "graph_digest": GRAPH,
            "training_digest": TRAINING,
        },
    )
    return path


def test_checked_in_policy_is_complete_and_authenticated() -> None:
    policy = load_equivalence_policy(POLICY)
    assert policy["comparisons"]["numeric"]["forward_logits"] == {
        "atol": 1e-6,
        "rtol": 1e-6,
        "equal_nan": False,
    }
    assert "figures" in policy["review_scope"]


def test_policy_rejects_content_changed_without_new_digest(tmp_path: Path) -> None:
    policy = json.loads(POLICY.read_text())
    policy["comparisons"]["numeric"]["forward_logits"]["atol"] = 0.5
    changed = tmp_path / "policy.json"
    changed.write_text(json.dumps(policy))
    with pytest.raises(ContractError, match="digest does not match"):
        load_equivalence_policy(changed)


def test_preflight_reports_every_missing_evidence_gate(tmp_path: Path) -> None:
    report = migration_preflight(
        policy_path=POLICY,
        legacy_campaign=tmp_path / "legacy",
        graph_campaign=tmp_path / "graph",
        accelerator_evidence=tmp_path / "accelerator.json",
        graph_digest=GRAPH,
        training_digest=TRAINING,
    )
    assert report.ready is False
    assert report.gates == {
        "legacy_campaign": False,
        "graph_campaign": False,
        "accelerator_conformance": False,
    }
    assert len(report.failures) == 3


def test_preflight_is_ready_only_for_matching_complete_evidence(tmp_path: Path) -> None:
    policy = load_equivalence_policy(POLICY)
    report = migration_preflight(
        policy_path=POLICY,
        legacy_campaign=_campaign(tmp_path / "legacy", executor="legacy"),
        graph_campaign=_campaign(tmp_path / "graph", executor="graph"),
        accelerator_evidence=_accelerator(tmp_path / "accelerator.json", policy),
        graph_digest=GRAPH,
        training_digest=TRAINING,
    )
    assert report.ready is True
    assert report.failures == ()
    assert report.to_dict()["schema"] == "tools/snnsim.migration-preflight/v1"


def test_preflight_rejects_identity_drift(tmp_path: Path) -> None:
    policy = load_equivalence_policy(POLICY)
    report = migration_preflight(
        policy_path=POLICY,
        legacy_campaign=_campaign(tmp_path / "legacy", executor="legacy"),
        graph_campaign=_campaign(tmp_path / "graph", executor="graph"),
        accelerator_evidence=_accelerator(tmp_path / "accelerator.json", policy),
        graph_digest="sha256:" + "c" * 64,
        training_digest=TRAINING,
    )
    assert report.ready is False
    assert report.gates["legacy_campaign"] is True
    assert report.gates["graph_campaign"] is False
    assert report.gates["accelerator_conformance"] is False
