"""Frozen equivalence policy and read-only SNNLANG campaign preflight."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.runstore.contract import (
    ContractError,
    load_json,
    validate_inventory,
    validate_run_manifest,
    verify_payload,
)

POLICY_SCHEMA = "tools/snnsim.equivalence-policy/v1"
PREFLIGHT_SCHEMA = "tools/snnsim.migration-preflight/v1"
ACCELERATOR_EVIDENCE_SCHEMA = "tools/snnsim.accelerator-conformance/v1"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def _canonical_digest(value: dict[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("policy_digest", None)
    payload = json.dumps(unsigned, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def load_equivalence_policy(path: str | Path) -> dict[str, Any]:
    """Load a complete policy and authenticate its content against its digest."""
    policy = load_json(Path(path))
    required = {
        "schema",
        "policy_id",
        "policy_digest",
        "freeze_rule",
        "comparisons",
        "required_evidence",
        "review_scope",
    }
    if set(policy) != required:
        raise ContractError(
            f"equivalence policy fields differ: missing={sorted(required - set(policy))}, "
            f"extra={sorted(set(policy) - required)}"
        )
    if policy["schema"] != POLICY_SCHEMA:
        raise ContractError(f"equivalence policy schema must be {POLICY_SCHEMA!r}")
    if not isinstance(policy["policy_id"], str) or not policy["policy_id"]:
        raise ContractError("equivalence policy policy_id must be a non-empty string")
    comparisons = policy["comparisons"]
    if not isinstance(comparisons, dict) or set(comparisons) != {"exact", "numeric"}:
        raise ContractError(
            "equivalence policy comparisons must contain exact and numeric"
        )
    if not isinstance(comparisons["exact"], list) or not comparisons["exact"]:
        raise ContractError("equivalence policy exact comparisons must be non-empty")
    if not isinstance(comparisons["numeric"], dict) or not comparisons["numeric"]:
        raise ContractError("equivalence policy numeric comparisons must be non-empty")
    for name, tolerance in comparisons["numeric"].items():
        if not isinstance(name, str) or set(tolerance) != {"atol", "rtol", "equal_nan"}:
            raise ContractError(
                "each numeric comparison needs atol, rtol, and equal_nan"
            )
        if any(
            not isinstance(tolerance[key], (int, float)) or tolerance[key] < 0
            for key in ("atol", "rtol")
        ):
            raise ContractError("numeric tolerances must be non-negative numbers")
        if not isinstance(tolerance["equal_nan"], bool):
            raise ContractError("numeric equal_nan must be boolean")
    for field in ("required_evidence", "review_scope"):
        if (
            not isinstance(policy[field], list)
            or not policy[field]
            or not all(isinstance(item, str) and item for item in policy[field])
        ):
            raise ContractError(
                f"equivalence policy {field} must be a non-empty string array"
            )
    if policy["policy_digest"] != _canonical_digest(policy):
        raise ContractError(
            "equivalence policy digest does not match its canonical content"
        )
    return policy


@dataclass(frozen=True)
class PreflightReport:
    policy_id: str
    policy_digest: str
    gates: dict[str, bool]
    failures: tuple[str, ...]
    schema: str = PREFLIGHT_SCHEMA

    @property
    def ready(self) -> bool:
        return all(self.gates.values()) and not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "policy_id": self.policy_id,
            "policy_digest": self.policy_digest,
            "ready": self.ready,
            "gates": self.gates,
            "failures": list(self.failures),
        }


def migration_preflight(
    *,
    policy_path: str | Path,
    legacy_campaign: str | Path,
    graph_campaign: str | Path,
    accelerator_evidence: str | Path,
    graph_digest: str,
    training_digest: str,
) -> PreflightReport:
    """Check immutable evidence without creating, executing, or activating campaigns."""
    policy = load_equivalence_policy(policy_path)
    if not _DIGEST.fullmatch(graph_digest) or not _DIGEST.fullmatch(training_digest):
        raise ContractError(
            "preflight graph and training identities must be prefixed SHA-256 digests"
        )
    failures: list[str] = []
    gates = {name: False for name in policy["required_evidence"]}

    def campaign_gate(name: str, root_value: str | Path, executor: str) -> None:
        root = Path(root_value)
        try:
            run = validate_run_manifest(load_json(root / "run.json"))
            inventory = validate_inventory(load_json(root / "inventory.json"))
            if run["kind"] not in {"campaign", "legacy"}:
                raise ContractError("evidence is not a campaign")
            expected_status = {"legacy"} if executor == "legacy" else {"complete"}
            if run["status"] not in expected_status:
                raise ContractError(
                    f"{executor} campaign status must be one of {sorted(expected_status)}"
                )
            if run["execution"].get("executor", "legacy") != executor:
                raise ContractError(f"campaign executor must be {executor}")
            if run["execution"].get("collection") != "snnlang":
                raise ContractError("campaign collection must be snnlang")
            if inventory["run_id"] != run["run_id"]:
                raise ContractError("campaign inventory run_id differs from run.json")
            verify_payload(root, inventory)
            if executor == "graph" and (
                run["execution"].get("graph_digest") != graph_digest
                or run["execution"].get("training_digest") != training_digest
            ):
                raise ContractError("graph campaign graph/training identity differs")
            gates[name] = True
        except (ContractError, OSError) as exc:
            failures.append(f"{name}: {exc}")

    campaign_gate("legacy_campaign", legacy_campaign, "legacy")
    campaign_gate("graph_campaign", graph_campaign, "graph")
    try:
        evidence = load_json(Path(accelerator_evidence))
        if evidence.get("schema") != ACCELERATOR_EVIDENCE_SCHEMA:
            raise ContractError("accelerator evidence schema is unsupported")
        if evidence.get("passed") is not True:
            raise ContractError("accelerator conformance did not pass")
        if evidence.get("device_class") not in {"cuda", "mps"}:
            raise ContractError(
                "accelerator evidence must name a cuda or mps device class"
            )
        if evidence.get("policy_digest") != policy["policy_digest"]:
            raise ContractError("accelerator evidence uses a different policy")
        if (
            evidence.get("graph_digest") != graph_digest
            or evidence.get("training_digest") != training_digest
        ):
            raise ContractError("accelerator evidence graph/training identity differs")
        gates["accelerator_conformance"] = True
    except (ContractError, OSError) as exc:
        failures.append(f"accelerator_conformance: {exc}")
    return PreflightReport(
        policy["policy_id"], policy["policy_digest"], gates, tuple(failures)
    )
