"""Phase identity, allocation and consumer boundaries; not executed pre-commit."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from pingstore import stages
from pingstore.contracts import STAGE_ID_RE, PingstoreError, validate_compute_scope


def test_phase_counter_is_shared_with_all_other_stages(tmp_path):
    requests = [
        ("compute", "snntorch"),
        ("compute", "snnlab"),
        ("compute", None),
        ("analyse", None),
    ] * 3
    with ThreadPoolExecutor(max_workers=4) as pool:
        identities = list(
            pool.map(
                lambda row: stages.reserve_stage(
                    tmp_path, "exp126", row[0], phase=row[1]
                ),
                requests,
            )
        )
    counters = [
        int(STAGE_ID_RE.fullmatch(identity).group(2)) for identity in identities
    ]
    assert len(set(counters)) == len(requests)
    for identity, (stage, phase) in zip(identities, requests, strict=True):
        reservation = stages.stage_reservation(tmp_path / "runs" / f".{identity}.tmp")
        assert reservation.get("phase") == phase
        if stage == "compute":
            assert reservation["scope"] == ("phase" if phase else "full")


@pytest.mark.parametrize(
    "record",
    [
        {
            "run_id": "exp126-r001-compute-phase-snntorch",
            "stage": "compute",
            "scope": "full",
        },
        {
            "run_id": "exp126-r001-compute-phase-snntorch",
            "stage": "compute",
            "scope": "phase",
            "phase": "snnlab",
        },
        {
            "run_id": "exp126-r001-compute",
            "stage": "compute",
            "scope": "phase",
            "phase": "snntorch",
        },
        {"run_id": "exp126-r001-analyse", "stage": "analyse", "scope": "full"},
    ],
)
def test_phase_scope_and_identity_must_agree(record):
    with pytest.raises(PingstoreError):
        validate_compute_scope(record)


def test_legacy_compute_scope_is_not_inferred_or_rewritten():
    record = {"run_id": "exp126-r001-compute", "stage": "compute"}
    before = dict(record)
    validate_compute_scope(record)
    assert record == before


def test_ordinary_consumer_rejects_phase(tmp_path, monkeypatch):
    record = {
        "run_id": "exp126-r001-compute-phase-snntorch",
        "stage": "compute",
        "experiment": "exp126",
        "scope": "phase",
        "phase": "snntorch",
    }
    monkeypatch.setattr(stages, "validate_operational_run_directory", lambda _: record)
    with pytest.raises(PingstoreError, match="phase-aware consumer"):
        stages.source_run(tmp_path, record["run_id"], stage="compute")
    assert (
        stages.source_run(
            tmp_path, record["run_id"], stage="compute", phase="snntorch"
        ).record
        == record
    )
