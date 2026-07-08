"""Tests for the runner-side provenance helper (RULES §4.7)."""
import os
from datetime import datetime

from experiments.helpers import provenance


def test_run_provenance_shape():
    prov = provenance.run_provenance()
    assert set(prov) == {"commit", "dirty", "generated_at"}
    assert isinstance(prov["dirty"], bool)
    # commit is a SHA inside a repo, None outside one.
    assert prov["commit"] is None or isinstance(prov["commit"], str)


def test_generated_at_is_utc_iso():
    ts = provenance.run_provenance()["generated_at"]
    parsed = datetime.fromisoformat(ts)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0


def test_stamp_adds_provenance_without_mutating():
    config = {"seed": 0, "n": 100}
    stamped = provenance.stamp(config)
    assert "_provenance" not in config  # input untouched
    assert stamped["seed"] == 0 and stamped["n"] == 100
    assert set(stamped["_provenance"]) == {"commit", "dirty", "generated_at"}


def test_write_run_sh(tmp_path):
    path = provenance.write_run_sh(tmp_path)
    assert path == tmp_path / "run.sh"
    assert os.access(path, os.X_OK)
    lines = path.read_text().splitlines()
    assert lines[0] == "#!/bin/sh"
    assert lines[-1].startswith("exec uv run python ")
