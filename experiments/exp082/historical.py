"""Explicit historical aggregate evidence; never a fallback for new compute."""

import math

from pingstore.contracts import PingstoreError, file_sha256, load_json

from . import recipe

SCHEMA = "exp082.gold2-import/v1"
PRODUCER = "73f0883edc14aa634f5a6d55e4f4123fbfeb7508"


def aggregate(path, job, cfg):
    row = load_json(path)
    keys = {
        "seed",
        "duration_ms",
        "rate_hz",
        "stream_batch_size",
        "n_correct",
        "n_total",
        "accuracy",
        "output_spikes_per_presentation",
        "silent_fraction",
        "class_spike_totals",
        "rate_e_hz",
        "rate_i_hz",
    }
    if set(row) != keys or any(
        row[k] != job[k] for k in ("seed", "duration_ms", "rate_hz")
    ):
        raise PingstoreError("historical condition identity differs")
    total, correct = row["n_total"], row["n_correct"]
    if (
        type(total) is not int
        or total != cfg["digits_per_seed_cell"]
        or type(correct) is not int
        or not 0 <= correct <= total
        or row["stream_batch_size"] != cfg["stream_batch_size"]
        or row["accuracy"] != correct / total
    ):
        raise PingstoreError("invalid historical condition totals")
    spikes = row["class_spike_totals"]
    if (
        len(spikes) != 10
        or any(type(v) is not int or v < 0 for v in spikes)
        or row["output_spikes_per_presentation"] != sum(spikes) / total
    ):
        raise PingstoreError("invalid historical output counts")
    silent = row["silent_fraction"]
    if (
        not isinstance(silent, (int, float))
        or not 0 <= silent <= 1
        or not math.isclose(silent * total, round(silent * total), abs_tol=1e-9)
    ):
        raise PingstoreError("invalid historical silent fraction")
    for key in ("rate_e_hz", "rate_i_hz"):
        if (
            not isinstance(row[key], (float, int))
            or not math.isfinite(row[key])
            or row[key] < 0
        ):
            raise PingstoreError("invalid historical population rate")
    return row


def validate_import(run, cfg):
    record = run.record.get("historical_import", {})
    if (
        run.record["execution"].get("operation") != "historical-import"
        or record.get("schema") != SCHEMA
        or record.get("producer_commit") != PRODUCER
        or cfg != recipe.configuration()
    ):
        raise PingstoreError(
            "aggregate evidence requires the explicit historical import contract"
        )
    proof = load_json(run.directory / "provenance/import.json")
    if (
        proof.get("schema") != SCHEMA
        or proof.get("source_files") != 199
        or proof.get("source_bytes") != 6079619
    ):
        raise PingstoreError("historical import selection differs")
    mappings = proof.get("files", [])
    if len(mappings) != 199 or len({m["source"] for m in mappings}) != 199:
        raise PingstoreError("incomplete historical source mapping")
    for item in mappings:
        target = run.directory / item["target"]
        if not target.is_relative_to(run.directory) or ".." in target.parts:
            raise PingstoreError("unsafe historical source mapping")
        if (
            target.stat().st_size != item["size_bytes"]
            or file_sha256(target) != item["sha256"]
        ):
            raise PingstoreError("historical source mapping checksum differs")
    old = load_json(
        run.directory / "provenance/archive/derived/artifacts/data/exp082/numbers.json"
    )
    rows = [
        aggregate(run.export / j["path"] / "condition.json", j, cfg)
        for j in recipe.jobs(cfg)
    ]

    def by_key(r):
        return r["duration_ms"], r["rate_hz"], r["seed"]

    if sorted(rows, key=by_key) != sorted(old["grid_per_seed"], key=by_key):
        raise PingstoreError(
            "historical aggregate rows differ from retained source summary"
        )
