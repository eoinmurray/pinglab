"""Use explicitly pinned historical measurements without inventing solver trajectories."""

import json
import zipfile

import numpy as np
from experiments.exp033 import historical as upstream_history
from experiments.exp033 import measurements as upstream_measurements
from pingstore.contracts import PingstoreError, load_json


def mean_field(source, frequencies):
    if source.record["execution"]["operation"] != "historical-import":
        raise PingstoreError(
            "exp054 historical theory requires retained exp033 import evidence"
        )
    old = load_json(source.export / "historical-numbers.json")
    with zipfile.ZipFile(source.export / "mean-field.zip") as archive:
        if archive.namelist() != ["numerical-evidence.json"]:
            raise PingstoreError("unexpected retained exp054 mean-field entries")
        subset = json.loads(archive.read("numerical-evidence.json"))
    upstream_history.validate_summary(old, subset)
    info = source.record.get("historical_import", {})
    producer = info.get("cache_producer", {})
    if producer.get("experiment") != "exp054" or producer.get("job_id") != "33913631":
        raise PingstoreError("retained mean-field producer is not historical exp054")
    if source.record["inputs"].get("frequencies") != frequencies.reference:
        raise PingstoreError(
            "historical exp054 theory has different frequency ancestry"
        )
    medians = upstream_measurements.spiking_medians(
        load_json(frequencies.export / "results.json")
    )
    deltas = {str(k): v - subset["spiking_exp041"][str(k)] for k, v in medians.items()}
    if deltas != info.get("frequency_deltas_hz"):
        raise PingstoreError("historical exp054 frequency delta differs")
    recomputed = upstream_history.verify_amplitudes(
        subset["criticality"], subset["hopf"]["I_ext_star"]
    )
    return subset, {
        "source": source.reference,
        "original_producer": producer,
        "frequency_deltas_hz": deltas,
        "amplitude_recheck": recomputed,
        "policy": "retain historical scalars and overlay; no missing trajectories regenerated",
    }


def compare_numbers(current, original):
    differences = {}

    def compare(a, b, path):
        if isinstance(a, dict):
            if not isinstance(b, dict) or set(a) != set(b):
                raise PingstoreError("exp054 historical fields differ: " + path)
            for key in a:
                compare(a[key], b[key], path + "/" + key)
        elif isinstance(a, list):
            if not isinstance(b, list) or len(a) != len(b):
                raise PingstoreError("exp054 historical dimensions differ: " + path)
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                compare(x, y, path + "/" + str(i))
        elif a != b:
            # FFT roundoff may differ across platforms; rates and recipe remain exact.
            is_contrast = "contrast" in path or path.endswith("null_max")
            if not (
                is_contrast
                and isinstance(a, (int, float))
                and isinstance(b, (int, float))
                and np.isfinite(a)
                and np.isfinite(b)
                and np.isclose(a, b, rtol=1e-12, atol=1e-15)
            ):
                raise PingstoreError(
                    "exp054 retained recordings do not reproduce historical " + path
                )
            differences[path] = a - b

    for key in ("config", "grid", "rate_invariance"):
        compare(current[key], original[key], key)
    return {
        "contrast_tolerance": {"rtol": 1e-12, "atol": 1e-15},
        "contrast_deltas": differences,
        "maximum_absolute_contrast_delta": max(
            map(abs, differences.values()), default=0.0
        ),
        "rates_and_configuration_exact": True,
        "policy": "preserve original scalars; retain independent numerical recheck",
    }
