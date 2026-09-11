"""Lossless numerical storage and validation of retained scientific evidence."""

import copy
import math

import numpy as np
from pingstore.contracts import (
    PingstoreError,
    load_json,
    write_json_atomic,
)

from . import measurements, recipe


def exact_values(a, b):
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and set(a) == set(b)
            and all(exact_values(a[k], b[k]) for k in a)
        )
    if isinstance(a, (list, tuple)):
        return (
            isinstance(b, (list, tuple))
            and len(a) == len(b)
            and all(exact_values(x, y) for x, y in zip(a, b))
        )
    if isinstance(a, float):
        return isinstance(b, (int, float)) and a.hex() == float(b).hex()
    return a == b


def validate_summary(numbers, subset):
    r = numbers["results"]
    cfg = recipe.configuration(version=1)
    if numbers.get("slug") != recipe.SLUG or numbers["config"] != {
        k: cfg[k] for k in numbers["config"]
    }:
        raise PingstoreError("retained exp033 configuration differs")
    if set(numbers["config"]) != {
        "tau_E_ms",
        "tau_I_ms",
        "tau_AMPA_ms",
        "tau_GABA_ms",
        "W_tilde_EI",
        "W_tilde_IE",
        "dV_inh_mV",
        "dV_exc_mV",
        "sigma_V_mV",
        "cell_E",
        "cell_I",
    }:
        raise PingstoreError("retained exp033 configuration is incomplete")
    for key in ("hopf", "criticality"):
        if not exact_values(subset[key], r[key]):
            raise PingstoreError("exp054 cache disagrees with exp033 " + key)
    if not exact_values(
        subset["frequency_vs_tau_gaba"], r["frequency_vs_tau_gaba"]["mean_field"]
    ) or not exact_values(
        subset["spiking_exp041"], r["frequency_vs_tau_gaba"]["spiking_exp041"]
    ):
        raise PingstoreError("retained frequency overlays disagree")
    if len(subset["sweep"]) != 401:
        raise PingstoreError("retained sweep must contain all 401 points")
    measurements.validate_continuation(
        {"sweep": subset["sweep"], "hopf": subset["hopf"]}, np.linspace(0, 4, 401)
    )
    if [row["sigma_V_mV"] for row in r["sigma_sensitivity"]["rows"]] != list(
        recipe.SIGMA_V_GRID_MV
    ):
        raise PingstoreError("retained noise sensitivity is incomplete")


def amplitude_summary(criticality, onset):
    """Re-evaluate the original estimator from all retained branch amplitudes."""
    up, down = criticality["up"], criticality["down"]
    expected = np.linspace(onset - 0.1, onset + 0.55, 25).tolist()
    if any([r["I_ext"] for r in branch] != expected for branch in (up, down)):
        raise PingstoreError("retained hysteresis grid is incomplete")
    if any(
        not np.isfinite(row["amp"]) or row["amp"] < 0
        for branch in (up, down)
        for row in branch
    ):
        raise PingstoreError("invalid retained amplitudes")
    gap = float(max(abs(d["amp"] - u["amp"]) for u, d in zip(up, down)))
    on = next((u["I_ext"] for u in up if u["amp"] > 1e-4), None)
    off = next((d["I_ext"] for d in down if d["amp"] > 1e-4), None)
    above = [(u["I_ext"] - onset, u["amp"]) for u in up if u["I_ext"] > onset + 1e-9]
    x = np.array([a[0] for a in above])
    ysq = np.array([a[1] ** 2 for a in above])
    m, c = np.polyfit(x, ysq, 1)
    ss_res = float(np.sum((ysq - (m * x + c)) ** 2))
    ss_tot = float(np.sum((ysq - ysq.mean()) ** 2))
    slope = float(m)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "up": copy.deepcopy(up),
        "down": copy.deepcopy(down),
        "hyst_gap": gap,
        "hyst_width_nA": float(on - off)
        if on is not None and off is not None
        else None,
        "A2_slope": slope,
        "A2_r2": r2,
        "verdict": "supercritical"
        if gap < 1e-4 and slope > 0 and r2 > 0.9
        else "subcritical/inconclusive",
    }


def verify_amplitudes(criticality, onset):
    measured = amplitude_summary(criticality, onset)
    for key, value in measured.items():
        if key in ("A2_slope", "A2_r2"):
            if not np.isclose(value, criticality[key], rtol=1e-12, atol=1e-15):
                raise PingstoreError("retained regression does not reproduce")
        elif not exact_values(value, criticality[key]):
            raise PingstoreError("retained hysteresis evidence disagrees")
    return measured


def write(directory, document):
    arrays = {}

    def pack(value):
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                raise PingstoreError("exp033 arrays must be finite real numbers")
            name = f"a{len(arrays):04d}"
            arrays[name] = value
            return {"array": name}
        if isinstance(value, dict):
            return {k: pack(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [pack(v) for v in value]
        if isinstance(value, np.generic):
            return pack(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            raise PingstoreError("exp033 numerical scalars must be finite")
        return value

    index = pack(document)
    np.savez_compressed(directory / "arrays.npz", **arrays)
    write_json_atomic(directory / "evidence.json", index)


def read(directory):
    index = load_json(directory / "evidence.json")
    used = set()
    with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:

        def unpack(value):
            if isinstance(value, dict) and set(value) == {"array"}:
                name = value["array"]
                if name not in arrays:
                    raise PingstoreError("missing exp033 numerical array")
                result = arrays[name]
                if result.dtype.kind not in "biuf" or not np.isfinite(result).all():
                    raise PingstoreError("exp033 arrays must be finite real numbers")
                used.add(name)
                return result
            if isinstance(value, dict):
                return {k: unpack(v) for k, v in value.items()}
            if isinstance(value, list):
                return [unpack(v) for v in value]
            if isinstance(value, float) and not math.isfinite(value):
                raise PingstoreError("exp033 numerical scalars must be finite")
            return value

        result = unpack(index)
        if used != set(arrays.files):
            raise PingstoreError("unreferenced exp033 numerical arrays")
    return result
