"""Delayed rate-conductance counterpart to the saved exp133 paired circuit.

The source bundle and effective sampled matrices remain authoritative. This
module defines one unfitted closure and a coupling-onset example;
it neither selects runs nor executes integration.
"""

import math

from snnlab import meanfield as mf
from snnlab.meanfield import closures

SLUG = "exp153"
SNAPSHOT = "coupling-120-execution/start"


def configuration(
    source_configuration,
    incoming_weights_us,
    *,
    difference_hz=10.0,
    coupling_us=0.08,
    rate_scale_b=0.8,
):
    """Record imported physics separately from unfitted closure choices."""
    source = source_configuration
    if source.get("schema") != "exp133.recipe/v1":
        raise ValueError("requires an exp133 grid compute, not the onset diagnostic")
    if source["base_coupling_us"] != 0.12 or source["duration_ms"] != 5000:
        raise ValueError("requires the 0.12 µS reference and 5,000 ms grid protocol")
    if not incoming_weights_us or any(
        not math.isfinite(value) or value <= 0 for value in incoming_weights_us.values()
    ):
        raise ValueError("mean incoming effective weights must be finite and positive")
    if (
        not math.isfinite(difference_hz)
        or abs(difference_hz) >= 2 * source["mean_input_hz"]
    ):
        raise ValueError("input difference must leave both drives positive")
    if (
        not math.isfinite(coupling_us)
        or not 0 < coupling_us <= source["base_coupling_us"]
    ):
        raise ValueError("coupling must be positive and within the reference range")
    if not math.isfinite(rate_scale_b) or rate_scale_b <= 0:
        raise ValueError("B rate-relaxation scale must be finite and positive")
    return dict(
        schema="exp153.recipe/v2",
        protocol="phase-trace",
        imported_protocol=source,
        duration_ms=2500.0,
        example=dict(
            difference_hz=float(difference_hz),
            coupling_us=float(coupling_us),
            rate_scale_b=float(rate_scale_b),
            rationale="B rate relaxation scaled to increase intrinsic frequency mismatch; one uncoupled control and one coupling-onset trajectory",
        ),
        incoming_weights_us=dict(incoming_weights_us),
        imported_snapshot=SNAPSHOT + "/snapshot.json",
        weight_interpretation="sum(effective W)/N_target, including zero edges; native scales already applied, including input gain; no second scaling",
        closure=dict(
            name="snnlab.meanfield.closures.RateConductance",
            sigma_mv=4.0,
            tau_rate_ms={
                f"PING_{circuit}_{population}": (20.0 if population == "E" else 5.0)
                * (rate_scale_b if circuit == "B" else 1.0)
                for circuit in ("A", "B")
                for population in ("E", "I")
            },
            quadrature_nodes=48,
            provenance="Unfitted rate-conductance closure; B relaxation deliberately scaled to exaggerate detuning, not inferred from exp133 spikes",
            external_filters="dynamic",
            delays="explicit",
            modulation="translate",
        ),
        omitted_effects=[
            "input covariance and finite-population fluctuations",
            "within-population weight heterogeneity",
            "voltage-dependent driving force and shunting",
        ],
        initial=dict(rate_hz=1.0, conductance_us=0.0, history="constant"),
        solver=dict(method="DOP853", rtol=1e-8, atol=1e-10, max_step_ms=0.1),
        sample_interval_ms=0.5,
        gate=dict(
            windows_ms=[[800.0, 1600.0], [1600.0, 2500.0]],
            min_amplitude_hz=0.5,
            min_relative_amplitude=0.01,
            min_late_to_early_amplitude=0.5,
            min_peaks=15,
            max_interval_cv=0.2,
            frequency_band_hz=None,
            interpretation="Check sustained rhythms at the actual example inputs in both trajectories",
        ),
        phase=dict(
            start_ms=source["onset_ms"] + source["settling_ms"],
            min_peak_distance_ms=source["volley_distance_ms"],
            prominence_fraction=source["volley_prominence_fraction"],
            min_amplitude_hz=0.5,
            min_peaks=15,
            max_interval_cv=source["max_iei_cv"],
            frequency_band_hz=None,
            **source["classification"],
            smoothing="none; deterministic population rates, not binned spikes",
        ),
    )


def author_network(bundle):
    """Retain the authenticated source graph, without sampling new matrices."""
    populations = {p["id"] for p in bundle.graph["populations"]}
    expected = {f"PING_{c}_{p}" for c in ("A", "B") for p in ("E", "I")}
    if populations != expected or len(bundle.graph["projections"]) != 10:
        raise ValueError("expected exp133's four populations and ten projections")
    for p in bundle.graph["projections"]:
        cross = "_K_E" in p["id"]
        expected_delay = 2 if cross else (0 if "input_to_E" in p["id"] else 0.1)
        if (p.get("delay") or {}).get("value", 0) != expected_delay:
            raise ValueError("source graph delays differ from exp133")
        if cross and p.get("modulation") != dict(
            kind="step_gain", at_ms=500.0, before=0.0, after=1.0
        ):
            raise ValueError("expected reciprocal coupling onset at 500 ms")
    return bundle


def build_model(bundle, cfg, *, difference_hz=0.0, coupling_us=0.0):
    """Scale only cross-circuit reference sums; keep imported input gain once."""
    source = cfg["imported_protocol"]
    if (
        not math.isfinite(difference_hz)
        or abs(difference_hz) >= 2 * source["mean_input_hz"]
    ):
        raise ValueError("input difference must leave both drives positive")
    if not 0 <= coupling_us <= source["base_coupling_us"]:
        raise ValueError("coupling outside the recorded grid range")
    weights = {
        name: value * coupling_us / source["base_coupling_us"]
        if "_K_E" in name
        else value
        for name, value in cfg["incoming_weights_us"].items()
    }
    closure = cfg["closure"]
    return mf.reduce(
        author_network(bundle),
        closure=closures.RateConductance(
            gain=closures.NoisyLIF(
                sigma=closure["sigma_mv"],
                quadrature_nodes=closure["quadrature_nodes"],
            ),
            tau_rate=closure["tau_rate_ms"],
        ),
        weights=mf.MeanIncomingWeights(
            weights, provenance=cfg["weight_interpretation"]
        ),
        inputs={
            "drive_A": mf.StationaryRate(source["mean_input_hz"] + difference_hz / 2),
            "drive_B": mf.StationaryRate(source["mean_input_hz"] - difference_hz / 2),
        },
        external_filters=closure["external_filters"],
        delays=closure["delays"],
        modulation=closure["modulation"],
    )


def initial_state(model, cfg):
    return {
        name: cfg["initial"]["rate_hz" if name.endswith(".rate") else "conductance_us"]
        for name in model.states
    }
