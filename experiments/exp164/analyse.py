"""Analyse saved constant-cue prefixes; reconstruct linear conductances offline."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
from importlib.metadata import version
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, lfilter
from experiments.exp164 import recipe
from pingstore.library import source_run, stage_run, write_json_atomic
from snnlab.sim import NetworkSnapshot, plan_graph

PATHS = recipe.PATHS


def reconstruct(spikes, weights, *, decay, delay_steps):
    """Mean recipient conductance: causal delayed impulses, then linear filtering."""
    if not 0 <= decay < 1 or delay_steps < 0:
        raise ValueError("invalid native synapse decay or delay")
    spikes = np.asarray(spikes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if spikes.shape[1] != weights.shape[0]:
        raise ValueError("presynaptic cells do not match native weights")
    impulse = spikes @ weights.mean(axis=1)
    delayed = np.zeros_like(impulse)
    if delay_steps == 0:
        delayed[:] = impulse
    elif delay_steps < len(impulse):
        delayed[delay_steps:] = impulse[:-delay_steps]
    return lfilter([1.0], [1.0, -decay], delayed)


def prepare(source, settings):
    end_ms = settings["display_end_ms"]
    window_ms = settings["window_ms"]
    closeup_ms = settings["closeup_ms"]
    cfg = source.record["execution"]["configuration"]
    if source.record["execution"]["snnlab_version"] != version("snnlab"):
        raise ValueError("installed native planner differs from source SNNLab version")
    if not 0 < window_ms[0] < window_ms[1] <= end_ms < cfg["switch_ms"]:
        raise ValueError("all displayed and measurement times must precede the switch")
    if not window_ms[0] <= closeup_ms[0] < closeup_ms[1] <= window_ms[1]:
        raise ValueError("close-up must lie inside the measurement window")
    dt = cfg["dt_ms"]
    steps = round(cfg["switch_ms"] / dt)
    time = np.arange(steps) * dt
    with np.load(source.file("spikes.npz")) as d:
        spikes = {k: v[:steps].copy() for k, v in d.items()}
    cases = cfg["conditions"]
    indices = {
        c: next(
            i
            for i, x in enumerate(cases)
            if x["present"] == "AB" and x["order"][0] == c and not x["probe"]
        )
        for c in "AB"
    }
    with np.load(source.file("inputs.npz")) as d:
        for name in ["X_A", "X_B", "X_C"]:
            np.testing.assert_array_equal(
                d[name][:steps, indices["A"]], d[name][:steps, indices["B"]]
            )
        for c in "AB":
            assert not d["U_" + ("B" if c == "A" else "A")][:steps, indices[c]].any()
            assert d["U_" + c][:steps, indices[c]].any()
    graph = json.loads(source.file("network.bundle/graph.json").read_text())
    plan = plan_graph(graph)
    projections = {p.id: p for p in plan.projections}
    graph_paths = {p["id"]: p for p in graph["projections"]}
    native = NetworkSnapshot.read(
        source.file("execution/start/snapshot.json")
    ).load_parameters()
    measured = (time >= window_ms[0]) & (time < window_ms[1])
    shown = time <= end_ms
    results = {}
    for cued, j in indices.items():
        rates = {
            c: gaussian_filter1d(
                spikes[c + "_E"][:, j].mean(1) * 1000 / dt,
                settings["rate_smoothing_sd_ms"] / dt,
            )
            for c in "ABC"
        }
        phases = {}
        for c, y in rates.items():
            peaks, _ = find_peaks(
                y,
                distance=round(settings["min_peak_distance_ms"] / dt),
                prominence=max(
                    settings["min_prominence_hz"],
                    settings["prominence_fraction"]
                    * np.ptp(y[time > settings["reference_start_ms"]]),
                ),
            )
            if len(peaks) < 2:
                raise ValueError("insufficient gamma peaks")
            phases[c] = np.interp(
                time, time[peaks], np.arange(len(peaks)), left=np.nan, right=np.nan
            )
        conductances = {}
        for name in PATHS:
            p = projections[name]
            g = graph_paths[name]
            assert (
                p.enabled
                and not g.get("modulation")
                and g["synapse"]["kind"] in ["ampa", "gaba"]
            )
            owner = p.source.split(".")[0]
            conductances[name] = reconstruct(
                spikes[owner][:, j],
                native[p.parameter].numpy(),
                decay=p.decay,
                delay_steps=p.delay_steps,
            )
        maps = {
            c: spikes[c + "_E"][measured, j].mean(0).reshape(8, 8) * 1000 / dt
            for c in "ABC"
        }
        differences = {c: phases[c] - phases["C"] for c in "AB"}
        plvs = {}
        for c in "AB":
            valid = measured & np.isfinite(differences[c])
            plvs[c] = float(abs(np.exp(2j * np.pi * differences[c][valid]).mean()))
        results[cued] = dict(
            rates=rates,
            differences=differences,
            conductances=conductances,
            maps=maps,
            plvs=plvs,
            case=cases[j]["name"],
        )
    excitation_peak = max(
        r["conductances"][p][measured].max()
        for r in results.values()
        for p in PATHS[:2]
    )
    inhibition_peak = max(
        r["conductances"][PATHS[2]][measured].max() for r in results.values()
    )
    inhib = np.concatenate(
        [r["conductances"][PATHS[2]][measured] for r in results.values()]
    )
    scales = dict(
        excitation_peak_us=float(excitation_peak),
        inhibition_peak_us=float(inhibition_peak),
        low_inhibition_threshold_us=float(
            inhib.min() + settings["low_inhibition_fraction"] * np.ptp(inhib)
        ),
        map_max_hz={
            c: float(max(r["maps"][c].max() for r in results.values())) for c in "ABC"
        },
        rate_max_hz=float(
            max(r["rates"][c][shown].max() for r in results.values() for c in "ABC")
            * 1.06
        ),
    )
    finite = np.concatenate(
        [
            r["differences"][c][shown & np.isfinite(r["differences"][c])]
            for r in results.values()
            for c in "AB"
        ]
    )
    scales["phase_limits_cycles"] = [
        float(finite.min() - 0.2),
        float(finite.max() + 0.2),
    ]
    return cfg, time, results, scales


def analyse(*, source, run_id=None):
    started = perf_counter()
    computed = source_run(
        REPO / ".pingstore", source, experiment=recipe.SLUG, stage="compute"
    )
    settings = recipe.configuration(
        source_configuration=computed.record["execution"]["configuration"]
    )
    cfg, time, results, scales = prepare(computed, settings)
    numbers = dict(
        schema=recipe.SCHEMA,
        configuration=settings,
        labels=cfg["labels"],
        plvs={c: r["plvs"] for c, r in results.items()},
        scales=scales,
        cue_conditions={c: r["case"] for c, r in results.items()},
    )
    with stage_run(
        REPO,
        recipe.SLUG,
        "analyse",
        inputs={"compute": computed},
        configuration=settings,
        run_id=run_id,
    ) as run:
        np.savez_compressed(
            run.export / "traces.npz",
            time_ms=time,
            **{
                f"{cue}_{kind}_{name}": value
                for cue, row in results.items()
                for kind in ["rates", "differences", "conductances", "maps"]
                for name, value in row[kind].items()
            },
        )
        write_json_atomic(run.export / "numbers.json", numbers)
        run.record["execution"].update(
            elapsed_seconds=perf_counter() - started, no_simulation=True
        )
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--run-id")
    analyse(**vars(p.parse_args()))
