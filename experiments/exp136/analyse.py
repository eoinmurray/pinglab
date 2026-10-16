"""Measure PING timing and held-out sensory-signal reconstruction."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import numpy as np
from experiments.exp136 import recipe
from pingstore.contracts import write_json_atomic
from pingstore.stages import source_run, stage_run
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def load_spikes(source, condition, cfg):
    with np.load(source.file(f"{condition}--spikes.npz"), allow_pickle=False) as raw:
        values = {key: raw[key].copy() for key in raw.files}
    expected = {
        f"{name}_{pop}_spikes": count
        for name in (*cfg.get("senders", ["S1", "S2"]), "R")
        for pop, count in (("E", recipe.N_E), ("I", recipe.N_I))
    }
    expected.update({f"cue_{name}_spikes": recipe.N_CUE for name in ("S1", "S2")})
    steps = round(cfg["duration_ms"] / cfg["dt_ms"])
    if set(values) != set(expected):
        raise ValueError("missing or unexpected spike outputs")
    for key, count in expected.items():
        value = values[key]
        if (
            value.shape != (steps, 1, count)
            or value.dtype != np.uint8
            or not np.isin(value, [0, 1]).all()
        ):
            raise ValueError(f"invalid spike output: {key}")
    return values


def volley_phase(spikes, cfg):
    """Interpolate observed volleys, leaving long gaps and edges undefined."""
    dt, settings = cfg["dt_ms"], cfg["analysis"]
    rate = gaussian_filter1d(
        spikes.mean(axis=1) * 1000 / dt, settings["smoothing_ms"] / dt
    )
    peaks, _ = find_peaks(
        rate,
        distance=round(settings["peak_distance_ms"] / dt),
        prominence=settings["prominence_fraction"] * rate.max(),
    )
    time = np.arange(len(spikes)) * dt
    phase = np.full(len(spikes), np.nan)
    if len(peaks) >= 2:
        phase = np.interp(
            time, peaks * dt, np.arange(len(peaks)), left=np.nan, right=np.nan
        )
        for left, right in zip(peaks[:-1], peaks[1:]):
            if (right - left) * dt > settings["max_phase_gap_ms"]:
                phase[left + 1 : right] = np.nan
    return phase, peaks * dt


def sliding_plv(difference, cfg):
    """Time-average phase vectors within complete, centred windows."""
    dt, settings = cfg["dt_ms"], cfg["analysis"]
    width = round(settings["plv_window_ms"] / dt)
    stride = round(settings["plv_step_ms"] / dt)
    starts = np.arange(0, len(difference) - width + 1, stride)
    values = np.full(len(starts), np.nan)
    for i, start in enumerate(starts):
        window = difference[start : start + width]
        finite = np.isfinite(window)
        if finite.mean() >= settings["plv_min_coverage"]:
            values[i] = np.clip(abs(np.exp(2j * np.pi * window[finite]).mean()), 0, 1)
    return (starts + width / 2) * dt, values


def rhythm_summary(peaks, start, end, cfg):
    """Mean inter-volley frequency within one explicit half-open time window."""
    settings = cfg["analysis"]
    selected = peaks[(peaks >= start) & (peaks < end)]
    intervals = np.diff(selected)
    frequency = float(1000 / intervals.mean()) if len(intervals) else None
    cv = float(intervals.std() / intervals.mean()) if len(intervals) else None
    return dict(
        start_ms=start,
        end_ms=end,
        frequency_hz=frequency,
        iei_cv=cv,
        volleys=len(selected),
        valid=bool(
            len(selected) >= settings["min_volleys"]
            and frequency is not None
            and settings["gamma_hz"][0] <= frequency <= settings["gamma_hz"][1]
            and cv <= settings["max_iei_cv"]
        ),
    )


def window_metrics(time, phases, peaks, spikes, start, end, cfg):
    mask = (time >= start) & (time < end)
    result = dict(start_ms=start, end_ms=end, rhythms={}, pairs={})
    for name in (*cfg.get("senders", ["S1", "S2"]), "R"):
        result["rhythms"][name] = rhythm_summary(peaks[name], start, end, cfg)
    for name in cfg.get("senders", ["S1", "S2"]):
        difference = phases[name] - phases["R"]
        selected = mask & np.isfinite(difference)
        coverage = float(selected.sum() / mask.sum())
        enough = selected.sum() >= 100
        d, t = difference[selected], time[selected] / 1000
        result["pairs"][name + "R"] = dict(
            valid=bool(
                enough
                and coverage >= 0.9
                and result["rhythms"][name]["valid"]
                and result["rhythms"]["R"]["valid"]
            ),
            phase_coverage=coverage,
            plv=float(abs(np.mean(np.exp(2j * np.pi * d)))) if enough else None,
            drift_cycles_s=float(np.polyfit(t, d, 1)[0]) if enough else None,
            phase_span_cycles=float(np.ptp(d)) if enough else None,
        )
    result["rates_hz"] = {
        key: float(value[mask].mean() * 1000 / cfg["dt_ms"])
        for key, value in spikes.items()
    }
    return result


def analyse(identity, *, run_id=None, task=None):
    if task == "readout":
        from experiments.exp136.readout_audit import analyse_readouts

        return analyse_readouts(REPO, identity, run_id=run_id)
    source = source_run(
        REPO / ".pingstore", identity, stage="compute", experiment=recipe.SLUG
    )
    cfg = source.record["execution"]["configuration"]
    if cfg.get("schema") == "exp136.receiver-refine/v1":
        from experiments.exp136.receiver_refine import analyse_refinement

        return analyse_refinement(REPO, identity, run_id=run_id)
    if cfg.get("schema") == "exp136.routing-search/v1":
        from experiments.exp136.routing_search import analyse_search

        return analyse_search(REPO, identity, run_id=run_id)
    if cfg.get("schema") == "exp136.attention/v1":
        from experiments.exp136.attention_analyse import analyse_attention

        return analyse_attention(REPO, identity, run_id=run_id)
    if cfg.get("schema") != "exp136.fixed/v1" or not source.record["execution"].get(
        "paired_parameters_and_inputs_verified"
    ):
        raise ValueError("requires a verified fixed-setting compute run")
    time = np.arange(round(cfg["duration_ms"] / cfg["dt_ms"])) * cfg["dt_ms"]
    result = dict(schema="exp136.fixed-analysis/v1", configuration=cfg, conditions={})
    with stage_run(
        REPO,
        recipe.SLUG,
        "analyse",
        inputs={"compute": source},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        for condition in cfg["conditions"]:
            spikes = load_spikes(source, condition, cfg)
            phases, peaks = {}, {}
            for name in ("S1", "S2", "R"):
                phases[name], peaks[name] = volley_phase(
                    spikes[f"{name}_E_spikes"][:, 0], cfg
                )
            result["conditions"][condition] = dict(
                windows={
                    label: window_metrics(time, phases, peaks, spikes, start, end, cfg)
                    for label, start, end in cfg["analysis_windows"]
                },
                frequencies={
                    name: [
                        rhythm_summary(peaks[name], start, start + 1000, cfg)
                        for start in range(0, round(cfg["duration_ms"]), 1000)
                    ]
                    for name in phases
                },
            )
            plv = {}
            for name in ("S1", "S2"):
                plv["time_ms"], plv[f"{name}R"] = sliding_plv(
                    phases[name] - phases["R"], cfg
                )
            np.savez_compressed(run.export / f"{condition}--plv.npz", **plv)
        write_json_atomic(run.export / "results.json", result)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--task", choices=("readout",))
    args = parser.parse_args()
    analyse(args.source, run_id=args.run_id, task=args.task)
