"""Measure completed exp023 snapshots; no simulation, drawing or publication."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp023 import inputs, recipe
from pingstore.contracts import PingstoreError, write_json_atomic
from pingstore.stages import stage_run


def population_psd(spk_2d: np.ndarray, dt_ms: float, band_hz):
    """Welch periodogram on the population-mean spike trace, matching
    exp041 / exp049: one window per trial (nperseg = T), density scaling,
    mean-subtracted input. Returns (freqs_hz, psd, f_peak_hz_or_None)
    with parabolic-interpolated peak frequency inside the gamma band."""
    from scipy import signal as sp_signal

    T, N = spk_2d.shape
    if T < 2 or N == 0:
        return np.array([0.0]), np.array([0.0]), None
    x = spk_2d.mean(axis=1).astype(np.float64)
    x = x - x.mean()
    fs = 1000.0 / dt_ms
    freqs, psd = sp_signal.welch(x, fs=fs, nperseg=T, scaling="density")
    band = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if not band.any() or psd[band].max() == 0 or not np.isfinite(psd[band]).any():
        return freqs, psd, None
    abs_idx = int(np.where(band)[0][int(np.argmax(psd[band]))])
    if 0 < abs_idx < len(psd) - 1:
        y0, y1, y2 = psd[abs_idx - 1], psd[abs_idx], psd[abs_idx + 1]
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        delta = float(max(-0.5, min(0.5, delta)))
    else:
        delta = 0.0
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0
    f_peak = float(freqs[abs_idx] + delta * df)
    return freqs, psd, f_peak


def snapshot(path: Path, cfg: dict, point: dict, *, traces: bool = False) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.array(archive[key]) for key in archive.files}
    dt = float(data["dt"])
    steps = int(round(point["t_ms"] / point["dt_ms"]))
    if not np.isfinite(dt) or not np.isclose(dt, point["dt_ms"]):
        raise PingstoreError(f"{path}: timestep differs from retained recipe")
    if not traces and "spk_e_count" in data:
        for population in ("e", "i"):
            count = data.get(f"spk_{population}_count")
            if (
                count is None
                or count.shape != ()
                or count.dtype.kind not in "iu"
                or not 0 <= count.item() <= steps * cfg[f"n_{population}"]
            ):
                raise PingstoreError(f"{path}: invalid population spike count")
        if any(
            np.asarray(data.get(key)).shape != () or data.get(key) != value
            for key, value in (("T", steps), ("n_e", cfg["n_e"]), ("n_i", cfg["n_i"]))
        ):
            raise PingstoreError(f"{path}: count dimensions differ from recipe")
        data["dt"] = dt
        return data
    for key, n in (
        ("spk_e", cfg["n_e"]),
        ("spk_i", cfg["n_i"]),
        ("input_spikes", point["n_in"]),
    ):
        if key == "input_spikes" and key not in data:
            continue
        spikes = data[key]
        if spikes.shape != (steps, n) or not np.isin(spikes, [0, 1]).all():
            raise PingstoreError(f"{path}: invalid {key} geometry or spikes")
    if traces and "v_e_selected" in data:
        for population in ("e", "i"):
            index = pick_active(data[f"spk_{population}"])
            expected_index = index if index is not None else 0
            if (
                np.asarray(data.get(f"{population}_trace_index")).shape != ()
                or data.get(f"{population}_trace_index") != expected_index
            ):
                raise PingstoreError(
                    f"{path}: trace selection differs from spike counts"
                )
            signals = ("v", "ge", "gi") if population == "e" else ("v", "ge")
            for signal in signals:
                value = data.get(f"{signal}_{population}_selected")
                if (
                    value is None
                    or value.shape != (steps,)
                    or not np.isfinite(value).all()
                ):
                    raise PingstoreError(f"{path}: missing or invalid selected trace")
                if signal.startswith("g") and (value < 0).any():
                    raise PingstoreError(f"{path}: negative conductance")
        flag = data.get("has_gi_e")
        if flag is None or flag.shape != () or flag.dtype.kind != "b":
            raise PingstoreError(f"{path}: invalid inhibitory conductance flag")
    elif traces:
        for key, n in (
            ("v_e_1", cfg["n_e"]),
            ("ge_e_1", cfg["n_e"]),
            ("gi_e_1", cfg["n_e"]),
            ("v_i_1", cfg["n_i"]),
            ("ge_i_1", cfg["n_i"]),
        ):
            value = data.get(key)
            if (
                value is None
                or value.shape != (steps, n)
                or not np.isfinite(value).all()
            ):
                raise PingstoreError(f"{path}: missing or invalid {key}")
            if key.startswith("g") and (value < 0).any():
                raise PingstoreError(f"{path}: negative conductance")
    data["dt"] = dt
    return data


def population_rate(spikes: np.ndarray, dt: float) -> float:
    steps, neurons = spikes.shape
    return float(spikes.sum() / (neurons * steps * dt / 1000.0)) if neurons else 0.0


def pick_active(spikes: np.ndarray) -> int | None:
    counts = spikes.sum(axis=0)
    return int(np.argmax(counts)) if counts.size and np.any(counts > 0) else None


def select_traces(data: dict, biophysics: dict) -> tuple[dict, dict]:
    e_index = pick_active(data["spk_e"])
    i_index = pick_active(data["spk_i"])
    selected = {
        "e_index": e_index if e_index is not None else 0,
        "i_index": i_index,
        "e_active": e_index is not None,
        "has_gi_e": bool(data["has_gi_e"])
        if "has_gi_e" in data
        else bool(data["gi_e_1"].any()),
    }
    values = {"time_ms": np.arange(data["spk_e"].shape[0]) * data["dt"]}
    for population, index in (("e", selected["e_index"]), ("i", i_index)):
        if index is None:
            continue

        def trace(signal):
            key = f"{signal}_{population}_selected"
            return (
                data[key] if key in data else data[f"{signal}_{population}_1"][:, index]
            )

        v, ge = trace("v"), trace("ge")
        gi = trace("gi") if population == "e" else np.zeros_like(ge)
        gl = biophysics[f"g_L_{population.upper()}_uS"]
        values.update(
            {
                f"v_{population}": v,
                f"ge_{population}": ge,
                f"gi_{population}": gi,
                f"ie_{population}": -ge * (v - biophysics["E_E_mV"]),
                f"ii_{population}": -gi * (v - biophysics["E_I_mV"]),
                f"il_{population}": -gl * (v - biophysics["E_L_mV"]),
            }
        )
    return values, selected


def analyse(identity: str, *, run_id: str | None = None) -> str:
    compute = inputs.source(REPO, identity, "compute")
    cfg = inputs.configuration(compute)
    measurement = {
        "schema": "exp023.measurement/v1",
        "frequency_band_hz": list(recipe.F_GAMMA_BAND_HZ),
        "spectrum": "Welch density, mean-subtracted population spikes, nperseg=trial length",
        "peak": "band argmax with clamped parabolic interpolation",
        "reported_peak": "only when inhibitory spikes are present; not a rhythmicity test",
        "rate": "all spikes / (population size * full duration in seconds)",
        "trace_selection": "first maximum spike-count cell; E index zero if silent; omit silent I",
    }
    with stage_run(
        REPO,
        recipe.SLUG,
        "analyse",
        inputs={"compute": compute},
        run_id=run_id,
        configuration=measurement,
    ) as run:
        spectra, traces, raster, peaks = {}, {}, {}, {}
        for cell in cfg["cells"]:
            point = cfg["drive"]["raster_operating_points"][cell]
            data = snapshot(
                compute.file("scope", cell, "snapshot.npz"),
                cfg,
                point,
                traces=True,
            )
            frequencies, density, peak = population_psd(
                data["spk_e"], data["dt"], measurement["frequency_band_hz"]
            )
            peaks[cell] = (
                float(peak) if data["spk_i"].any() and peak is not None else None
            )
            spectra[f"{cell}__frequency_hz"] = frequencies
            spectra[f"{cell}__density"] = density
            trace_values, selected = select_traces(data, cfg["biophysics"])
            traces.update(
                {f"{cell}__{key}": value for key, value in trace_values.items()}
            )
            raster[cell] = {
                **selected,
                "e_rate_hz": population_rate(data["spk_e"], data["dt"]),
                "i_rate_hz": population_rate(data["spk_i"], data["dt"]),
                "duration_ms": data["spk_e"].shape[0] * data["dt"],
            }
        fi_point = cfg["drive"]["fi_sweep"]
        fi = {cell: {"in": [], "e": [], "i": []} for cell in cfg["cells"]}
        for cell in cfg["cells"]:
            for rate in fi_point["input_rates_hz"]:
                data = snapshot(
                    compute.file("fi", f"{cell}__r{rate}", "snapshot.npz"),
                    cfg,
                    fi_point,
                )
                fi[cell]["in"].append(rate)
                for population in ("e", "i"):
                    key = f"spk_{population}_count"
                    if key in data:
                        neurons = cfg[f"n_{population}"]
                        rate_hz = (
                            float(
                                data[key]
                                / (neurons * int(data["T"]) * data["dt"] / 1000.0)
                            )
                            if neurons
                            else 0.0
                        )
                    else:
                        rate_hz = population_rate(data[f"spk_{population}"], data["dt"])
                    fi[cell][population].append(rate_hz)
        np.savez_compressed(run.export / "spectra.npz", **spectra)
        np.savez_compressed(run.export / "traces.npz", **traces)
        write_json_atomic(
            run.export / "results.json",
            {
                "schema": "exp023.analysis/v1",
                "config": cfg,
                "measurement": measurement,
                "raster": raster,
                "f_gamma_hz": peaks,
                "fi_curves": fi,
            },
        )
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="completed exp023 compute run ID"
    )
    parser.add_argument("--run-id", help="unused v3 identity reserved before dispatch")
    args = parser.parse_args()
    analyse(args.source, run_id=args.run_id)


if __name__ == "__main__":
    main()
