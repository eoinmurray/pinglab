import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from pinglab.analysis import population_rate
from pinglab.io import compile_graph_to_runtime
from pinglab.backends.pytorch import simulate_network
from pinglab.plots.raster import save_raster
from plots import (
    save_input_mean_vs_neuron_plot,
    save_neuron_phase_vs_id_plot,
    save_true_vs_decoded_input_plot,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import ARTIFACTS_ROOT


def _runtime_layer_bounds_from_spec(spec: dict) -> list[tuple[int, int, str]]:
    nodes = spec.get("nodes", [])
    e_nodes = [n for n in nodes if n.get("kind") == "population" and str(n.get("type")) == "E"]
    i_nodes = [n for n in nodes if n.get("kind") == "population" and str(n.get("type")) == "I"]
    ordered = e_nodes + i_nodes
    bounds: list[tuple[int, int, str]] = []
    cursor = 0
    for node in ordered:
        size = int(node.get("size", 0))
        label = str(node.get("id"))
        bounds.append((cursor, cursor + size, label))
        cursor += size
    return bounds


def _i_peak_times_ms(
    *,
    spikes,
    T_ms: float,
    n_e: int,
    n_i: int,
    bin_ms: float = 2.0,
) -> np.ndarray:
    t_ms, i_rate = population_rate(
        spikes=spikes,
        T_ms=T_ms,
        dt_ms=bin_ms,
        pop="I",
        N_E=n_e,
        N_I=n_i,
        smooth_sigma_ms=None,
    )
    if i_rate.size < 3:
        return np.zeros(0, dtype=float)
    rate_max = float(np.max(i_rate))
    if rate_max <= 0.0:
        return np.zeros(0, dtype=float)
    thr = max(0.25 * rate_max, float(np.mean(i_rate)) + 0.5 * float(np.std(i_rate)))
    min_sep_bins = max(1, int(np.floor(10.0 / bin_ms)))
    peaks, _ = find_peaks(i_rate, height=thr, distance=min_sep_bins)
    if peaks.size == 0:
        return np.zeros(0, dtype=float)
    return t_ms[peaks].astype(float, copy=False)


def _mean_phase_by_neuron(
    *,
    spike_times_ms: np.ndarray,
    spike_ids: np.ndarray,
    e_start: int,
    e_stop: int,
    peak_times_ms: np.ndarray,
) -> np.ndarray:
    n_e = int(e_stop - e_start)
    out = np.full(n_e, np.nan, dtype=float)
    if n_e <= 0 or peak_times_ms.size < 2 or spike_times_ms.size == 0:
        return out
    e_mask = (spike_ids >= int(e_start)) & (spike_ids < int(e_stop))
    t_e = spike_times_ms[e_mask].astype(float, copy=False)
    id_e = spike_ids[e_mask].astype(int, copy=False)
    if t_e.size == 0:
        return out
    cyc = np.searchsorted(peak_times_ms, t_e, side="right") - 1
    valid = (cyc >= 0) & (cyc < (peak_times_ms.size - 1))
    t_e = t_e[valid]
    id_e = id_e[valid]
    cyc = cyc[valid]
    if t_e.size == 0:
        return out
    t0 = peak_times_ms[cyc]
    t1 = peak_times_ms[cyc + 1]
    dur = t1 - t0
    valid_dur = dur > 0.0
    t_e = t_e[valid_dur]
    id_e = id_e[valid_dur]
    t0 = t0[valid_dur]
    dur = dur[valid_dur]
    phase = (t_e - t0) / dur
    for nid in range(int(e_start), int(e_stop)):
        nm = id_e == nid
        if np.any(nm):
            out[nid - int(e_start)] = float(np.mean(phase[nm]))
    return out


def _decode_tonic_from_phase(
    *,
    mean_phase: np.ndarray,
    true_input: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    decoded = np.full_like(true_input, np.nan, dtype=float)
    valid = np.isfinite(mean_phase) & np.isfinite(true_input)
    if np.sum(valid) < 2:
        return decoded, {"valid_points": float(np.sum(valid))}

    phase_v = mean_phase[valid].astype(float)
    true_v = true_input[valid].astype(float)

    # Phase is circular. Search phase shifts and polarity, then decode with rank mapping.
    best_abs_r = -np.inf
    best_feat = None
    best_shift = 0.0
    best_raw_r = np.nan
    for shift in np.linspace(0.0, 1.0, 72, endpoint=False):
        feat = np.mod(phase_v + shift, 1.0)
        r = float(np.corrcoef(true_v, feat)[0, 1])
        if np.isnan(r):
            continue
        if abs(r) > best_abs_r:
            best_abs_r = abs(r)
            best_raw_r = r
            best_feat = feat if r >= 0.0 else (1.0 - feat)
            best_shift = float(shift)

    if best_feat is None:
        return decoded, {"valid_points": float(np.sum(valid))}

    n = best_feat.size
    order = np.argsort(best_feat, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    if n > 1:
        rank_norm = ranks / float(n - 1)
    else:
        rank_norm = np.zeros_like(ranks)
    lo = float(np.min(true_v))
    hi = float(np.max(true_v))
    decoded[valid] = lo + rank_norm * (hi - lo)
    final_r = float(np.corrcoef(true_v, decoded[valid])[0, 1])
    return decoded, {
        "valid_points": float(np.sum(valid)),
        "best_shift": best_shift,
        "best_abs_r_raw_phase": float(best_abs_r),
        "best_r_raw_phase": float(best_raw_r),
        "final_r_decoded": final_r,
    }


def main() -> None:
    data_path = ARTIFACTS_ROOT / Path(__file__).parent.name
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    config_path = Path(__file__).parent / "config.json"
    with config_path.open(encoding="utf-8") as f:
        spec = json.load(f)
    shutil.copy2(config_path, data_path / "config.json")
    scan_id = str(spec.get("meta", {}).get("scan_id", "pingloop"))
    grad_cfg = dict(spec.get("meta", {}).get("input_gradient", {}))
    min_add = float(grad_cfg.get("min_add", 0.0))
    max_add = float(grad_cfg.get("max_add", 1.2))

    runtime = compile_graph_to_runtime(spec, backend="pytorch")
    config = runtime.config
    external_input = runtime.external_input.detach().clone()
    layer_bounds = _runtime_layer_bounds_from_spec(spec)
    by_label = {label: (start, stop) for start, stop, label in layer_bounds}
    e_start, e_stop = by_label["E"]

    # Experiment-local spatial gradient: higher E neuron index gets higher tonic offset.
    n_e = int(e_stop - e_start)
    gradient = np.linspace(min_add, max_add, n_e, dtype=float)
    external_input[:, int(e_start):int(e_stop)] += external_input.new_tensor(gradient)[None, :]

    e_ids = np.arange(n_e, dtype=int)
    e_input_mean = (
        external_input[:, int(e_start):int(e_stop)].mean(dim=0).detach().cpu().numpy()
    )
    save_input_mean_vs_neuron_plot(
        data_path / f"input_row-2_{scan_id}_tonic-mean-vs-neuron.png",
        neuron_ids=e_ids,
        input_mean=e_input_mean,
    )

    runtime_i = replace(runtime, external_input=external_input)
    result = simulate_network(runtime_i)
    i_peak_times = _i_peak_times_ms(
        spikes=result.spikes,
        T_ms=float(config.T),
        n_e=int(config.N_E),
        n_i=int(config.N_I),
        bin_ms=2.0,
    )

    save_raster(
        result.spikes,
        data_path / f"raster_row-1_{scan_id}_single-loop",
        label="Raster | single PING loop",
        vertical_lines=i_peak_times.tolist(),
        vertical_line_kwargs={"linestyle": "-", "lw": 0.8, "alpha": 0.10, "color": "gray"},
        external_input=external_input.detach().cpu().numpy(),
        dt=float(config.dt),
        xlim=(0.0, float(config.T)),
        layer_bounds=layer_bounds,
        layer_order=["I", "E"],
    )
    spike_times = np.asarray(result.spikes.times, dtype=float)
    spike_ids = np.asarray(result.spikes.ids, dtype=int)
    mean_phase = _mean_phase_by_neuron(
        spike_times_ms=spike_times,
        spike_ids=spike_ids,
        e_start=int(e_start),
        e_stop=int(e_stop),
        peak_times_ms=i_peak_times,
    )
    save_neuron_phase_vs_id_plot(
        data_path / f"phase_row-3_{scan_id}_e-neuron-vs-mean-phase.png",
        neuron_ids=e_ids,
        mean_phase=mean_phase,
        input_mean=e_input_mean,
    )

    decoded, decode_dbg = _decode_tonic_from_phase(
        mean_phase=mean_phase,
        true_input=e_input_mean,
    )
    print(f"[decode] stats={decode_dbg}")
    save_true_vs_decoded_input_plot(
        data_path / f"decode_row-4_{scan_id}_true-vs-decoded-tonic.png",
        true_input=e_input_mean,
        decoded_input=decoded,
    )
    print(f"saved raster to {data_path}")


if __name__ == "__main__":
    main()
