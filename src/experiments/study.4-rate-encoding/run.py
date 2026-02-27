import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from pinglab.analysis import decode_fit_metrics, lowpass_first_order, minmax_normalize
from pinglab.io import compile_graph_to_runtime, layer_bounds_from_spec
from pinglab.backends.pytorch import simulate_network
from pinglab.plots.raster import save_raster
from pinglab.io import linspace_from_scan, overwrite_spec_value

from plots import (
    save_decoded_envelopes_plot,
    save_input_signal_plot,
    save_layer_population_rates_plot,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import ARTIFACTS_ROOT


def _population_rate_for_range(
    *,
    spike_times_ms: np.ndarray,
    spike_ids: np.ndarray,
    start_id: int,
    stop_id: int,
    T_ms: float,
    dt_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_bins = int(np.ceil(T_ms / dt_ms))
    edges = np.linspace(0.0, T_ms, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    n_neurons = max(0, int(stop_id) - int(start_id))
    if n_neurons <= 0:
        return centers, np.zeros_like(centers, dtype=float)

    mask = (spike_ids >= int(start_id)) & (spike_ids < int(stop_id))
    counts, _ = np.histogram(spike_times_ms[mask], bins=edges)
    rate_hz = counts / (n_neurons * (dt_ms / 1000.0))
    return centers, rate_hz.astype(float, copy=False)


def _best_lag_ms(reference: np.ndarray, decoded: np.ndarray, dt_ms: float, max_lag_ms: float = 300.0) -> float:
    x = np.asarray(reference, dtype=float)
    y = np.asarray(decoded, dtype=float)
    n = min(x.size, y.size)
    if n < 3:
        return 0.0
    x = x[:n] - float(np.mean(x[:n]))
    y = y[:n] - float(np.mean(y[:n]))
    max_lag_bins = int(np.floor(max_lag_ms / dt_ms))
    max_lag_bins = min(max_lag_bins, n - 1)
    lags = np.arange(-max_lag_bins, max_lag_bins + 1, dtype=int)
    best_lag = 0
    best_score = -1.0
    for lag in lags:
        if lag < 0:
            a = x[-lag:]
            b = y[: n + lag]
        elif lag > 0:
            a = x[: n - lag]
            b = y[lag:]
        else:
            a = x
            b = y
        if a.size < 3 or b.size < 3:
            continue
        denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
        if denom <= 1e-12:
            continue
        score = float(np.sum(a * b) / denom)
        if score > best_score:
            best_score = score
            best_lag = int(lag)
    return float(best_lag) * float(dt_ms)


def main() -> None:
    data_path = ARTIFACTS_ROOT / Path(__file__).parent.name
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    config_path = Path(__file__).parent / "config.json"
    with config_path.open(encoding="utf-8") as f:
        spec = json.load(f)
    shutil.copy2(config_path, data_path / "config.json")

    sim = dict(spec.get("sim", {}))
    meta = dict(spec.get("meta", {}))
    msg = dict(meta.get("message", {}))
    scan = dict(meta.get("scan", {}))
    scan_id = str(scan.get("scan_id", "ff_sweep"))
    scan_values = linspace_from_scan(scan)
    scan_param = str(scan.get("parameter", "e1_to_e2.w.mean"))

    dt_ms = float(sim.get("dt_ms", 0.1))
    t_end_ms = float(sim.get("T_ms", 1000.0))
    t_ms = np.arange(0.0, t_end_ms, dt_ms, dtype=float)
    t_s = t_ms / 1000.0

    message_freq_hz = float(msg.get("message_freq_hz", 3.0))
    message_offset = float(msg.get("message_offset", 1.0))
    message_amplitude = float(msg.get("message_amplitude", 0.5))

    message = message_offset + message_amplitude * np.sin(2.0 * np.pi * message_freq_hz * t_s)
    input_current = message.copy()

    save_input_signal_plot(
        data_path / f"input_row-1_{scan_id}_signal-components.png",
        t_ms=t_ms,
        message=message,
        input_current=input_current,
    )

    payload = {
        "t_ms": [float(v) for v in t_ms],
        "message": [float(v) for v in message],
        "input_current": [float(v) for v in input_current],
    }
    (data_path / f"input_row-0_{scan_id}_signal.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    layer_labels = ["E1", "E2", "E3"]
    per_layer_scan_metrics = {
        label: {"corr": [], "rmse": [], "lag_ms": [], "gain": []} for label in layer_labels
    }
    point_metrics: list[dict[str, float | dict[str, float]]] = []

    best_idx = 0
    best_e3_corr = -1.0

    for idx, scan_value in enumerate(scan_values):
        spec_i = overwrite_spec_value(spec, scan_param, float(scan_value))
        spec_i = overwrite_spec_value(spec_i, "e2_to_e3.w.mean", float(scan_value))

        runtime = compile_graph_to_runtime(spec_i, backend="pytorch")
        config = runtime.config

        layer_bounds = layer_bounds_from_spec(spec_i)
        bounds_by_label = {label: (start, stop) for start, stop, label in layer_bounds}
        e1_start, e1_stop = bounds_by_label["E1"]

        external_input = runtime.external_input.detach().clone()
        external_input[:, int(e1_start):int(e1_stop)] = (
            external_input.new_tensor(input_current)[:, None]
        )
        runtime_i = replace(runtime, external_input=external_input)

        result = simulate_network(runtime_i)

        rate_dt_ms = 5.0
        t_rate_ms, e1_rate = _population_rate_for_range(
            spike_times_ms=np.asarray(result.spikes.times),
            spike_ids=np.asarray(result.spikes.ids),
            start_id=int(e1_start),
            stop_id=int(e1_stop),
            T_ms=float(config.T),
            dt_ms=rate_dt_ms,
        )
        e2_start, e2_stop = bounds_by_label["E2"]
        _, e2_rate = _population_rate_for_range(
            spike_times_ms=np.asarray(result.spikes.times),
            spike_ids=np.asarray(result.spikes.ids),
            start_id=int(e2_start),
            stop_id=int(e2_stop),
            T_ms=float(config.T),
            dt_ms=rate_dt_ms,
        )
        e3_start, e3_stop = bounds_by_label["E3"]
        _, e3_rate = _population_rate_for_range(
            spike_times_ms=np.asarray(result.spikes.times),
            spike_ids=np.asarray(result.spikes.ids),
            start_id=int(e3_start),
            stop_id=int(e3_stop),
            T_ms=float(config.T),
            dt_ms=rate_dt_ms,
        )
        rates = {"E1": e1_rate, "E2": e2_rate, "E3": e3_rate}

        decode_cutoff_hz = max(0.5, 2.0 * message_freq_hz)
        message_at_rate = np.interp(t_rate_ms, t_ms, message, left=message[0], right=message[-1])
        ref_norm = minmax_normalize(message_at_rate)
        point_payload: dict[str, float | dict[str, float]] = {"scan_value": float(scan_value)}
        for label in layer_labels:
            decoded = lowpass_first_order(rates[label], rate_dt_ms, decode_cutoff_hz)
            dec_norm = minmax_normalize(decoded)
            c, r = decode_fit_metrics(dec_norm, ref_norm, normalize=False)
            lag = _best_lag_ms(ref_norm, dec_norm, rate_dt_ms)
            ref_std = float(np.std(ref_norm))
            dec_std = float(np.std(dec_norm))
            g = float(dec_std / ref_std) if ref_std > 1e-12 else 0.0
            per_layer_scan_metrics[label]["corr"].append(float(c))
            per_layer_scan_metrics[label]["rmse"].append(float(r))
            per_layer_scan_metrics[label]["lag_ms"].append(float(lag))
            per_layer_scan_metrics[label]["gain"].append(float(g))
            point_payload[label] = {"corr": float(c), "rmse": float(r), "lag_ms": float(lag), "gain": float(g)}
        point_metrics.append(point_payload)

        current_e3_corr = float(per_layer_scan_metrics["E3"]["corr"][-1])
        if current_e3_corr > best_e3_corr:
            best_e3_corr = current_e3_corr
            best_idx = idx

        print(
            f"[{idx + 1}/{len(scan_values)}] ff.mean={float(scan_value):.4f} "
            f"E3 corr={current_e3_corr:.3f}"
        )

    best_scan_value = float(scan_values[best_idx])
    best_spec = overwrite_spec_value(spec, scan_param, best_scan_value)
    best_spec = overwrite_spec_value(best_spec, "e2_to_e3.w.mean", best_scan_value)
    best_runtime = compile_graph_to_runtime(best_spec, backend="pytorch")
    best_config = best_runtime.config
    best_layer_bounds = layer_bounds_from_spec(best_spec)
    best_bounds_by_label = {label: (start, stop) for start, stop, label in best_layer_bounds}
    best_e1_start, best_e1_stop = best_bounds_by_label["E1"]
    best_external_input = best_runtime.external_input.detach().clone()
    best_external_input[:, int(best_e1_start):int(best_e1_stop)] = (
        best_external_input.new_tensor(input_current)[:, None]
    )
    best_runtime_i = replace(best_runtime, external_input=best_external_input)
    best_result = simulate_network(best_runtime_i)
    best_t_rate_ms, best_e1_rate = _population_rate_for_range(
        spike_times_ms=np.asarray(best_result.spikes.times),
        spike_ids=np.asarray(best_result.spikes.ids),
        start_id=int(best_e1_start),
        stop_id=int(best_e1_stop),
        T_ms=float(best_config.T),
        dt_ms=rate_dt_ms,
    )
    best_e2_start, best_e2_stop = best_bounds_by_label["E2"]
    _, best_e2_rate = _population_rate_for_range(
        spike_times_ms=np.asarray(best_result.spikes.times),
        spike_ids=np.asarray(best_result.spikes.ids),
        start_id=int(best_e2_start),
        stop_id=int(best_e2_stop),
        T_ms=float(best_config.T),
        dt_ms=rate_dt_ms,
    )
    best_e3_start, best_e3_stop = best_bounds_by_label["E3"]
    _, best_e3_rate = _population_rate_for_range(
        spike_times_ms=np.asarray(best_result.spikes.times),
        spike_ids=np.asarray(best_result.spikes.ids),
        start_id=int(best_e3_start),
        stop_id=int(best_e3_stop),
        T_ms=float(best_config.T),
        dt_ms=rate_dt_ms,
    )
    best_rates = {"E1": best_e1_rate, "E2": best_e2_rate, "E3": best_e3_rate}
    best_message_at_rate = np.interp(best_t_rate_ms, t_ms, message, left=message[0], right=message[-1])
    best_ref_norm = minmax_normalize(best_message_at_rate)
    best_decoded = {
        label: minmax_normalize(lowpass_first_order(rate, rate_dt_ms, decode_cutoff_hz))
        for label, rate in best_rates.items()
    }

    save_raster(
        best_result.spikes,
        data_path / f"raster_row-2_{scan_id}_layers",
        external_input=best_external_input.detach().cpu().numpy(),
        dt=float(best_config.dt),
        label=f"Best ff.mean={best_scan_value:.3f} (E3 corr={best_e3_corr:.3f})",
        xlim=(0.0, float(best_config.T)),
        layer_bounds=best_layer_bounds,
        layer_order=["I1", "I2", "I3", "E1", "E2", "E3"],
    )
    save_layer_population_rates_plot(
        data_path / f"rate_row-3_{scan_id}_layers",
        t_ms=best_t_rate_ms,
        rates_by_layer=best_rates,
    )
    save_decoded_envelopes_plot(
        data_path / f"decode_row-4_{scan_id}_envelopes",
        t_ms=best_t_rate_ms,
        message_ref=best_ref_norm,
        decoded_by_layer=best_decoded,
    )
    metrics_payload = {
        "scan_values": [float(v) for v in scan_values],
        "layers": layer_labels,
        "by_layer": per_layer_scan_metrics,
        "points": point_metrics,
    }
    (data_path / f"metrics_row-0_{scan_id}_fidelity.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )
    rows = ["scan_value,layer,corr,rmse,lag_ms,gain_ratio"]
    for i, scan_value in enumerate(scan_values):
        for label in layer_labels:
            rows.append(
                f"{float(scan_value):.9f},{label},"
                f"{per_layer_scan_metrics[label]['corr'][i]:.9f},"
                f"{per_layer_scan_metrics[label]['rmse'][i]:.9f},"
                f"{per_layer_scan_metrics[label]['lag_ms'][i]:.9f},"
                f"{per_layer_scan_metrics[label]['gain'][i]:.9f}"
            )
    (data_path / f"metrics_row-0_{scan_id}_fidelity.csv").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )

    print(
        "saved input + sweep + best-point plots to "
        f"{data_path} (best ff.mean={best_scan_value:.4f}, E3 corr={best_e3_corr:.3f})"
    )


if __name__ == "__main__":
    main()
