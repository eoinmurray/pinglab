import shutil
import sys
import json
from pathlib import Path

import numpy as np
from pinglab.analysis import (
    autocorr_peak,
    coherence_peak,
    first_significant_peak_lag_ms,
    lagged_coherence,
    mean_pairwise_xcorr_peak,
    total_e_spikes,
)
from pinglab.io import compile_graph_to_runtime
from pinglab.backends.pytorch import simulate_network
from pinglab.plots.raster import save_raster
from pinglab.io import collect_scans, linspace_from_scan, overwrite_spec_value, scan_variant
from pinglab.plots.rhythmicity import (
    normalized_e_rate_trace,
    save_autocorr_curve_plot,
    save_ee_autocorr_heatmap,
    save_e_rate_plot,
    save_scan_metrics_plot,
    save_stacked_autocorr_curves_plot,
    save_stacked_e_rates_plot,
    save_total_e_spikes_vs_parameter_plot,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import ARTIFACTS_ROOT


def main() -> None:
    corr_min_lag_ms = 20.0
    corr_max_lag_ms = 150.0

    data_path = ARTIFACTS_ROOT / Path(__file__).parent.name
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    config_path = Path(__file__).parent / "config.json"
    with config_path.open(encoding="utf-8") as f:
        spec = json.load(f)
    shutil.copy2(config_path, data_path / "config.json")

    scans = collect_scans(spec.get("meta", {}))
    if not scans:
        raise ValueError("meta must include at least one scan object: 'scan' or 'scan_*'")
    scan_by_parameter = {
        str(scan_cfg.get("parameter", "")): (scan_name, scan_cfg)
        for scan_name, scan_cfg in scans
    }

    for scan_name, scan in scans:
        parameter = str(scan.get("parameter", ""))
        values = linspace_from_scan(scan)
        param_key = parameter.replace(".", "_")
        scan_key = scan_name.replace(".", "_")
        variant = scan_variant(parameter, scan_key)
        prefix = f"{variant}_{scan_key}"
        stacked_rates: list[np.ndarray] = []
        stacked_labels: list[str] = []
        stacked_t_ms: np.ndarray | None = None
        autocorr_curves: list[np.ndarray] = []
        autocorr_lags_ms_ref: np.ndarray | None = None
        autocorr_peak_lags_ms: list[float] = []
        autocorr_peak_values: list[float] = []
        total_e_spikes_counts: list[int] = []
        metrics: dict[str, list[float]] = {
            "lagged_coherence": [],
            "mean_pairwise_xcorr_peak": [],
            "coherence_peak": [],
            "autocorr_peak": [],
        }

        for idx, scan_value in enumerate(values):
            spec_i = overwrite_spec_value(spec, parameter, float(scan_value))

            runtime = compile_graph_to_runtime(spec_i, backend="pytorch")
            config = runtime["config"]
            external_input = runtime["external_input"]
            result = simulate_network(runtime)

            save_raster(
                result.spikes,
                data_path / f"raster_{prefix}_{idx:02d}_{param_key}_{scan_value:.6f}",
                label=f"{parameter} {scan_value:.6f}",
                external_input=external_input.detach().cpu().numpy(),
                dt=config.dt,
                xlim=(0.0, config.T),
            )
            save_e_rate_plot(
                spikes=result.spikes,
                T_ms=float(config.T),
                N_E=int(config.N_E),
                parameter=f"{scan_name}.{parameter}",
                scan_value=float(scan_value),
                out_path=data_path / f"rate_e_{prefix}_{idx:02d}_{param_key}_{scan_value:.6f}.png",
            )
            t_ms, norm_rate = normalized_e_rate_trace(
                spikes=result.spikes,
                T_ms=float(config.T),
                N_E=int(config.N_E),
            )
            lagged_value, *_ = lagged_coherence(
                result.spikes,
                T_ms=float(config.T),
                pop="E",
                N_E=int(config.N_E),
                N_I=int(config.N_I),
            )
            xcorr_peak_value, *_ = mean_pairwise_xcorr_peak(
                result.spikes,
                T_ms=float(config.T),
                N_E=int(config.N_E),
            )
            coherence_value, *_ = coherence_peak(
                result.spikes,
                T_ms=float(config.T),
                N_E=int(config.N_E),
            )
            autocorr_value, _, _, autocorr_lags_ms, autocorr_curve = autocorr_peak(
                result.spikes,
                T_ms=float(config.T),
                pop="E",
                N_E=int(config.N_E),
                N_I=int(config.N_I),
            )
            metrics["lagged_coherence"].append(float(lagged_value))
            metrics["mean_pairwise_xcorr_peak"].append(float(xcorr_peak_value))
            metrics["coherence_peak"].append(float(coherence_value))
            metrics["autocorr_peak"].append(float(autocorr_value))
            if stacked_t_ms is None:
                stacked_t_ms = t_ms
            stacked_rates.append(norm_rate)
            stacked_labels.append(f"{scan_value:.6f}")
            if autocorr_lags_ms_ref is None:
                autocorr_lags_ms_ref = autocorr_lags_ms
            autocorr_curves.append(autocorr_curve)
            peak_lag_ms = first_significant_peak_lag_ms(
                autocorr_lags_ms,
                autocorr_curve,
                corr_min_lag_ms=corr_min_lag_ms,
                corr_max_lag_ms=corr_max_lag_ms,
            )
            autocorr_peak_lags_ms.append(peak_lag_ms)
            autocorr_peak_values.append(float(autocorr_value))
            total_e_spikes_counts.append(total_e_spikes(result.spikes, int(config.N_E)))
            save_autocorr_curve_plot(
                lags_ms=autocorr_lags_ms,
                corr=autocorr_curve,
                parameter=f"{scan_name}.{parameter}",
                scan_value=float(scan_value),
                peak_lag_ms=peak_lag_ms,
                peak_value=float(autocorr_value),
                out_path=data_path / f"autocorr_curve_{prefix}_{idx:02d}_{param_key}_{scan_value:.6f}.png",
            )
            print(
                f"[{scan_name} {idx + 1}/{len(values)}] "
                f"{parameter}={scan_value:.6f} -> raster+E-rate+metrics saved"
            )

        if stacked_t_ms is not None:
            save_stacked_e_rates_plot(
                traces=stacked_rates,
                labels=stacked_labels,
                t_ms=stacked_t_ms,
                parameter=f"{scan_name}.{parameter}",
                out_path=data_path / f"stacked_rate_e_{prefix}_{param_key}.png",
            )
        save_scan_metrics_plot(
            scan_values=values,
            metrics={"autocorr_peak": metrics["autocorr_peak"]},
            parameter=f"{scan_name}.{parameter}",
            out_path=data_path / f"metrics_vs_{prefix}_{param_key}.png",
        )
        save_total_e_spikes_vs_parameter_plot(
            scan_values=values,
            total_e_spikes=total_e_spikes_counts,
            parameter=f"{scan_name}.{parameter}",
            out_path=data_path / f"e_spikes_vs_{prefix}_{param_key}.png",
        )
        if autocorr_lags_ms_ref is not None:
            save_stacked_autocorr_curves_plot(
                lags_ms=autocorr_lags_ms_ref,
                curves=autocorr_curves,
                labels=stacked_labels,
                peak_lags_ms=autocorr_peak_lags_ms,
                peak_values=autocorr_peak_values,
                processing_window_ms=(corr_min_lag_ms, corr_max_lag_ms),
                parameter=f"{scan_name}.{parameter}",
                out_path=data_path / f"stacked_autocorr_{prefix}_{param_key}.png",
            )

        metrics_summary = {
            "scan": scan_name,
            "parameter": parameter,
            "scan_values": [float(v) for v in values],
            "total_e_spikes": [int(v) for v in total_e_spikes_counts],
            **{key: [float(v) for v in vals] for key, vals in metrics.items()},
        }
        (data_path / f"metrics_summary_{prefix}.json").write_text(
            json.dumps(metrics_summary, indent=2),
            encoding="utf-8",
        )
        rows = [
            "scan_value,lagged_coherence,mean_pairwise_xcorr_peak,coherence_peak,autocorr_peak"
        ]
        for i, scan_value in enumerate(values):
            rows.append(
                f"{float(scan_value):.9f},"
                f"{metrics['lagged_coherence'][i]:.9f},"
                f"{metrics['mean_pairwise_xcorr_peak'][i]:.9f},"
                f"{metrics['coherence_peak'][i]:.9f},"
                f"{metrics['autocorr_peak'][i]:.9f}"
            )
        (data_path / f"metrics_summary_{prefix}.csv").write_text(
            "\n".join(rows) + "\n",
            encoding="utf-8",
        )
        autocorr_curves_payload = {
            "scan": scan_name,
            "parameter": parameter,
            "scan_values": [float(v) for v in values],
            "lags_ms": (
                [float(v) for v in autocorr_lags_ms_ref]
                if autocorr_lags_ms_ref is not None
                else []
            ),
            "curves": [[float(x) for x in curve] for curve in autocorr_curves],
            "peak_lags_ms": [float(v) for v in autocorr_peak_lags_ms],
            "peak_values": [float(v) for v in autocorr_peak_values],
        }
        (data_path / f"autocorr_curves_{prefix}.json").write_text(
            json.dumps(autocorr_curves_payload, indent=2),
            encoding="utf-8",
        )

    mean_scan = scan_by_parameter.get("e_to_e.w.mean")
    std_scan = scan_by_parameter.get("e_to_e.w.std")
    if mean_scan is not None and std_scan is not None:
        _, mean_scan_cfg = mean_scan
        _, std_scan_cfg = std_scan
        mean_values = linspace_from_scan(mean_scan_cfg)
        std_values = linspace_from_scan(std_scan_cfg)
        autocorr_matrix = np.zeros((std_values.size, mean_values.size), dtype=float)
        for i, std_value in enumerate(std_values):
            for j, mean_value in enumerate(mean_values):
                spec_ij = overwrite_spec_value(spec, "e_to_e.w.std", float(std_value))
                spec_ij = overwrite_spec_value(spec_ij, "e_to_e.w.mean", float(mean_value))
                runtime = compile_graph_to_runtime(spec_ij, backend="pytorch")
                result = simulate_network(runtime)
                autocorr_value, *_ = autocorr_peak(
                    result.spikes,
                    T_ms=float(runtime["config"].T),
                    pop="E",
                    N_E=int(runtime["config"].N_E),
                    N_I=int(runtime["config"].N_I),
                )
                autocorr_matrix[i, j] = float(autocorr_value)
            print(
                f"[heatmap {i + 1}/{len(std_values)}] "
                f"std={std_value:.6f} complete"
            )

        save_ee_autocorr_heatmap(
            mean_values=mean_values,
            std_values=std_values,
            autocorr_matrix=autocorr_matrix,
            out_path=data_path / "heatmap_ee_mean_std_autocorr_peak.png",
        )
        heatmap_summary = {
            "x_parameter": "e_to_e.w.mean",
            "x_values": [float(v) for v in mean_values],
            "y_parameter": "e_to_e.w.std",
            "y_values": [float(v) for v in std_values],
            "metric": "autocorr_peak",
            "matrix": autocorr_matrix.tolist(),
        }
        (data_path / "heatmap_ee_mean_std_autocorr_peak.json").write_text(
            json.dumps(heatmap_summary, indent=2),
            encoding="utf-8",
        )
        rows = [",".join(["std\\mean"] + [f"{float(v):.9f}" for v in mean_values])]
        for i, std_value in enumerate(std_values):
            row = [f"{float(std_value):.9f}"] + [f"{float(v):.9f}" for v in autocorr_matrix[i]]
            rows.append(",".join(row))
        (data_path / "heatmap_ee_mean_std_autocorr_peak.csv").write_text(
            "\n".join(rows) + "\n",
            encoding="utf-8",
        )

    print(
        "saved scan rasters, E-rate plots, autocorr curves, scan metrics, "
        f"and EE autocorr heatmap to {data_path}"
    )


if __name__ == "__main__":
    main()
