from __future__ import annotations

import json
import time
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np

from pinglab.analysis import (
    decode_fit_metrics,
    envelope_rate_hz,
    lowpass_first_order,
    mean_firing_rates,
    population_rate,
    rate_psd,
)
from pinglab.analysis.autocorr_peak import autocorr_peak
from pinglab.analysis.coherence import coherence_peak
from pinglab.analysis.lagged_coherence import lagged_coherence
from pinglab.analysis.mean_pairwise_xcorr_peak import mean_pairwise_xcorr_peak
from pinglab.io.external_spike_train import external_spike_train
from pinglab.io.oscillating import oscillating
from pinglab.io.pulse import add_pulse_to_input, add_pulse_train_to_input
from pinglab.io.ramp import ramp
from pinglab.backends.pytorch import simulate_network
from pinglab.backends.types import InstrumentsConfig, NetworkConfig, Spikes
from pinglab.io.weights import ResolvedWeightMatrices, build_adjacency_matrices, to_torch_weights
from pinglab.io.slice_spikes import slice_spikes

DEFAULT_CONFIG = NetworkConfig(
    dt=0.1,
    T=1000.0,
    N_E=800,
    N_I=200,
    seed=0,
    neuron_model="lif",
    delay_ei=0.5,
    delay_ie=1.2,
    delay_ee=0.5,
    delay_ii=0.5,
    V_init=-65.0,
    E_L=-65.0,
    E_e=0.0,
    E_i=-80.0,
    C_m_E=1.0,
    g_L_E=0.05,
    C_m_I=1.0,
    g_L_I=0.1,
    V_th=-50.0,
    V_reset=-65.0,
    t_ref_E=3.0,
    t_ref_I=1.5,
    tau_ampa=2.0,
    tau_gaba=6.5,
    mqif_a=[0.02],
    mqif_Vr=[-55.0],
    mqif_w_a=[0.02],
    mqif_w_Vr=[-55.0],
    mqif_w_tau=[100.0],
    g_L_heterogeneity_sd=0.15,
    C_m_heterogeneity_sd=0.10,
    V_th_heterogeneity_sd=1.2,
    t_ref_heterogeneity_sd=0.3,
    instruments=InstrumentsConfig(
        variables=[],
        all_neurons=False,
        neuron_ids=[],
    ),
)

DEFAULT_INPUTS = {
    "input_type": "ramp",
    "input_population": "e",
    "I_E_start": 0.7,
    "I_E_end": 0.7,
    "I_I_start": 0.7,
    "I_I_end": 0.7,
    "I_E_base": 0.7,
    "I_I_base": 0.7,
    "noise_std": 0.5,
    "noise_std_E": None,
    "noise_std_I": None,
    "seed": 0,
    "sine_freq_hz": 5.0,
    "sine_amp": 2.0,
    "sine_y_offset": 0.0,
    "sine_phase": 0.0,
    "sine_phase_offset_i": 0.0,
    "lambda0_hz": 30.0,
    "mod_depth": 0.5,
    "envelope_freq_hz": 5.0,
    "phase_rad": 0.0,
    "w_in": 0.25,
    "tau_in_ms": 3.0,
    "pulse_t_ms": 200.0,
    "pulse_width_ms": 20.0,
    "pulse_interval_ms": 100.0,
    "pulse_amp_E": 1.0,
    "pulse_amp_I": 1.0,
    "targeted_subset_enabled": False,
    "target_population": "all",
    "target_strategy": "random",
    "target_fraction": 1.0,
    "target_seed": 0,
}

DEFAULT_WEIGHTS = {
    "ee": {"mean": 0.02, "std": 0.0},
    "ei": {"mean": 0.015, "std": 0.0},
    "ie": {"mean": 0.015, "std": 0.0},
    "ii": {"mean": 0.02, "std": 0.0},
    "clamp_min": 0.0,
    "seed": 0,
}

RHYTHM_BIN_MS = 5.0
RHYTHM_BURN_IN_MS = 200.0


def _weights_cache_key(n_e: int, n_i: int, weights: dict[str, Any]) -> str:
    payload = {
        "N_E": int(n_e),
        "N_I": int(n_i),
        "weights": weights,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _freeze_weight_matrices(weight_mats: ResolvedWeightMatrices) -> ResolvedWeightMatrices:
    weight_mats.W.setflags(write=False)
    weight_mats.W_ee.setflags(write=False)
    weight_mats.W_ei.setflags(write=False)
    weight_mats.W_ie.setflags(write=False)
    weight_mats.W_ii.setflags(write=False)
    if weight_mats.D_ee is not None:
        weight_mats.D_ee.setflags(write=False)
    if weight_mats.D_ei is not None:
        weight_mats.D_ei.setflags(write=False)
    if weight_mats.D_ie is not None:
        weight_mats.D_ie.setflags(write=False)
    if weight_mats.D_ii is not None:
        weight_mats.D_ii.setflags(write=False)
    return weight_mats


@lru_cache(maxsize=128)
def _cached_weight_matrices(cache_key: str) -> ResolvedWeightMatrices:
    payload = json.loads(cache_key)
    n_e = int(payload["N_E"])
    n_i = int(payload["N_I"])
    weights = payload["weights"]
    mats = build_adjacency_matrices(
        N_E=n_e,
        N_I=n_i,
        ee=weights["ee"],
        ei=weights["ei"],
        ie=weights["ie"],
        ii=weights["ii"],
        clamp_min=weights["clamp_min"],
        seed=weights.get("seed"),
    )
    return _freeze_weight_matrices(mats)


def get_weight_matrices(n_e: int, n_i: int, weights: dict[str, Any]) -> ResolvedWeightMatrices:
    cache_key = _weights_cache_key(n_e, n_i, weights)
    return _cached_weight_matrices(cache_key)


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _weights_histograms(
    weights: Any, bins: int = 40
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    ee = _as_numpy(weights.W_ee).ravel()
    ei = _as_numpy(weights.W_ei).ravel()
    ie = _as_numpy(weights.W_ie).ravel()
    ii = _as_numpy(weights.W_ii).ravel()
    max_val = 0.0
    for arr in (ee, ei, ie, ii):
        if arr.size:
            max_val = max(max_val, float(np.max(arr)))
    if max_val <= 0:
        return [0.0, 1.0], [0.0], [0.0], [0.0], [0.0]

    edges = np.linspace(0.0, max_val, bins + 1)
    centers = ((edges[:-1] + edges[1:]) / 2.0).tolist()

    def _hist(arr: np.ndarray) -> list[float]:
        if arr.size == 0:
            return [0.0] * len(centers)
        counts, _ = np.histogram(arr, bins=edges)
        return counts.astype(float).tolist()

    return centers, _hist(ee), _hist(ei), _hist(ie), _hist(ii)


def _block_slices(total: int, blocks: int) -> list[slice]:
    safe_total = max(0, int(total))
    safe_blocks = max(1, int(blocks))
    out: list[slice] = []
    for block_idx in range(safe_blocks):
        start = int(np.floor((block_idx * safe_total) / safe_blocks))
        end = int(np.floor(((block_idx + 1) * safe_total) / safe_blocks))
        out.append(slice(start, end))
    return out


def apply_input_population_mask(
    external_input: np.ndarray,
    n_e: int,
    n_i: int,
    input_population: str,
) -> np.ndarray:
    if input_population == "all":
        return external_input
    if input_population == "e":
        if n_i > 0:
            external_input[:, n_e : n_e + n_i] = 0.0
        return external_input
    if input_population == "i":
        if n_e > 0:
            external_input[:, :n_e] = 0.0
        return external_input
    return external_input


def clip_input_nonnegative(external_input: Any) -> Any:
    if isinstance(external_input, np.ndarray):
        np.maximum(external_input, 0.0, out=external_input)
        return external_input
    if hasattr(external_input, "clamp_"):
        external_input.clamp_(min=0.0)
        return external_input
    arr = np.asarray(external_input)
    np.maximum(arr, 0.0, out=arr)
    return arr


def _select_subset_neurons(
    neuron_ids: np.ndarray,
    *,
    fraction: float,
    strategy: str,
    seed: int,
) -> np.ndarray:
    if neuron_ids.size == 0:
        return neuron_ids

    safe_fraction = max(0.0, min(1.0, float(fraction)))
    if safe_fraction >= 1.0:
        return neuron_ids
    if safe_fraction <= 0.0:
        return np.array([], dtype=int)

    target_count = int(np.floor(neuron_ids.size * safe_fraction))
    if target_count <= 0:
        return np.array([], dtype=int)
    if target_count >= neuron_ids.size:
        return neuron_ids

    if strategy == "first":
        return neuron_ids[:target_count]

    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(neuron_ids, size=target_count, replace=False)
    return np.sort(chosen.astype(int, copy=False))


def _resolve_targeted_pulse_ids(
    *,
    e_ids: np.ndarray,
    i_ids: np.ndarray,
    targeted_subset_enabled: bool,
    target_population: str,
    target_strategy: str,
    target_fraction: float,
    target_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not targeted_subset_enabled:
        return e_ids, i_ids

    include_e = target_population in {"all", "e"}
    include_i = target_population in {"all", "i"}
    selected_e = (
        _select_subset_neurons(
            e_ids,
            fraction=target_fraction,
            strategy=target_strategy,
            seed=target_seed,
        )
        if include_e
        else np.array([], dtype=int)
    )
    selected_i = (
        _select_subset_neurons(
            i_ids,
            fraction=target_fraction,
            strategy=target_strategy,
            seed=target_seed + 1,
        )
        if include_i
        else np.array([], dtype=int)
    )
    return selected_e, selected_i


def _histogram_from_values(values: np.ndarray, bins: list[float]) -> list[float]:
    if not bins:
        return []
    counts = [0.0] * len(bins)
    if values.size == 0:
        return counts
    min_bin = bins[0]
    max_bin = bins[-1]
    span = max(1e-9, max_bin - min_bin)
    flat = values.ravel()
    for value in flat:
        if not np.isfinite(value):
            continue
        normalized = (float(value) - min_bin) / span
        idx = int(np.floor(normalized * (len(bins) - 1)))
        idx = min(max(idx, 0), len(bins) - 1)
        counts[idx] += 1.0
    return counts


def _weights_histograms_by_e_block(
    weights: Any,
    bins: list[float],
    n_e: int,
    blocks: int,
) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[list[float]]]:
    if not bins or n_e <= 0 or blocks <= 0:
        return [], [], [], []
    ranges = _block_slices(n_e, blocks)

    ee_counts: list[list[float]] = []
    ei_counts: list[list[float]] = []
    ie_counts: list[list[float]] = []
    ii_counts: list[list[float]] = []
    w_ee = _as_numpy(weights.W_ee)
    w_ei = _as_numpy(weights.W_ei)
    w_ie = _as_numpy(weights.W_ie)
    w_ii = _as_numpy(weights.W_ii)

    for block_slice in ranges:
        ee_counts.append(_histogram_from_values(w_ee[block_slice, block_slice], bins))
        ei_counts.append(_histogram_from_values(w_ei[:, block_slice], bins))
        ie_counts.append(_histogram_from_values(w_ie[block_slice, :], bins))
        ii_counts.append(_histogram_from_values(w_ii, bins))

    return ee_counts, ei_counts, ie_counts, ii_counts


def _downsample_matrix(matrix: Any, max_size: int = 200) -> list[list[float]]:
    matrix_np = _as_numpy(matrix)
    rows, cols = matrix_np.shape
    if rows == 0 or cols == 0:
        return []
    row_step = max(1, int(np.ceil(rows / max_size)))
    col_step = max(1, int(np.ceil(cols / max_size)))
    out_rows = int(np.ceil(rows / row_step))
    out_cols = int(np.ceil(cols / col_step))
    out = np.zeros((out_rows, out_cols), dtype=float)
    for i in range(out_rows):
        r0 = i * row_step
        r1 = min(rows, r0 + row_step)
        for j in range(out_cols):
            c0 = j * col_step
            c1 = min(cols, c0 + col_step)
            block = matrix_np[r0:r1, c0:c1]
            if block.size == 0:
                out[i, j] = 0.0
            else:
                out[i, j] = float(np.mean(block))
    return out.tolist()


def _isi_cv(spikes: Spikes, n_e: int, n_i: int, pop: Literal["E", "I"] = "E") -> float:
    if spikes.times.size == 0:
        return 0.0
    if pop == "E":
        neuron_ids = np.arange(n_e)
    else:
        neuron_ids = np.arange(n_e, n_e + n_i)
    cvs: list[float] = []
    for nid in neuron_ids:
        t = spikes.times[spikes.ids == nid]
        if t.size < 2:
            continue
        t = np.sort(t)
        isis = np.diff(t)
        if isis.size == 0:
            continue
        mean_isi = float(np.mean(isis))
        if mean_isi <= 0.0:
            continue
        cv = float(np.std(isis) / mean_isi)
        cvs.append(cv)
    if not cvs:
        return 0.0
    return float(np.mean(cvs))


def _isi_mean_and_inverse_hz(
    spikes: Spikes, n_e: int, n_i: int, pop: Literal["E", "I"] = "E"
) -> tuple[float, float]:
    if spikes.times.size == 0:
        return 0.0, 0.0
    if pop == "E":
        neuron_ids = np.arange(n_e)
    else:
        neuron_ids = np.arange(n_e, n_e + n_i)
    mean_isis_ms: list[float] = []
    for nid in neuron_ids:
        t = spikes.times[spikes.ids == nid]
        if t.size < 2:
            continue
        t = np.sort(t)
        isis = np.diff(t)
        if isis.size == 0:
            continue
        mean_isi_ms = float(np.mean(isis))
        if mean_isi_ms <= 0.0:
            continue
        mean_isis_ms.append(mean_isi_ms)
    if not mean_isis_ms:
        return 0.0, 0.0
    mean_isi_ms = float(np.mean(mean_isis_ms))
    inv_hz = float(1000.0 / mean_isi_ms) if mean_isi_ms > 0.0 else 0.0
    return mean_isi_ms, inv_hz


def _layer_labels(n_e: int, n_i: int, blocks: int) -> list[str]:
    labels = [f"L{i + 1}" for i in range(len(_block_slices(n_e, blocks)))]
    if n_i > 0:
        labels.append("I")
    return labels


def _population_rate_for_neuron_ids(
    spikes: Spikes,
    *,
    neuron_ids: np.ndarray,
    t_ms: np.ndarray,
    dt_ms: float,
) -> np.ndarray:
    if t_ms.size == 0:
        return np.array([], dtype=float)
    if neuron_ids.size == 0:
        return np.zeros_like(t_ms, dtype=float)
    edges = np.concatenate([t_ms - 0.5 * dt_ms, [t_ms[-1] + 0.5 * dt_ms]])
    mask = np.isin(spikes.ids, neuron_ids)
    counts, _ = np.histogram(spikes.times[mask], bins=edges)
    dt_s = dt_ms / 1000.0
    return counts.astype(float) / (max(1, neuron_ids.size) * dt_s)


def _subset_spikes_reindexed(spikes: Spikes, neuron_ids: np.ndarray) -> Spikes:
    if neuron_ids.size == 0 or spikes.times.size == 0:
        return Spikes(times=np.array([], dtype=float), ids=np.array([], dtype=int), types=np.array([], dtype=int))
    sorted_ids = np.sort(neuron_ids.astype(int, copy=False))
    mask = np.isin(spikes.ids, sorted_ids)
    if not np.any(mask):
        return Spikes(times=np.array([], dtype=float), ids=np.array([], dtype=int), types=np.array([], dtype=int))
    times = spikes.times[mask]
    original_ids = spikes.ids[mask]
    mapped_ids = np.searchsorted(sorted_ids, original_ids).astype(int, copy=False)
    mapped_types = None
    if spikes.types is not None:
        mapped_types = spikes.types[mask]
    return Spikes(times=times, ids=mapped_ids, types=mapped_types)


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _merge_defaults(defaults: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    if overrides is None:
        return defaults.copy()
    return _deep_merge(defaults, overrides)


def build_weights_preview(
    *,
    config_overrides: dict[str, Any] | None = None,
    weights_overrides: dict[str, Any] | None = None,
    runtime_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if runtime_overrides is not None and runtime_overrides.get("weights") is not None:
        config_raw = runtime_overrides.get("config")
        if isinstance(config_raw, NetworkConfig):
            config = config_raw
        else:
            config_dict = _merge_defaults(DEFAULT_CONFIG.model_dump(), config_overrides)
            config = NetworkConfig.model_validate(config_dict)
        weight_mats = runtime_overrides["weights"]
    else:
        config_dict = _merge_defaults(DEFAULT_CONFIG.model_dump(), config_overrides)
        config = NetworkConfig.model_validate(config_dict)
        weights = _merge_defaults(DEFAULT_WEIGHTS, weights_overrides)
        weights_seed = weights["seed"] if weights["seed"] is not None else config.seed
        resolved_weights = dict(weights)
        resolved_weights["seed"] = int(weights_seed) if weights_seed is not None else None
        weight_mats = get_weight_matrices(config.N_E, config.N_I, resolved_weights)
    (
        weights_hist_bins,
        weights_hist_counts_ee,
        weights_hist_counts_ei,
        weights_hist_counts_ie,
        weights_hist_counts_ii,
    ) = _weights_histograms(weight_mats)
    e_blocks = 1
    (
        weights_hist_blocks_ee,
        weights_hist_blocks_ei,
        weights_hist_blocks_ie,
        weights_hist_blocks_ii,
    ) = _weights_histograms_by_e_block(
        weight_mats,
        weights_hist_bins,
        config.N_E,
        e_blocks,
    )
    return {
        "weights_hist_bins": weights_hist_bins,
        "weights_hist_counts_ee": weights_hist_counts_ee,
        "weights_hist_counts_ei": weights_hist_counts_ei,
        "weights_hist_counts_ie": weights_hist_counts_ie,
        "weights_hist_counts_ii": weights_hist_counts_ii,
        "weights_hist_blocks_ee": weights_hist_blocks_ee,
        "weights_hist_blocks_ei": weights_hist_blocks_ei,
        "weights_hist_blocks_ie": weights_hist_blocks_ie,
        "weights_hist_blocks_ii": weights_hist_blocks_ii,
        "weights_heatmap": _downsample_matrix(weight_mats.W, max_size=200),
    }


def run_simulation(
    *,
    config_overrides: dict[str, Any] | None = None,
    inputs_overrides: dict[str, Any] | None = None,
    weights_overrides: dict[str, Any] | None = None,
    runtime_overrides: dict[str, Any] | None = None,
    performance_mode: bool = True,
    max_spikes: int | None = 30000,
    burn_in_ms: float | None = None,
) -> dict[str, Any]:
    total_start = time.perf_counter()
    using_compiled_runtime = runtime_overrides is not None
    if using_compiled_runtime:
        config_raw = runtime_overrides.get("config")
        if not isinstance(config_raw, NetworkConfig):
            raise ValueError("runtime_overrides.config must be a NetworkConfig")
        config = config_raw
        runtime_backend = str(runtime_overrides.get("backend", "pytorch"))
        inputs = DEFAULT_INPUTS.copy()
        weights = DEFAULT_WEIGHTS.copy()
    else:
        runtime_backend = "pytorch"
        config = DEFAULT_CONFIG.model_copy(update=(config_overrides or {}))
        inputs = _merge_defaults(DEFAULT_INPUTS, inputs_overrides)
        weights = _merge_defaults(DEFAULT_WEIGHTS, weights_overrides)
    e_blocks = 1
    e_layer_slices = _block_slices(config.N_E, e_blocks)
    layer_labels = _layer_labels(config.N_E, config.N_I, e_blocks)

    representative_neuron_ids: list[int] = []
    for layer_slice in e_layer_slices:
        if layer_slice.stop > layer_slice.start:
            representative_neuron_ids.append(layer_slice.start)
    if config.N_I > 0:
        representative_neuron_ids.append(config.N_E)
    if performance_mode:
        config = config.model_copy(
            update={
                "instruments": InstrumentsConfig(
                    variables=[],
                    neuron_ids=[],
                    all_neurons=False,
                )
            }
        )
    else:
        config = config.model_copy(
            update={
                "instruments": InstrumentsConfig(
                    variables=["V", "g_e", "g_i"],
                    neuron_ids=representative_neuron_ids,
                    all_neurons=False,
                )
            }
        )

    num_steps = int(np.ceil(config.T / config.dt))

    input_seed = inputs["seed"] if inputs["seed"] is not None else config.seed
    weights_seed = weights["seed"] if weights["seed"] is not None else config.seed

    input_type = str(inputs.get("input_type", "ramp"))
    input_population = str(inputs.get("input_population", "e"))
    noise_std_E = inputs.get("noise_std_E")
    noise_std_I = inputs.get("noise_std_I")
    if noise_std_E is None:
        noise_std_E = inputs.get("noise_std", 0.0)
    if noise_std_I is None:
        noise_std_I = inputs.get("noise_std", 0.0)
    targeted_subset_enabled = bool(inputs.get("targeted_subset_enabled", False))
    target_population = str(inputs.get("target_population", "all"))
    target_strategy = str(inputs.get("target_strategy", "random"))
    target_fraction = float(inputs.get("target_fraction", 1.0))
    target_seed = int(inputs.get("target_seed", 0))
    source_spikes_e: np.ndarray | None = None
    source_spikes_i: np.ndarray | None = None
    input_phase_start = time.perf_counter()
    if using_compiled_runtime:
        external_input = runtime_overrides.get("external_input")
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise ImportError("PyTorch backend requested but torch is not installed") from exc
        if not isinstance(external_input, torch.Tensor):
            raise ValueError("runtime_overrides.external_input must be a torch tensor")
    elif input_type == "ramp":
        external_input = ramp(
            config.N_E,
            config.N_I,
            inputs["I_E_start"],
            inputs["I_E_end"],
            inputs["I_I_start"],
            inputs["I_I_end"],
            noise_std_E,
            noise_std_I,
            num_steps,
            config.dt,
            int(input_seed) if input_seed is not None else 0,
        )
    elif input_type == "sine":
        sine_y_offset = float(inputs.get("sine_y_offset", 0.0))
        external_input = oscillating(
            config.N_E,
            config.N_I,
            float(inputs["I_E_base"]) + sine_y_offset,
            float(inputs["I_I_base"]) + sine_y_offset,
            float(max(noise_std_E, noise_std_I)),
            num_steps,
            config.dt,
            int(input_seed) if input_seed is not None else 0,
            oscillation_freq=float(inputs["sine_freq_hz"]),
            oscillation_amplitude=float(inputs["sine_amp"]),
            oscillation_phase=float(inputs["sine_phase"]),
            phase_offset_I=float(inputs["sine_phase_offset_i"]),
        )
    elif input_type == "external_spike_train":
        spike_result = external_spike_train(
            config.N_E,
            config.N_I,
            0.0,
            0.0,
            0.0,
            0.0,
            num_steps,
            config.dt,
            int(input_seed) if input_seed is not None else 0,
            lambda0_hz=float(inputs.get("lambda0_hz", 30.0)),
            mod_depth=float(inputs.get("mod_depth", 0.5)),
            envelope_freq_hz=float(inputs.get("envelope_freq_hz", 5.0)),
            phase_rad=float(inputs.get("phase_rad", 0.0)),
            w_in=float(inputs.get("w_in", 0.25)),
            tau_in_ms=float(inputs.get("tau_in_ms", 3.0)),
            return_spikes=True,
        )
        external_input = spike_result[0]
        source_spikes_e = spike_result[1]
        source_spikes_i = spike_result[2]
        # V1 spike-train input targets E only by design.
        input_population = "e"
    else:
        baseline = ramp(
            config.N_E,
            config.N_I,
            inputs["I_E_base"],
            inputs["I_E_base"],
            inputs["I_I_base"],
            inputs["I_I_base"],
            noise_std_E,
            noise_std_I,
            num_steps,
            config.dt,
            int(input_seed) if input_seed is not None else 0,
        )
        e_ids = np.arange(config.N_E)
        i_ids = np.arange(config.N_E, config.N_E + config.N_I)
        targeted_e_ids, targeted_i_ids = _resolve_targeted_pulse_ids(
            e_ids=e_ids,
            i_ids=i_ids,
            targeted_subset_enabled=targeted_subset_enabled,
            target_population=target_population,
            target_strategy=target_strategy,
            target_fraction=target_fraction,
            target_seed=target_seed,
        )
        pulse_t_ms = inputs["pulse_t_ms"]
        pulse_width_ms = inputs["pulse_width_ms"]
        pulse_amp_E = inputs["pulse_amp_E"]
        pulse_amp_I = inputs["pulse_amp_I"]
        if input_type == "pulse":
            if targeted_e_ids.size > 0:
                baseline = add_pulse_to_input(
                    baseline,
                    target_neurons=targeted_e_ids,
                    pulse_t=pulse_t_ms,
                    pulse_width_ms=pulse_width_ms,
                    pulse_amp=pulse_amp_E,
                    dt=config.dt,
                    num_steps=num_steps,
                )
            if targeted_i_ids.size > 0:
                baseline = add_pulse_to_input(
                    baseline,
                    target_neurons=targeted_i_ids,
                    pulse_t=pulse_t_ms,
                    pulse_width_ms=pulse_width_ms,
                    pulse_amp=pulse_amp_I,
                    dt=config.dt,
                    num_steps=num_steps,
                )
            external_input = baseline
        elif input_type == "pulses":
            if targeted_e_ids.size > 0:
                baseline = add_pulse_train_to_input(
                    baseline,
                    target_neurons=targeted_e_ids,
                    pulse_t=pulse_t_ms,
                    pulse_width_ms=pulse_width_ms,
                    pulse_amp=pulse_amp_E,
                    pulse_interval_ms=inputs["pulse_interval_ms"],
                    dt=config.dt,
                    num_steps=num_steps,
                )
            if targeted_i_ids.size > 0:
                baseline = add_pulse_train_to_input(
                    baseline,
                    target_neurons=targeted_i_ids,
                    pulse_t=pulse_t_ms,
                    pulse_width_ms=pulse_width_ms,
                    pulse_amp=pulse_amp_I,
                    pulse_interval_ms=inputs["pulse_interval_ms"],
                    dt=config.dt,
                    num_steps=num_steps,
                )
            external_input = baseline
        else:
            raise ValueError(f"Unsupported input_type: {input_type}")
    if not using_compiled_runtime:
        external_input = apply_input_population_mask(
            external_input=external_input,
            n_e=config.N_E,
            n_i=config.N_I,
            input_population=input_population,
        )
        external_input = clip_input_nonnegative(external_input)
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise ImportError("PyTorch backend requested but torch is not installed") from exc
        external_input = torch.as_tensor(external_input, dtype=torch.float32)
    input_prep_ms = (time.perf_counter() - input_phase_start) * 1000.0

    weights_phase_start = time.perf_counter()
    if using_compiled_runtime:
        weights_runtime = runtime_overrides.get("weights")
        if weights_runtime is None:
            raise ValueError("runtime_overrides.weights must be provided")
        weight_mats = weights_runtime
    else:
        resolved_weights = dict(weights)
        resolved_weights["seed"] = int(weights_seed) if weights_seed is not None else None
        weight_mats = get_weight_matrices(config.N_E, config.N_I, resolved_weights)
    weights_hist_bins: list[float] = []
    weights_hist_counts_ee: list[float] = []
    weights_hist_counts_ei: list[float] = []
    weights_hist_counts_ie: list[float] = []
    weights_hist_counts_ii: list[float] = []
    weights_hist_blocks_ee: list[list[float]] = []
    weights_hist_blocks_ei: list[list[float]] = []
    weights_hist_blocks_ie: list[list[float]] = []
    weights_hist_blocks_ii: list[list[float]] = []
    if not performance_mode:
        (
            weights_hist_bins,
            weights_hist_counts_ee,
            weights_hist_counts_ei,
            weights_hist_counts_ie,
            weights_hist_counts_ii,
        ) = _weights_histograms(weight_mats)
        (
            weights_hist_blocks_ee,
            weights_hist_blocks_ei,
            weights_hist_blocks_ie,
            weights_hist_blocks_ii,
        ) = _weights_histograms_by_e_block(
            weight_mats,
            weights_hist_bins,
            config.N_E,
            e_blocks,
        )
    weights_build_ms = (time.perf_counter() - weights_phase_start) * 1000.0

    t0 = time.perf_counter()
    if using_compiled_runtime:
        runtime_obj = runtime_overrides
    else:
        runtime_obj = SimpleNamespace(
            config=config,
            external_input=external_input,
            weights=to_torch_weights(weight_mats, device=external_input.device, dtype=external_input.dtype),
            model=runtime_overrides.get("model") if runtime_overrides is not None else None,
            backend="pytorch",
            device=str(external_input.device),
        )
        from pinglab.backends.pytorch import lif_step

        runtime_obj.model = lif_step
    max_spikes_pt = int(max_spikes) if max_spikes is not None else None
    result = simulate_network(runtime_obj, max_spikes=max_spikes_pt)
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    spikes = result.spikes
    total_e_spikes = int(np.sum(spikes.ids < config.N_E))
    total_i_spikes = int(spikes.ids.size - total_e_spikes)
    spikes_truncated = False
    spikes_for_response = spikes
    if max_spikes is not None and spikes.times.size > max_spikes:
        spikes_truncated = True
        idx = np.linspace(0, spikes.times.size - 1, num=max_spikes, dtype=int)
        spikes_for_response = Spikes(
            times=spikes.times[idx],
            ids=spikes.ids[idx],
            types=spikes.types[idx] if spikes.types is not None else None,
        )
    spikes_response = {
        "times": spikes_for_response.times.tolist(),
        "ids": spikes_for_response.ids.tolist(),
        "types": (spikes_for_response.types.tolist() if spikes_for_response.types is not None else []),
    }

    analysis_phase_start = time.perf_counter()

    if burn_in_ms is None:
        burn_in_ms = RHYTHM_BURN_IN_MS
    analysis_window_start = min(burn_in_ms, max(0.0, config.T - config.dt))
    analysis_stop = config.T
    analysis_T = analysis_stop - analysis_window_start
    mean_rate_E = 0.0
    mean_rate_I = 0.0
    isi_cv_E = 0.0
    isi_mean_E_ms = 0.0
    isi_inverse_E_hz = 0.0
    autocorr_peak_val = 0.0
    xcorr_peak_val = 0.0
    coherence_peak_val = 0.0
    lagged_coherence_val = 0.0
    population_rate_t_ms: list[float] = []
    population_rate_hz_E: list[float] = []
    population_rate_hz_I: list[float] = []
    population_rate_hz_layers: list[list[float]] = []
    decode_lowpass_hz_E: list[float] = []
    decode_lowpass_hz_I: list[float] = []
    decode_lowpass_hz_layers: list[list[float]] = []
    membrane_t_ms: list[float] = []
    membrane_V_E: list[float] = []
    membrane_V_I: list[float] = []
    membrane_V_layers: list[list[float]] = []
    membrane_g_e_E: list[float] = []
    membrane_g_i_E: list[float] = []
    membrane_g_e_I: list[float] = []
    membrane_g_i_I: list[float] = []
    autocorr_lags_ms: list[float] = []
    autocorr_corr: list[float] = []
    autocorr_lags_layers_ms: list[list[float]] = []
    autocorr_corr_layers: list[list[float]] = []
    xcorr_lags_ms: list[float] = []
    xcorr_corr: list[float] = []
    xcorr_lags_layers_ms: list[list[float]] = []
    xcorr_corr_layers: list[list[float]] = []
    coherence_lags_ms: list[float] = []
    coherence_corr: list[float] = []
    psd_freqs_hz: list[float] = []
    psd_power: list[float] = []
    psd_power_layers: list[list[float]] = []
    input_t_ms: list[float] = []
    input_mean_E: list[float] = []
    input_mean_I: list[float] = []
    input_mean_layers: list[list[float]] = []
    input_raw_spike_fraction_E: list[float] = []
    input_raw_spike_fraction_I: list[float] = []
    input_raw_spike_fraction_layers: list[list[float]] = []
    input_raw_raster_times_ms_layers: list[list[float]] = []
    input_raw_raster_ids_layers: list[list[int]] = []
    input_envelope_hz: list[float] = []
    input_envelope_hz_layers: list[list[float]] = []
    input_spike_fraction_E: list[float] = []
    input_spike_fraction_I: list[float] = []
    input_spike_fraction_layers: list[list[float]] = []
    decode_envelope_hz: list[float] = []
    decode_envelope_hz_layers: list[list[float]] = []
    decode_corr: float = 0.0
    decode_rmse: float = 0.0
    decode_corr_layers: list[float] = []
    decode_rmse_layers: list[float] = []
    psd_peak_freq_hz = 0.0
    psd_peak_bandwidth_hz = 0.0
    psd_peak_q_factor = 0.0
    def _finite_list(values: list[float]) -> list[float]:
        arr = np.array(values, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr.tolist()

    def _finite_scalar(value: float) -> float:
        return float(value) if np.isfinite(value) else 0.0

    def _psd_peak_bandwidth_q(
        freqs_hz: np.ndarray,
        power: np.ndarray,
        *,
        min_freq_hz: float = 20.0,
        max_freq_hz: float = 120.0,
    ) -> tuple[float, float, float]:
        valid = (
            np.isfinite(freqs_hz)
            & np.isfinite(power)
            & (freqs_hz >= float(min_freq_hz))
            & (freqs_hz <= float(max_freq_hz))
        )
        if not np.any(valid):
            return 0.0, 0.0, 0.0
        f = freqs_hz[valid]
        p = power[valid]
        if f.size == 0 or p.size == 0:
            return 0.0, 0.0, 0.0
        peak_idx = int(np.argmax(p))
        peak_freq = float(f[peak_idx])
        peak_power = float(p[peak_idx])
        if peak_power <= 0.0:
            return peak_freq, 0.0, 0.0
        half_power = 0.5 * peak_power

        left_idx = peak_idx
        while left_idx > 0 and float(p[left_idx]) >= half_power:
            left_idx -= 1
        right_idx = peak_idx
        while right_idx < (p.size - 1) and float(p[right_idx]) >= half_power:
            right_idx += 1

        bandwidth_hz = float(max(0.0, f[right_idx] - f[left_idx]))
        q_factor = float(peak_freq / bandwidth_hz) if bandwidth_hz > 0.0 else 0.0
        return peak_freq, bandwidth_hz, q_factor

    if (not performance_mode) and analysis_T > 0:
        sliced = slice_spikes(spikes, analysis_window_start, analysis_stop)
        shifted = Spikes(
            times=sliced.times - analysis_window_start,
            ids=sliced.ids,
            types=sliced.types,
        )
        mean_rate_E, mean_rate_I = mean_firing_rates(
            shifted,
            config.N_E,
            config.N_I,
        )
        isi_cv_E = _isi_cv(shifted, config.N_E, config.N_I, pop="E")
        isi_mean_E_ms, isi_inverse_E_hz = _isi_mean_and_inverse_hz(
            shifted, config.N_E, config.N_I, pop="E"
        )
        lagged_coherence_val, _, _, _, _, _ = lagged_coherence(
            shifted,
            T_ms=analysis_T,
            dt_ms=RHYTHM_BIN_MS,
            pop="E",
            N_E=config.N_E,
            N_I=config.N_I,
        )
        autocorr_peak_val, _, _, auto_lags, auto_corr = autocorr_peak(
            shifted,
            T_ms=analysis_T,
            dt_ms=RHYTHM_BIN_MS,
            pop="E",
            N_E=config.N_E,
            N_I=config.N_I,
        )
        xcorr_peak_val, xcorr_lags, xcorr_corr_vals = mean_pairwise_xcorr_peak(
            shifted,
            T_ms=analysis_T,
            N_E=config.N_E,
            dt_ms=RHYTHM_BIN_MS,
        )
        coherence_peak_val, coh_lags, coh_vals = coherence_peak(
            shifted,
            T_ms=analysis_T,
            N_E=config.N_E,
            dt_ms=RHYTHM_BIN_MS,
        )
        autocorr_lags_ms = auto_lags.tolist()
        autocorr_corr = auto_corr.tolist()
        xcorr_lags_ms = xcorr_lags.tolist()
        xcorr_corr = xcorr_corr_vals.tolist()
        coherence_lags_ms = coh_lags.tolist()
        coherence_corr = coh_vals.tolist()

        correlation_groups = [np.arange(s.start, s.stop, dtype=int) for s in e_layer_slices]
        if config.N_I > 0:
            correlation_groups.append(np.arange(config.N_E, config.N_E + config.N_I, dtype=int))
        for neuron_group in correlation_groups:
            if neuron_group.size == 0:
                autocorr_lags_layers_ms.append([])
                autocorr_corr_layers.append([])
                xcorr_lags_layers_ms.append([])
                xcorr_corr_layers.append([])
                continue
            layer_spikes = _subset_spikes_reindexed(shifted, neuron_group)
            _, _, _, layer_auto_lags, layer_auto_corr = autocorr_peak(
                layer_spikes,
                T_ms=analysis_T,
                dt_ms=RHYTHM_BIN_MS,
                pop="E",
                N_E=int(neuron_group.size),
                N_I=0,
            )
            _, layer_x_lags, layer_x_corr = mean_pairwise_xcorr_peak(
                layer_spikes,
                T_ms=analysis_T,
                N_E=int(neuron_group.size),
                dt_ms=RHYTHM_BIN_MS,
            )
            autocorr_lags_layers_ms.append(layer_auto_lags.tolist())
            autocorr_corr_layers.append(_finite_list(layer_auto_corr.tolist()))
            xcorr_lags_layers_ms.append(layer_x_lags.tolist())
            xcorr_corr_layers.append(_finite_list(layer_x_corr.tolist()))

        t_psd_ms, _ = population_rate(
            shifted,
            T_ms=analysis_T,
            dt_ms=RHYTHM_BIN_MS,
            pop="E",
            N_E=config.N_E,
            N_I=config.N_I,
        )
        layer_psd_series: list[np.ndarray] = []
        for layer_slice in e_layer_slices:
            layer_ids = np.arange(layer_slice.start, layer_slice.stop)
            rate_layer = _population_rate_for_neuron_ids(
                shifted,
                neuron_ids=layer_ids,
                t_ms=t_psd_ms,
                dt_ms=RHYTHM_BIN_MS,
            )
            layer_psd_series.append(rate_layer)
        if config.N_I > 0:
            i_ids = np.arange(config.N_E, config.N_E + config.N_I)
            rate_i_psd = _population_rate_for_neuron_ids(
                shifted,
                neuron_ids=i_ids,
                t_ms=t_psd_ms,
                dt_ms=RHYTHM_BIN_MS,
            )
            layer_psd_series.append(rate_i_psd)
        for series in layer_psd_series:
            if series.size == 0:
                psd_power_layers.append([])
                continue
            psd_freqs, psd_vals = rate_psd(series, dt_ms=RHYTHM_BIN_MS)
            if not psd_freqs_hz:
                psd_freqs_hz = psd_freqs.tolist()
            psd_power_layers.append(_finite_list(psd_vals.tolist()))
        if psd_power_layers:
            psd_power = psd_power_layers[0]
            psd_peak_freq_hz, psd_peak_bandwidth_hz, psd_peak_q_factor = _psd_peak_bandwidth_q(
                np.array(psd_freqs_hz, dtype=float),
                np.array(psd_power, dtype=float),
                min_freq_hz=20.0,
                max_freq_hz=120.0,
            )

    if not performance_mode:
        t_ms_full, rate_hz_full_E = population_rate(
            spikes,
            T_ms=config.T,
            dt_ms=RHYTHM_BIN_MS,
            pop="E",
            N_E=config.N_E,
            N_I=config.N_I,
        )
        _, rate_hz_full_I = population_rate(
            spikes,
            T_ms=config.T,
            dt_ms=RHYTHM_BIN_MS,
            pop="I",
            N_E=config.N_E,
            N_I=config.N_I,
        )

    if not performance_mode:
        population_rate_t_ms = t_ms_full.tolist()
        population_rate_hz_E = _finite_list(rate_hz_full_E.tolist())
        population_rate_hz_I = _finite_list(rate_hz_full_I.tolist())
        decode_lowpass_hz_E = _finite_list(
            lowpass_first_order(rate_hz_full_E, dt_ms=RHYTHM_BIN_MS, cutoff_hz=10.0).tolist()
        )
        decode_lowpass_hz_I = _finite_list(
            lowpass_first_order(rate_hz_full_I, dt_ms=RHYTHM_BIN_MS, cutoff_hz=10.0).tolist()
        )
        for layer_slice in e_layer_slices:
            layer_ids = np.arange(layer_slice.start, layer_slice.stop)
            layer_rate = _population_rate_for_neuron_ids(
                spikes,
                neuron_ids=layer_ids,
                t_ms=t_ms_full,
                dt_ms=RHYTHM_BIN_MS,
            )
            population_rate_hz_layers.append(_finite_list(layer_rate.tolist()))
            decode_lowpass_hz_layers.append(
                _finite_list(
                    lowpass_first_order(layer_rate, dt_ms=RHYTHM_BIN_MS, cutoff_hz=10.0).tolist()
                )
            )
        if config.N_I > 0:
            i_ids = np.arange(config.N_E, config.N_E + config.N_I)
            i_rate = _population_rate_for_neuron_ids(
                spikes,
                neuron_ids=i_ids,
                t_ms=t_ms_full,
                dt_ms=RHYTHM_BIN_MS,
            )
            population_rate_hz_layers.append(_finite_list(i_rate.tolist()))
            decode_lowpass_hz_layers.append(
                _finite_list(lowpass_first_order(i_rate, dt_ms=RHYTHM_BIN_MS, cutoff_hz=10.0).tolist())
            )

    if not performance_mode:
        external_input_np = _as_numpy(external_input)
        if external_input_np.ndim == 1:
            input_t_ms = (np.arange(num_steps) * config.dt).tolist()
            input_mean_E = external_input_np.tolist()
            input_mean_I = external_input_np.tolist()
        else:
            input_t_ms = (np.arange(num_steps) * config.dt).tolist()
            if config.N_E > 0:
                input_mean_E = np.mean(external_input_np[:, : config.N_E], axis=1).tolist()
            else:
                input_mean_E = [0.0] * num_steps
            if config.N_I > 0:
                input_mean_I = np.mean(external_input_np[:, config.N_E :], axis=1).tolist()
            else:
                input_mean_I = [0.0] * num_steps
            for layer_slice in e_layer_slices:
                if layer_slice.stop > layer_slice.start:
                    input_mean_layers.append(
                        np.mean(external_input_np[:, layer_slice], axis=1).tolist()
                    )
                else:
                    input_mean_layers.append([0.0] * num_steps)
            if config.N_I > 0:
                input_mean_layers.append(np.mean(external_input_np[:, config.N_E :], axis=1).tolist())
        if source_spikes_e is not None:
            if config.N_E > 0:
                input_raw_spike_fraction_E = np.mean(
                    source_spikes_e.astype(float), axis=1
                ).tolist()
            else:
                input_raw_spike_fraction_E = [0.0] * num_steps
            if config.N_I > 0 and source_spikes_i is not None:
                input_raw_spike_fraction_I = np.mean(
                    source_spikes_i.astype(float), axis=1
                ).tolist()
            else:
                input_raw_spike_fraction_I = [0.0] * num_steps
            for layer_slice in e_layer_slices:
                if layer_slice.stop > layer_slice.start:
                    local_slice = slice(layer_slice.start, layer_slice.stop)
                    input_raw_spike_fraction_layers.append(
                        np.mean(
                            source_spikes_e[:, local_slice].astype(float), axis=1
                        ).tolist()
                    )
                    layer_spikes = source_spikes_e[:, local_slice]
                    max_sources = min(64, layer_spikes.shape[1])
                    if max_sources > 0:
                        sparse = layer_spikes[:, :max_sources]
                        step_idx, src_idx = np.nonzero(sparse)
                        input_raw_raster_times_ms_layers.append(
                            (step_idx.astype(float) * float(config.dt)).tolist()
                        )
                        input_raw_raster_ids_layers.append(src_idx.astype(int).tolist())
                    else:
                        input_raw_raster_times_ms_layers.append([])
                        input_raw_raster_ids_layers.append([])
                else:
                    input_raw_spike_fraction_layers.append([0.0] * num_steps)
                    input_raw_raster_times_ms_layers.append([])
                    input_raw_raster_ids_layers.append([])
            if config.N_I > 0 and source_spikes_i is not None:
                input_raw_spike_fraction_layers.append(
                    np.mean(source_spikes_i.astype(float), axis=1).tolist()
                )
                max_i_sources = min(32, source_spikes_i.shape[1])
                if max_i_sources > 0:
                    sparse_i = source_spikes_i[:, :max_i_sources]
                    step_idx_i, src_idx_i = np.nonzero(sparse_i)
                    input_raw_raster_times_ms_layers.append(
                        (step_idx_i.astype(float) * float(config.dt)).tolist()
                    )
                    input_raw_raster_ids_layers.append(src_idx_i.astype(int).tolist())
                else:
                    input_raw_raster_times_ms_layers.append([])
                    input_raw_raster_ids_layers.append([])
            input_spike_fraction_E = list(input_raw_spike_fraction_E)
            input_spike_fraction_I = list(input_raw_spike_fraction_I)
            input_spike_fraction_layers = [list(series) for series in input_raw_spike_fraction_layers]
        if input_type == "external_spike_train" and input_t_ms:
            t_input = np.array(input_t_ms, dtype=float)
            input_envelope = envelope_rate_hz(
                t_input,
                lambda0_hz=float(inputs.get("lambda0_hz", 30.0)),
                mod_depth=float(inputs.get("mod_depth", 0.5)),
                envelope_freq_hz=float(inputs.get("envelope_freq_hz", 5.0)),
                phase_rad=float(inputs.get("phase_rad", 0.0)),
            )
            input_envelope_hz = _finite_list(input_envelope.tolist())
            input_envelope_hz_layers = [list(input_envelope_hz) for _ in e_layer_slices]
            if config.N_I > 0:
                input_envelope_hz_layers.append([0.0] * len(input_envelope_hz))

        if input_type == "external_spike_train" and population_rate_t_ms:
            t_decode = np.array(population_rate_t_ms, dtype=float)
            decode_env = envelope_rate_hz(
                t_decode,
                lambda0_hz=float(inputs.get("lambda0_hz", 30.0)),
                mod_depth=float(inputs.get("mod_depth", 0.5)),
                envelope_freq_hz=float(inputs.get("envelope_freq_hz", 5.0)),
                phase_rad=float(inputs.get("phase_rad", 0.0)),
            )
            decode_envelope_hz = _finite_list(decode_env.tolist())
            decode_envelope_hz_layers = [list(decode_envelope_hz) for _ in e_layer_slices]
            if config.N_I > 0:
                decode_envelope_hz_layers.append([0.0] * len(decode_envelope_hz))

            if decode_lowpass_hz_E:
                decode_corr, decode_rmse = decode_fit_metrics(
                    np.array(decode_lowpass_hz_E, dtype=float),
                    np.array(decode_envelope_hz, dtype=float),
                    normalize=True,
                )
            for layer_idx, layer_decoded in enumerate(decode_lowpass_hz_layers):
                target = (
                    decode_envelope_hz_layers[layer_idx]
                    if layer_idx < len(decode_envelope_hz_layers)
                    else decode_envelope_hz
                )
                layer_corr, layer_rmse = decode_fit_metrics(
                    np.array(layer_decoded, dtype=float),
                    np.array(target, dtype=float),
                    normalize=True,
                )
                decode_corr_layers.append(float(layer_corr))
                decode_rmse_layers.append(float(layer_rmse))

    instruments = getattr(result, "instruments", None)
    if instruments is not None and instruments.times.size > 0:
        membrane_t_ms = instruments.times.tolist()
        if instruments.V is not None:
            v_matrix = instruments.V
            for idx in range(v_matrix.shape[1]):
                membrane_V_layers.append(v_matrix[:, idx].tolist())
            if membrane_V_layers:
                membrane_V_E = membrane_V_layers[0]
                if len(membrane_V_layers) > len(e_layer_slices):
                    membrane_V_I = membrane_V_layers[-1]
        if instruments.g_e is not None:
            g_e_matrix = instruments.g_e
            if g_e_matrix.shape[1] > 0:
                membrane_g_e_E = g_e_matrix[:, 0].tolist()
            if g_e_matrix.shape[1] > len(e_layer_slices):
                membrane_g_e_I = g_e_matrix[:, -1].tolist()
        if instruments.g_i is not None:
            g_i_matrix = instruments.g_i
            if g_i_matrix.shape[1] > 0:
                membrane_g_i_E = g_i_matrix[:, 0].tolist()
            if g_i_matrix.shape[1] > len(e_layer_slices):
                membrane_g_i_I = g_i_matrix[:, -1].tolist()

    analysis_ms = (time.perf_counter() - analysis_phase_start) * 1000.0

    response_build_start = time.perf_counter()
    response = {
        "spikes": spikes_response,
        "core_sim_ms": runtime_ms,
        "runtime_ms": runtime_ms,
        "num_steps": num_steps,
        "num_spikes": len(spikes.times),
        "total_e_spikes": total_e_spikes,
        "total_i_spikes": total_i_spikes,
        "spikes_truncated": spikes_truncated,
        "mean_rate_E": _finite_scalar(mean_rate_E),
        "mean_rate_I": _finite_scalar(mean_rate_I),
        "isi_cv_E": _finite_scalar(isi_cv_E),
        "isi_mean_E_ms": _finite_scalar(isi_mean_E_ms),
        "isi_inverse_E_hz": _finite_scalar(isi_inverse_E_hz),
        "autocorr_peak": _finite_scalar(autocorr_peak_val),
        "xcorr_peak": _finite_scalar(xcorr_peak_val),
        "coherence_peak": _finite_scalar(coherence_peak_val),
        "lagged_coherence": _finite_scalar(lagged_coherence_val),
        "population_rate_t_ms": population_rate_t_ms,
        "population_rate_hz_E": population_rate_hz_E,
        "population_rate_hz_I": population_rate_hz_I,
        "population_rate_hz_layers": population_rate_hz_layers,
        "decode_lowpass_hz_E": decode_lowpass_hz_E,
        "decode_lowpass_hz_I": decode_lowpass_hz_I,
        "decode_lowpass_hz_layers": decode_lowpass_hz_layers,
        "membrane_t_ms": membrane_t_ms,
        "membrane_V_E": membrane_V_E,
        "membrane_V_I": membrane_V_I,
        "membrane_V_layers": membrane_V_layers,
        "membrane_g_e_E": _finite_list(membrane_g_e_E),
        "membrane_g_i_E": _finite_list(membrane_g_i_E),
        "membrane_g_e_I": _finite_list(membrane_g_e_I),
        "membrane_g_i_I": _finite_list(membrane_g_i_I),
        "autocorr_lags_ms": autocorr_lags_ms,
        "autocorr_corr": _finite_list(autocorr_corr),
        "autocorr_lags_layers_ms": autocorr_lags_layers_ms,
        "autocorr_corr_layers": autocorr_corr_layers,
        "xcorr_lags_ms": xcorr_lags_ms,
        "xcorr_corr": _finite_list(xcorr_corr),
        "xcorr_lags_layers_ms": xcorr_lags_layers_ms,
        "xcorr_corr_layers": xcorr_corr_layers,
        "coherence_lags_ms": coherence_lags_ms,
        "coherence_corr": _finite_list(coherence_corr),
        "weights_hist_bins": weights_hist_bins,
        "weights_hist_counts_ee": weights_hist_counts_ee,
        "weights_hist_counts_ei": weights_hist_counts_ei,
        "weights_hist_counts_ie": weights_hist_counts_ie,
        "weights_hist_counts_ii": weights_hist_counts_ii,
        "weights_hist_blocks_ee": weights_hist_blocks_ee,
        "weights_hist_blocks_ei": weights_hist_blocks_ei,
        "weights_hist_blocks_ie": weights_hist_blocks_ie,
        "weights_hist_blocks_ii": weights_hist_blocks_ii,
        "weights_heatmap": _downsample_matrix(weight_mats.W, max_size=200) if not performance_mode else [],
        "psd_freqs_hz": psd_freqs_hz,
        "psd_power": _finite_list(psd_power),
        "psd_power_layers": psd_power_layers,
        "input_t_ms": input_t_ms,
        "input_mean_E": input_mean_E,
        "input_mean_I": input_mean_I,
        "input_mean_layers": input_mean_layers,
        "input_raw_spike_fraction_E": _finite_list(input_raw_spike_fraction_E),
        "input_raw_spike_fraction_I": _finite_list(input_raw_spike_fraction_I),
        "input_raw_spike_fraction_layers": input_raw_spike_fraction_layers,
        "input_raw_raster_times_ms_layers": input_raw_raster_times_ms_layers,
        "input_raw_raster_ids_layers": input_raw_raster_ids_layers,
        "input_envelope_hz": _finite_list(input_envelope_hz),
        "input_envelope_hz_layers": input_envelope_hz_layers,
        "input_spike_fraction_E": _finite_list(input_spike_fraction_E),
        "input_spike_fraction_I": _finite_list(input_spike_fraction_I),
        "input_spike_fraction_layers": input_spike_fraction_layers,
        "decode_envelope_hz": _finite_list(decode_envelope_hz),
        "decode_envelope_hz_layers": decode_envelope_hz_layers,
        "decode_corr": _finite_scalar(decode_corr),
        "decode_rmse": _finite_scalar(decode_rmse),
        "decode_corr_layers": _finite_list(decode_corr_layers),
        "decode_rmse_layers": _finite_list(decode_rmse_layers),
        "psd_peak_freq_hz": _finite_scalar(psd_peak_freq_hz),
        "psd_peak_bandwidth_hz": _finite_scalar(psd_peak_bandwidth_hz),
        "psd_peak_q_factor": _finite_scalar(psd_peak_q_factor),
        "layer_labels": layer_labels,
        "input_prep_ms": input_prep_ms,
        "weights_build_ms": weights_build_ms,
        "analysis_ms": analysis_ms,
    }
    response_build_ms = (time.perf_counter() - response_build_start) * 1000.0
    server_compute_ms = (time.perf_counter() - total_start) * 1000.0
    response["response_build_ms"] = response_build_ms
    response["server_compute_ms"] = server_compute_ms
    return response
