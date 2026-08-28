"""Preserved numerical measurements of explicitly retained recordings."""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from .recipe import (
    COUPLING_DELAY_MS,
    CROSS_FAN_IN,
    DT_MS,
    E_TO_I_TAU_MS,
    E_TO_I_WEIGHT,
    INPUT_RATE_A_HZ,
    INPUT_RATE_B_HZ,
    K_EE,
    K_EI,
    N_E,
    N_I,
    PRC_PHASE_FRACTIONS,
    PRC_REFERENCE_MS,
    STATUS,
    detect_volleys,
    population_rate,
)


def interpolated_phase(peaks: np.ndarray, steps: int) -> np.ndarray:
    """Interpolate phase from zero to 2π between detected volleys."""
    phase = np.full(steps, np.nan)
    for left, right in zip(peaks[:-1], peaks[1:], strict=True):
        phase[left:right] = 2.0 * np.pi * np.arange(right - left) / (right - left)
    return phase


class RhythmSummary(TypedDict):
    volleys: int
    frequency_hz: float | None
    iei_cv: float | None


def rhythm_summary(peaks: np.ndarray) -> RhythmSummary:
    intervals_ms = np.diff(peaks) * DT_MS
    if intervals_ms.size == 0:
        return {"volleys": int(peaks.size), "frequency_hz": None, "iei_cv": None}
    mean_interval_ms = float(intervals_ms.mean())
    return {
        "volleys": int(peaks.size),
        "frequency_hz": 1_000.0 / mean_interval_ms,
        "iei_cv": float(intervals_ms.std() / mean_interval_ms),
    }


def inhibitory_cycle_summary(
    spikes: np.ndarray,
    excitatory_peaks: np.ndarray,
) -> dict[str, float | int]:
    """Summarize inhibitory spikes per neuron between excitatory volleys."""
    cycle_counts = [
        spikes[left:right, 0].sum(axis=0)
        for left, right in zip(excitatory_peaks[:-1], excitatory_peaks[1:], strict=True)
    ]
    if not cycle_counts:
        return {
            "cycles": 0,
            "mean_spikes_per_neuron": 0.0,
            "minimum": 0,
            "maximum": 0,
        }
    counts = np.concatenate(cycle_counts)
    return {
        "cycles": len(cycle_counts),
        "mean_spikes_per_neuron": float(counts.mean()),
        "minimum": int(counts.min()),
        "maximum": int(counts.max()),
    }


def analyse_uncoupled(recordings: dict[str, np.ndarray]) -> dict[str, object]:
    e_a = population_rate(recordings["population_0"], N_E)
    i_a = population_rate(recordings["population_1"], N_I)
    e_b = population_rate(recordings["population_2"], N_E)
    i_b = population_rate(recordings["population_3"], N_I)
    peaks_a = detect_volleys(e_a)
    peaks_b = detect_volleys(e_b)
    summary_a = rhythm_summary(peaks_a)
    summary_b = rhythm_summary(peaks_b)
    inhibition_a = inhibitory_cycle_summary(recordings["population_1"], peaks_a)
    inhibition_b = inhibitory_cycle_summary(recordings["population_3"], peaks_b)
    phase_a = interpolated_phase(peaks_a, len(e_a))
    phase_b = interpolated_phase(peaks_b, len(e_b))
    valid = np.isfinite(phase_a) & np.isfinite(phase_b)
    phase_difference = np.full_like(phase_a, np.nan)
    phase_difference[valid] = np.angle(np.exp(1j * (phase_a[valid] - phase_b[valid])))
    valid_phase = phase_difference[valid]
    drift_wraps = int(np.count_nonzero(np.abs(np.diff(valid_phase)) > np.pi))

    for name, summary in (("A", summary_a), ("B", summary_b)):
        if summary["volleys"] < 20 or summary["iei_cv"] is None:
            raise RuntimeError(f"PING {name} did not produce a sustained rhythm")
        if float(summary["iei_cv"]) > 0.2:
            raise RuntimeError(f"PING {name} rhythm was too irregular")
    frequency_a = summary_a["frequency_hz"]
    frequency_b = summary_b["frequency_hz"]
    if frequency_a is None or frequency_b is None:
        raise RuntimeError("the uncoupled rhythms have no complete volley intervals")
    if not (30.0 <= frequency_a <= 80.0 and 30.0 <= frequency_b <= 80.0):
        raise RuntimeError("the uncoupled rhythms were outside the gamma band")
    if abs(frequency_a - frequency_b) < 0.5:
        raise RuntimeError("the uncoupled PING rhythms were not frequency-detuned")
    if drift_wraps < 2:
        raise RuntimeError("the uncoupled relative phase did not repeatedly wrap")
    for name, inhibition in (("A", inhibition_a), ("B", inhibition_b)):
        if inhibition["minimum"] != 1 or inhibition["maximum"] != 1:
            raise RuntimeError(
                f"PING {name} did not produce exactly one inhibitory spike "
                "per neuron per cycle before coupling"
            )

    return {
        "rate_e_a": e_a,
        "rate_i_a": i_a,
        "rate_e_b": e_b,
        "rate_i_b": i_b,
        "peaks_a": peaks_a,
        "peaks_b": peaks_b,
        "phase_difference": phase_difference,
        "network_a": summary_a,
        "network_b": summary_b,
        "inhibition_a": inhibition_a,
        "inhibition_b": inhibition_b,
        "drift_wraps": drift_wraps,
    }


def population_volley_events(
    spikes: np.ndarray,
    *,
    start: int,
    stop: int,
) -> list[dict[str, float | int]]:
    """Group adjacent occupied timesteps into population spike volleys."""
    counts = spikes[start:stop, 0].sum(axis=1)
    occupied = np.flatnonzero(counts)
    if occupied.size == 0:
        return []
    groups = np.split(occupied, np.flatnonzero(np.diff(occupied) > 1) + 1)
    events = []
    for group in groups:
        group_counts = counts[group]
        centre = float(np.average(group + start, weights=group_counts))
        events.append(
            {
                "time_ms": centre * DT_MS,
                "spikes": int(group_counts.sum()),
            }
        )
    return events


def analyse_pathway_branch(
    recordings: dict[str, np.ndarray],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Measure relative-phase drift after coupling onset."""
    rate_a = population_rate(recordings["population_0"], N_E)
    rate_b = population_rate(recordings["population_2"], N_E)
    peaks_a = detect_volleys(rate_a, burn_ms=0.0)
    peaks_b = detect_volleys(rate_b, burn_ms=0.0)
    phase_a = interpolated_phase(peaks_a, len(rate_a))
    phase_b = interpolated_phase(peaks_b, len(rate_b))
    valid = np.isfinite(phase_a) & np.isfinite(phase_b)
    valid_steps = np.flatnonzero(valid)
    wrapped = np.angle(np.exp(1j * (phase_a[valid] - phase_b[valid])))
    unwrapped_cycles = np.unwrap(wrapped) / (2.0 * np.pi)
    unwrapped_cycles -= unwrapped_cycles[0]
    time_ms = valid_steps * DT_MS
    final = time_ms >= time_ms[-1] - 500.0
    final_time_s = time_ms[final] / 1_000.0
    final_cycles = unwrapped_cycles[final]
    drift_rate = float(np.polyfit(final_time_s, final_cycles, 1)[0])
    concentration = float(abs(np.mean(np.exp(1j * wrapped[final]))))
    locked = abs(drift_rate) < 0.25 and concentration > 0.95
    locked_phase = (
        float(np.angle(np.mean(np.exp(1j * wrapped[final])))) if locked else None
    )
    summary_a = rhythm_summary(peaks_a)
    summary_b = rhythm_summary(peaks_b)
    record = {
        "PING_A": summary_a,
        "PING_B": summary_b,
        "final_drift_rate_cycles_per_s": drift_rate,
        "final_phase_concentration": concentration,
        "phase_locked": locked,
        "locked_phase_rad": locked_phase,
    }
    traces = {
        "time_ms": time_ms,
        "unwrapped_phase_change_cycles": unwrapped_cycles,
    }
    return record, traces


def analyse_event_aligned_mechanism(
    recordings: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Resolve the first measurable E-to-E correction after coupling begins."""
    baseline = recordings["none"]
    coupled = recordings["e_to_e"]
    source_rate = population_rate(coupled["population_0"], N_E)
    source_peaks = detect_volleys(source_rate, burn_ms=0.0)
    delay_steps = round(COUPLING_DELAY_MS / DT_MS)
    left_steps = round(5.0 / DT_MS)
    right_steps = round(17.0 / DT_MS)
    candidates = source_peaks[
        (source_peaks + delay_steps >= left_steps)
        & (source_peaks + delay_steps + right_steps < len(source_rate))
    ]
    if candidates.size == 0:
        raise RuntimeError("no source volley has a complete mechanism window")
    source_step = int(candidates[0])
    arrival_step = source_step + delay_steps

    baseline_target_peaks = detect_volleys(
        population_rate(baseline["population_2"], N_E),
        burn_ms=0.0,
    )
    coupled_target_peaks = detect_volleys(
        population_rate(coupled["population_2"], N_E),
        burn_ms=0.0,
    )
    baseline_next = baseline_target_peaks[baseline_target_peaks > arrival_step]
    coupled_next = coupled_target_peaks[coupled_target_peaks > arrival_step]
    if baseline_next.size == 0 or coupled_next.size == 0:
        raise RuntimeError("no target volley follows the selected coupling event")
    baseline_next_step = int(baseline_next[0])
    coupled_next_step = int(coupled_next[0])

    start = arrival_step - left_steps
    stop = arrival_step + right_steps
    window = slice(start, stop)
    traces = {
        "time_from_arrival_ms": (np.arange(start, stop) - arrival_step) * DT_MS,
        "incoming_e_to_e_conductance": coupled["PING_A_E_to_PING_B_E_K_EE.conductance"][
            window, 0
        ].mean(axis=1),
        "baseline_target_e_rate": population_rate(baseline["population_2"], N_E)[
            window
        ],
        "coupled_target_e_rate": population_rate(coupled["population_2"], N_E)[window],
        "baseline_target_i_rate": population_rate(baseline["population_3"], N_I)[
            window
        ],
        "coupled_target_i_rate": population_rate(coupled["population_3"], N_I)[window],
        "baseline_inhibition_to_e": baseline["PING_B_I_to_E.conductance"][
            window, 0
        ].mean(axis=1),
        "coupled_inhibition_to_e": coupled["PING_B_I_to_E.conductance"][window, 0].mean(
            axis=1
        ),
    }
    record = {
        "source_network": "PING_A",
        "target_network": "PING_B",
        "source_volley_ms_after_coupling": source_step * DT_MS,
        "arrival_ms_after_coupling": arrival_step * DT_MS,
        "baseline_next_target_volley_ms_after_coupling": baseline_next_step * DT_MS,
        "coupled_next_target_volley_ms_after_coupling": coupled_next_step * DT_MS,
        "next_target_volley_advance_ms": (baseline_next_step - coupled_next_step)
        * DT_MS,
    }
    return record, traces


def experiment_record(
    analysis: dict[str, object],
    phase_response: dict[str, object],
    pathway_comparison: dict[str, object],
    event_aligned_mechanism: dict[str, object],
) -> dict[str, object]:
    return {
        "status": STATUS,
        "completed_methods": [1, 2, 3, 4, 5],
        "simulation_run": True,
        "network": {
            "local_circuit": "matched E-to-I-to-E PING",
            "populations_per_network": {"E": N_E, "I": N_I},
            "detuning_input_rates_hz": {
                "PING_A": INPUT_RATE_A_HZ,
                "PING_B": INPUT_RATE_B_HZ,
            },
            "cross_network_projections": ["E-to-E", "E-to-I"],
            "reciprocal": True,
            "exact_fan_in_per_target": CROSS_FAN_IN,
            "weights": {"K_EE": K_EE, "K_EI": K_EI},
            "delay_ms": COUPLING_DELAY_MS,
            "local_e_to_i": {
                "weight": E_TO_I_WEIGHT,
                "ampa_tau_ms": E_TO_I_TAU_MS,
            },
        },
        "uncoupled": {
            "PING_A": analysis["network_a"],
            "PING_B": analysis["network_b"],
            "inhibitory_spikes_per_cycle": {
                "PING_A": analysis["inhibition_a"],
                "PING_B": analysis["inhibition_b"],
            },
            "phase_wraps": analysis["drift_wraps"],
        },
        "phase_response": phase_response,
        "pathway_comparison": pathway_comparison,
        "event_aligned_mechanism": event_aligned_mechanism,
        "remaining_methods_unrun": [],
    }


def analyse_phase_response(baseline, probes):
    """Measure retained baseline and every scheduled probe, without simulation."""
    baseline_e = baseline["population_0"]
    baseline_i = baseline["population_1"]
    baseline_peaks = detect_volleys(population_rate(baseline_e, N_E))
    reference_step = round(PRC_REFERENCE_MS / DT_MS)
    left_index = int(np.searchsorted(baseline_peaks, reference_step) - 1)
    if left_index < 0 or left_index + 1 >= len(baseline_peaks):
        raise RuntimeError("no complete baseline cycle near the PRC reference time")
    left = int(baseline_peaks[left_index])
    baseline_next = int(baseline_peaks[left_index + 1])
    period_steps = baseline_next - left

    responses: dict[str, list[dict[str, float | int | None]]] = {"E": [], "I": []}
    representative_specs = {
        ("E", 0.70): "e_late_advance",
        ("I", 0.08): "i_early_no_doublet",
        ("I", 0.12): "i_early_doublet",
    }
    representative_cases: dict[str, dict[str, object]] = {}
    strongest_i_delay_steps = 0
    early_i_pulse_example: dict[str, object] | None = None
    for target in ("E", "I"):
        for index, fraction in enumerate(PRC_PHASE_FRACTIONS):
            arrival = left + round(float(fraction) * period_steps)
            perturbed = probes(f"prc-{target}-{index:02d}")
            perturbed_e = perturbed["population_0"]
            perturbed_i = perturbed["population_1"]
            perturbed_peaks = detect_volleys(population_rate(perturbed_e, N_E))
            candidates = perturbed_peaks[perturbed_peaks > arrival]
            if candidates.size == 0:
                raise RuntimeError(f"no E volley followed the {target}-targeted pulse")
            perturbed_next = int(candidates[0])
            shift_steps = baseline_next - perturbed_next
            response = {
                "pulse_phase_fraction": (arrival - left) / period_steps,
                "pulse_phase_rad": 2.0 * np.pi * (arrival - left) / period_steps,
                "next_volley_shift_ms": shift_steps * DT_MS,
                "next_volley_phase_shift_rad": 2.0 * np.pi * shift_steps / period_steps,
            }
            if target == "I":
                i_events = population_volley_events(
                    perturbed_i,
                    start=left,
                    stop=perturbed_next,
                )
                response["i_volleys_before_next_e"] = len(i_events)
                response["second_i_volley_latency_ms"] = (
                    i_events[1]["time_ms"] - i_events[0]["time_ms"]
                    if len(i_events) >= 2
                    else None
                )
            responses[target].append(response)
            case_name = representative_specs.get((target, round(float(fraction), 2)))
            if case_name is not None:
                representative_cases[case_name] = {
                    "target": target,
                    "pulse_phase_fraction": response["pulse_phase_fraction"],
                    "arrival_step": arrival,
                    "next_e_step": perturbed_next,
                    "rate_e": population_rate(perturbed_e, N_E),
                    "rate_i": population_rate(perturbed_i, N_I),
                    "i_volleys_before_next_e": response.get("i_volleys_before_next_e"),
                }
                if case_name == "i_early_doublet":
                    representative_cases[case_name].update(
                        {
                            "i_voltage": perturbed["PING_A_I.voltage"][:, 0].mean(
                                axis=1
                            ),
                            "local_e_to_i_conductance": perturbed[
                                "PING_A_E_to_I.conductance"
                            ][:, 0].mean(axis=1),
                            "probe_e_to_i_conductance": perturbed[
                                "probe_E_to_PING_A_I_K_EI.conductance"
                            ][:, 0].mean(axis=1),
                        }
                    )
            if target == "I" and shift_steps < strongest_i_delay_steps:
                strongest_i_delay_steps = shift_steps
                event_stop = perturbed_next + round(2.0 / DT_MS)
                early_i_pulse_example = {
                    "pulse_phase_fraction": response["pulse_phase_fraction"],
                    "pulse_arrival_ms": arrival * DT_MS,
                    "baseline_next_e_volley_ms": baseline_next * DT_MS,
                    "perturbed_next_e_volley_ms": perturbed_next * DT_MS,
                    "baseline_i_volleys": population_volley_events(
                        baseline_i,
                        start=left,
                        stop=event_stop,
                    ),
                    "perturbed_i_volleys": population_volley_events(
                        perturbed_i,
                        start=left,
                        stop=event_stop,
                    ),
                }

    if early_i_pulse_example is None:
        raise RuntimeError("no I-targeted pulse delayed the next excitatory volley")
    if len(representative_cases) != len(representative_specs):
        raise RuntimeError("not all representative PRC cases were recorded")

    record = {
        "network": "PING_A",
        "baseline_cycle_start_ms": left * DT_MS,
        "baseline_cycle_period_ms": period_steps * DT_MS,
        "pulse": {
            "source_channels": N_E,
            "exact_fan_in": CROSS_FAN_IN,
            "arrival_delay_ms": COUPLING_DELAY_MS,
            "e_target_strength": K_EE,
            "i_target_strength": K_EI,
        },
        "positive_response_means": "next excitatory volley advanced",
        "sampling": {
            "coarse_pilot_interval_fraction": 0.1,
            "refined_region_fraction": [0.02, 0.30],
            "refined_interval_fraction": 0.02,
            "reason": "coarse pilot located the inhibitory transition near phase 0.1",
        },
        "responses": responses,
        "early_i_pulse_example": early_i_pulse_example,
    }
    illustration = {
        "left_step": left,
        "baseline_next_step": baseline_next,
        "baseline_rate_e": population_rate(baseline_e, N_E),
        "cases": representative_cases,
    }
    return record, illustration
