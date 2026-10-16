"""Scientific checks for the fixed paired trial and its phase measurements."""

import numpy as np
import pytest
from experiments.exp136 import analyse, compute, recipe


def test_phase_estimator_recovers_period_and_leaves_silent_gaps():
    cfg = recipe.configuration()
    spikes = np.zeros((10000, 4), dtype=np.uint8)
    spikes[np.arange(100, 9900, 250)] = 1
    phase, _ = analyse.volley_phase(spikes, cfg)
    assert phase[600] - phase[350] == pytest.approx(1.0)
    assert np.isnan(phase[0]) and np.isnan(phase[-1])
    spikes[4000:6000] = 0
    phase, _ = analyse.volley_phase(spikes, cfg)
    assert np.isnan(phase[4500:5500]).all()
    silent, peaks = analyse.volley_phase(np.zeros_like(spikes), cfg)
    assert not len(peaks) and np.isnan(silent).all()


def test_metrics_distinguish_fixed_phase_from_relative_drift():
    cfg = recipe.configuration()
    time = np.arange(10000) * cfg["dt_ms"]
    reference = time / 25
    phases = {"R": reference, "S1": reference + 0.25, "S2": reference + time / 1000 * 5}
    peaks = {name: np.arange(10, 1000, 25) for name in phases}
    spikes = {"R_E_spikes": np.zeros((10000, 1, 4), dtype=np.uint8)}
    result = analyse.window_metrics(time, phases, peaks, spikes, 0, 1000, cfg)
    assert result["pairs"]["S1R"]["valid"]
    assert result["pairs"]["S1R"]["plv"] == pytest.approx(1.0)
    assert result["pairs"]["S1R"]["drift_cycles_s"] == pytest.approx(0.0, abs=1e-10)
    assert result["pairs"]["S2R"]["plv"] < 0.01
    assert result["pairs"]["S2R"]["drift_cycles_s"] == pytest.approx(5.0)


def test_raster_frequencies_use_each_full_500ms_window():
    cfg = recipe.configuration()
    peaks = np.r_[np.arange(10, 500, 25), np.arange(510, 1000, 20)]
    assert analyse.rhythm_summary(peaks, 0, 500, cfg)["frequency_hz"] == pytest.approx(
        40
    )
    assert analyse.rhythm_summary(peaks, 500, 1000, cfg)[
        "frequency_hz"
    ] == pytest.approx(50)
    assert analyse.rhythm_summary(peaks, 1000, 1500, cfg)["frequency_hz"] is None


def test_sliding_plv_recovers_fixed_phase_rejects_drift_and_masks_missing_data():
    cfg = recipe.configuration()
    time = np.arange(10000) * cfg["dt_ms"]
    centres, locked = analyse.sliding_plv(np.full(time.shape, 0.25), cfg)
    assert centres[0] == 100 and centres[-1] == 900
    assert np.allclose(np.diff(centres), 10)
    assert np.allclose(locked, 1)
    _, drifting = analyse.sliding_plv(time / 1000 * 5, cfg)
    assert np.allclose(drifting, 0, atol=1e-12)
    broken = np.full(time.shape, 0.25)
    broken[(time >= 300) & (time < 450)] = np.nan
    _, masked = analyse.sliding_plv(broken, cfg)
    assert np.isnan(masked[centres == 400]).all()
    assert masked[centres == 700] == pytest.approx(1)


def test_fixed_inputs_have_exclusive_identical_one_second_cues():
    cfg = recipe.configuration()
    cfg["input_channels"] = 4
    drives = compute.make_inputs(cfg)
    first, second = [drives[f"cue_{s}_drive"] for s in ("S1", "S2")]
    assert not first[:10000].any() and not first[20000:].any()
    assert not second[:20000].any() and not second[30000:].any()
    assert np.array_equal(first[10000:20000], second[20000:30000])
    assert not np.array_equal(first[10000:15000], first[15000:20000])
    assert cfg["conditions"] == ["coupled", "zero"]
    assert cfg["background_hz"] == {"S1": 33, "S2": 33, "R": 35}
    assert cfg["forward_strength_us"] == 0.045
    assert cfg["cue_to_e_strength_us"] == 0.06
    assert cfg["duration_ms"] == 4000


def test_zero_control_only_removes_sender_to_receiver_parameters():
    values = {
        f"{s}_E_to_R_{p}.weight": np.ones((2, 2))
        for s in ("S1", "S2")
        for p in ("E", "I")
    }
    values["cue_S1_E_to_S1_E.weight"] = np.ones((2, 2))
    coupled = compute.condition_parameters(values, "coupled")
    zero = compute.condition_parameters(values, "zero")
    assert all(np.array_equal(v, coupled[k]) for k, v in values.items())
    assert np.array_equal(
        zero["cue_S1_E_to_S1_E.weight"], values["cue_S1_E_to_S1_E.weight"]
    )
    assert all(not v.any() for k, v in zero.items() if k != "cue_S1_E_to_S1_E.weight")
    assert all(v.any() for v in values.values())


def test_fixed_network_has_no_modulation_or_feedback():
    graph = recipe.author_network().graph
    assert all(not p.get("modulation") for p in graph["projections"])
    assert not any(p["id"].startswith("R_E_to_S") for p in graph["projections"])


def test_attention_trials_pair_orders_and_separate_training_from_test():
    from experiments.exp136.attention_compute import make_task_inputs

    cfg = recipe.attention_configuration()
    cfg["input_channels"] = 8
    drives, signals, lanes = make_task_inputs(cfg, cfg["units"][0])
    assert signals.shape == (400, 4, 2)
    assert np.array_equal(signals[:, 0], signals[:, 1])
    for sender in ("S1", "S2", "R"):
        assert np.array_equal(
            drives[f"{sender}_sensory_drive"][:, 0],
            drives[f"{sender}_sensory_drive"][:, 1],
        )
    assert not np.array_equal(signals[:, 0], signals[:, 2])
    assert abs(signals).max() <= 1
    assert np.allclose(signals.mean(axis=0), 0, atol=1e-7)
    assert np.array_equal(
        drives["cue_S1_drive"][10000:20000, 0], drives["cue_S1_drive"][20000:30000, 1]
    )
    assert lanes[0]["cue_order"] != lanes[1]["cue_order"]
    trials = [t for unit in cfg["units"] for t in unit["trials"]]
    train = {t["noise_offset"] for t in trials if t["split"] == "train"}
    test = {t["noise_offset"] for t in trials if t["split"] == "test"}
    assert len(train) == len(test) == 3 and not train & test


def test_timing_control_preserves_per_cell_block_counts_and_breaks_volleys():
    from experiments.exp136.attention_compute import jitter_spikes

    cfg = recipe.attention_configuration()
    spikes = np.zeros((2000, 2, 80), dtype=np.uint8)
    spikes[np.arange(0, 2000, 180)] = 1
    shuffled = jitter_spikes(spikes, cfg, 42)
    assert np.array_equal(
        spikes.reshape(4, 500, 2, 80).sum(1), shuffled.reshape(4, 500, 2, 80).sum(1)
    )
    assert shuffled.sum(axis=-1).max() < spikes.sum(axis=-1).max() / 2
    assert np.array_equal(shuffled, jitter_spikes(spikes, cfg, 42))


def test_decoder_recovers_unseen_signal_without_using_test_scaling():
    from experiments.exp136.attention_analyse import fit_decoder, predict, r2_score

    rng = np.random.default_rng(44)
    x = rng.normal(size=(600, 10))
    y = np.einsum("ni,ij->nj", x, rng.normal(size=(10, 2)))
    model = fit_decoder(x[:400], y[:400], 1.0)
    assert r2_score(y[400:, 0], predict(model, x[400:])[:, 0]) > 0.99
    assert np.allclose(model["mean"], x[:400].mean(axis=0))
    assert r2_score(y[400:, 0], np.zeros(200)) < 0.01
    with pytest.raises(ValueError, match="constant"):
        r2_score(np.ones(10), np.ones(10))


def test_decoder_features_are_causal_and_cue_windows_counterbalance_time():
    from experiments.exp136.attention_analyse import causal_features, cue_samples

    cfg = recipe.attention_configuration()
    spikes = {p: np.zeros((40000, 1, 2), dtype=np.uint8) for p in ("E", "I")}
    times, baseline = causal_features(spikes, cfg)
    spikes["E"][20000:] = 1
    _, changed = causal_features(spikes, cfg)
    assert np.array_equal(baseline[times <= 2000], changed[times <= 2000])
    assert changed[times > 2000].any()
    first = cue_samples(times, dict(cue_order=["S1", "S2"]), cfg)
    second = cue_samples(times, dict(cue_order=["S2", "S1"]), cfg)
    assert np.array_equal(first[0], second[1]) and np.array_equal(first[1], second[0])
    assert times[first[0]].min() == 1160 and times[first[1]].min() == 2160


def test_readout_fit_is_unchanged_when_held_out_features_and_targets_change():
    from experiments.exp136.attention_analyse import score_readout

    rng = np.random.default_rng(12)
    features = rng.normal(size=(320, 10))
    targets = np.einsum("ni,ij->nj", features, rng.normal(size=(10, 2)))
    metadata = dict(
        split=np.r_[np.zeros(200), np.ones(120)],
        trial=np.r_[np.zeros(200), np.repeat(np.arange(3), 40)],
        cue=np.r_[np.zeros(200), np.tile(np.repeat([0, 1], 20), 3)],
    )
    cfg = recipe.attention_configuration()
    _, original, _ = score_readout(features, targets, metadata, cfg)
    changed = features.copy()
    changed[200:] += 100
    changed_targets = targets.copy()
    changed_targets[200:] *= -10
    _, modified, _ = score_readout(changed, changed_targets, metadata, cfg)
    assert all(np.array_equal(original[k], modified[k]) for k in original)


def test_search_is_reproducible_fine_is_local_and_test_streams_are_fresh():
    coarse = recipe.routing_candidates("coarse")
    assert len(coarse) == 17 and coarse == recipe.routing_candidates("coarse")
    first = coarse[0]
    assert first["forward_strength_us"] == recipe.FORWARD_US
    assert first["cue_to_e_strength_us"] == recipe.CUE_TO_E_US
    centre = coarse[4]
    fine = recipe.routing_candidates("fine", centre)
    assert len(fine) == 17
    for candidate in fine:
        for key, (lo, hi) in recipe.SEARCH_AXES.items():
            assert (
                max(lo, centre[key] * 0.65)
                <= candidate[key]
                <= min(hi, centre[key] * 1.35)
            )
    tuning = recipe.routing_configuration(first)
    confirmation = recipe.routing_configuration(
        dict(fine[0], candidate="confirmation"), confirmation=True
    )
    tuning_seeds = {t["noise_offset"] for u in tuning["units"] for t in u["trials"]}
    fresh_seeds = {
        t["noise_offset"] for u in confirmation["units"] for t in u["trials"]
    }
    assert not tuning_seeds & fresh_seeds
    assert len(fresh_seeds) == 6
    assert (
        tuning["background_hz"] == confirmation["background_hz"] == recipe.BACKGROUND_HZ
    )


def test_search_prioritises_readability_and_same_signal_cue_benefit():
    from experiments.exp136.routing_search import candidate_quality

    def result(local, scores):
        return dict(
            local_encoding={s: dict(mean_r2=local) for s in ("S1", "S2")},
            conditions=dict(coupled=dict(mean_r2=scores)),
        )

    unreadable = result(-0.1, [[0.9, -0.9], [-0.9, 0.9]])
    readable = result(0.5, [[0.4, 0.3], [0.2, 0.5]])
    assert candidate_quality(readable) > candidate_quality(unreadable)
    # S1 being inherently easier to decode must not count as cue selection.
    fixed_preference = result(0.5, [[0.8, 0.2], [0.8, 0.2]])
    assert candidate_quality(fixed_preference)[2] == pytest.approx(0)
    assert candidate_quality(readable)[2] == pytest.approx(0.2)


def test_search_scaling_preserves_masks_local_weights_and_source_arrays():
    from experiments.exp136.routing_search import scaled_parameters

    values = {
        f"{s}_E_to_R_{p}.weight": np.diag([recipe.FORWARD_US] * 2)
        for s in ("S1", "S2")
        for p in ("E", "I")
    }
    values.update(
        {
            f"cue_{s}_E_to_{s}_E.weight": np.diag([recipe.CUE_TO_E_US] * 2)
            for s in ("S1", "S2")
        }
    )
    values["S1_I_to_E.weight"] = np.array([[0.5, 0], [0.3, 0.4]])
    original = {k: v.copy() for k, v in values.items()}
    cfg = dict(forward_strength_us=0.09, cue_to_e_strength_us=0.03)
    scaled = scaled_parameters(values, cfg)
    for key, value in values.items():
        assert np.array_equal(value, original[key])
        assert np.array_equal(value == 0, scaled[key] == 0)
    assert np.array_equal(scaled["S1_I_to_E.weight"], original["S1_I_to_E.weight"])
    assert np.allclose(scaled["S1_E_to_R_I.weight"].sum(0), 0.09)
    assert np.allclose(scaled["cue_S1_E_to_S1_E.weight"].sum(0), 0.03)


def test_individual_features_retain_information_lost_by_population_average():
    from experiments.exp136.attention_analyse import causal_features

    cfg = recipe.attention_configuration()
    cfg["duration_ms"] = 100.0
    cfg["decoder"].update(bin_ms=10, lags_ms=[0], sample_ms=10)
    spikes = np.zeros((1000, 1, 2), dtype=np.uint8)
    for block in range(10):
        spikes[block * 100 + np.arange(5) * 10, 0, block % 2] = 1
    _, population = causal_features({"E": spikes}, cfg)
    cfg["decoder"]["feature_mode"] = "individual"
    times, individual = causal_features({"E": spikes}, cfg)
    assert np.allclose(population, population[0])
    assert individual.shape == (9, 1, 2)
    assert not np.array_equal(individual[0], individual[1])
    changed = spikes.copy()
    changed[500:] = 1
    _, later = causal_features({"E": changed}, cfg)
    assert np.array_equal(individual[times <= 50], later[times <= 50])


def test_readout_audit_scores_identical_times_and_targets_for_every_window():
    from experiments.exp136.attention_analyse import causal_features, cue_samples

    cfg = recipe.attention_configuration()
    prior = None
    for readout in recipe.readout_candidates():
        cfg["decoder"].update(readout)
        times, _ = causal_features({"E": np.zeros((40000, 1, 2))}, cfg)
        masks = cue_samples(times, dict(cue_order=["S1", "S2"]), cfg)
        selected = times[masks[0] | masks[1]]
        assert selected[0] == 1350
        if prior is not None:
            assert np.array_equal(selected, prior)
        prior = selected


def test_reconstruction_objective_prefers_better_decoding_to_large_gain():
    from experiments.exp136.readout_audit import reconstruction_quality

    def result(scores):
        scores = np.asarray(scores)
        return dict(
            local_encoding={s: dict(mean_r2=0.5) for s in ("S1", "S2")},
            conditions=dict(
                coupled=dict(
                    mean_r2=scores.tolist(),
                    mean_cue_gain=(np.diag(scores) - np.diag(scores[::-1])).tolist(),
                )
            ),
        )

    poor = result([[-0.1, -1], [-1, -0.1]])
    good = result([[0.3, 0.2], [0.2, 0.3]])
    assert reconstruction_quality(good) > reconstruction_quality(poor)


def test_receiver_refinement_freezes_encoding_and_reserves_new_final_sequences():
    network = recipe.routing_candidates("coarse")[0]
    candidates = recipe.receiver_candidates(network)
    readout = recipe.readout_candidates()[7]
    assert len(candidates) == 9
    development = recipe.receiver_configuration(candidates[0], readout)
    final = recipe.receiver_configuration(
        dict(candidates[0], candidate="refined-confirmation"),
        readout,
        confirmation=True,
    )
    assert development["signal"] == final["signal"]
    assert development["decoder"] == final["decoder"]
    assert {n["receiver_background_hz"] for n in candidates} == {29, 32, 35}
    dev = {t["noise_offset"] for u in development["units"] for t in u["trials"]}
    fresh = {t["noise_offset"] for u in final["units"] for t in u["trials"]}
    assert len(dev) == len(fresh) == 6 and not dev & fresh
    assert {t["split"] for u in development["units"] for t in u["trials"]} == {
        "train",
        "validation",
    }
    assert all(
        n["cue_to_e_strength_us"] == network["cue_to_e_strength_us"] for n in candidates
    )
