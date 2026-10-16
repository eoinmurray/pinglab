"""Causal spike-count readouts fitted exclusively on separate training trials."""

import numpy as np
from experiments.exp136 import recipe
from pingstore.contracts import write_json_atomic
from pingstore.stages import source_run, stage_run


def fit_decoder(features, targets, alpha):
    """Training data alone determines centring, scaling and ridge weights."""
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-8)
    target_mean = targets.mean(axis=0)
    x = (features - mean) / scale
    weights = np.linalg.solve(
        np.einsum("ni,nj->ij", x, x) + alpha * np.eye(x.shape[1]),
        np.einsum("ni,nj->ij", x, targets - target_mean),
    )
    return dict(mean=mean, scale=scale, target_mean=target_mean, weights=weights)


def predict(model, features):
    x = (features - model["mean"]) / model["scale"]
    return np.einsum("ni,ij->nj", x, model["weights"]) + model["target_mean"]


def r2_score(target, prediction):
    denominator = np.sum((target - target.mean()) ** 2)
    if denominator <= 1e-12:
        raise ValueError("cannot score a constant target")
    return float(1 - np.sum((target - prediction) ** 2) / denominator)


def causal_features(spikes, cfg):
    """No spike later than the prediction time enters any feature."""
    settings = cfg["decoder"]
    history = max(settings["lags_ms"]) + settings["bin_ms"]
    start_time = settings.get("observation_start_ms", history)
    if start_time < history:
        raise ValueError("observations cannot start before the complete history")
    times = np.arange(start_time, cfg["duration_ms"], settings["sample_ms"])
    columns = []
    for population in spikes:
        individual = settings.get("feature_mode", "population") == "individual"
        values = spikes[population]
        rates = (values if individual else values.mean(axis=-1)) * 1000.0 / cfg["dt_ms"]
        cumulative = np.concatenate(
            [np.zeros((1, *rates.shape[1:])), np.cumsum(rates, axis=0)]
        )
        for lag in settings["lags_ms"]:
            end = np.rint((times - lag) / cfg["dt_ms"]).astype(int)
            start = np.rint((times - lag - settings["bin_ms"]) / cfg["dt_ms"]).astype(
                int
            )
            denominator = (end - start).reshape(-1, *([1] * (rates.ndim - 1)))
            values = (cumulative[end] - cumulative[start]) / denominator
            columns.append(values if individual else values[..., None])
    return times, np.concatenate(columns, axis=-1)


def cue_samples(times, lane, cfg):
    """Exclude the first 100 ms plus the complete decoder history per cue."""
    history = max(cfg["decoder"]["lags_ms"]) + cfg["decoder"]["bin_ms"]
    masks = []
    for sender in ("S1", "S2"):
        segment = lane["cue_order"].index(sender) + 1
        start, end = segment * cfg["segment_ms"], (segment + 1) * cfg["segment_ms"]
        exclusion = cfg["decoder"].get("cue_exclusion_ms", 100 + history)
        masks.append((times >= start + exclusion) & (times < end))
    return masks


def score_readout(features, targets, metadata, cfg):
    train = metadata["split"] == 0
    if not train.any() or train.all():
        raise ValueError("independent train and test trials are required")
    model = fit_decoder(features[train], targets[train], cfg["decoder"]["ridge_alpha"])
    predicted = predict(model, features)
    test_trials = np.unique(metadata["trial"][~train])
    scores = np.empty((len(test_trials), 2, 2))
    for index, trial in enumerate(test_trials):
        for cue in range(2):
            keep = ~train & (metadata["trial"] == trial) & (metadata["cue"] == cue)
            if not keep.any():
                raise ValueError("missing counterbalanced held-out cue period")
            for sender in range(2):
                scores[index, cue, sender] = r2_score(
                    targets[keep, sender], predicted[keep, sender]
                )
    return scores, model, predicted


def gamma_power_fraction(spikes, cfg):
    """Population power in 30–80 Hz divided by power in 1–150 Hz."""
    width = round(1.0 / cfg["dt_ms"])
    binned = spikes.reshape(-1, width, *spikes.shape[1:]).sum(axis=1).mean(axis=-1)
    spectrum = abs(np.fft.rfft(binned - binned.mean(axis=0), axis=0)) ** 2
    frequency = np.fft.rfftfreq(len(binned), d=0.001)
    numerator = spectrum[(frequency >= 30) & (frequency <= 80)].sum(axis=0)
    denominator = spectrum[(frequency >= 1) & (frequency <= 150)].sum(axis=0)
    return numerator / np.maximum(denominator, 1e-12)


def attention_data(source, cfg):
    """Extract paired rows once; several fixed readout penalties can share them."""
    features = {c: {p: [] for p in ("S1", "S2", "R")} for c in cfg["conditions"]}
    targets = []
    input_features = {s: [] for s in ("S1", "S2")}
    gamma = {c: [] for c in cfg["conditions"]}
    metadata = {k: [] for k in ("split", "trial", "cue", "time_ms", "order")}
    for unit in cfg["units"]:
        with np.load(
            source.file(f"{unit['unit']}--inputs.npz"), allow_pickle=False
        ) as raw:
            signals = raw["signals"]
            for sender in ("S1", "S2"):
                input_times, x = causal_features(
                    {"input": raw[f"{sender}_sensory_drive"]},
                    dict(cfg, decoder=dict(cfg["decoder"], feature_mode="population")),
                )
                rows = [
                    x[mask, lane]
                    for lane, definition in enumerate(unit["lanes"])
                    for mask in cue_samples(input_times, definition, cfg)
                ]
                input_features[sender].append(np.concatenate(rows))
        selected = []
        for condition in cfg["conditions"]:
            with np.load(
                source.file(f"{unit['unit']}-{condition}--spikes.npz"),
                allow_pickle=False,
            ) as raw:
                gamma[condition].append(
                    np.stack(
                        [
                            gamma_power_fraction(raw[f"{sender}_E_spikes"], cfg)
                            for sender in ("S1", "S2")
                        ],
                        axis=-1,
                    )
                )
                for population in ("S1", "S2", "R"):
                    times, x = causal_features(
                        {p: raw[f"{population}_{p}_spikes"] for p in ("E", "I")},
                        cfg,
                    )
                    rows = []
                    for lane, definition in enumerate(unit["lanes"]):
                        for cue, mask in enumerate(cue_samples(times, definition, cfg)):
                            rows.append(x[mask, lane])
                            if condition == "coupled" and population == "R":
                                selected.append((lane, cue, mask))
                    features[condition][population].append(np.concatenate(rows))
        signal_times = np.rint(
            (times - cfg["decoder"]["target_delay_ms"]) / cfg["signal"]["bin_ms"]
        ).astype(int)
        for lane, cue, mask in selected:
            definition = unit["lanes"][lane]
            count = int(mask.sum())
            targets.append(signals[signal_times[mask], lane])
            for key, value in (
                ("split", 0 if definition["split"] == "train" else 1),
                ("trial", definition["trial"]),
                ("cue", cue),
                ("order", 0 if definition["cue_order"][0] == "S1" else 1),
            ):
                metadata[key].append(np.full(count, value))
            metadata["time_ms"].append(times[mask])
    targets = np.concatenate(targets).astype(float)
    metadata = {k: np.concatenate(v) for k, v in metadata.items()}
    return features, targets, metadata, input_features, gamma


def measure_attention(source, cfg, *, data=None):
    """Measure an explicit dataset without reserving or launching any stage."""
    features, targets, metadata, input_features, gamma = (
        attention_data(source, cfg) if data is None else data
    )
    models, predictions = {}, {}
    result = dict(
        schema="exp136.attention-analysis/v1",
        configuration=cfg,
        conditions={},
        local_encoding={},
        input_encoding={},
        samples=dict(
            train=int((metadata["split"] == 0).sum()),
            test=int((metadata["split"] == 1).sum()),
        ),
        limitation=cfg.get("limitation", "descriptive linear-readout results"),
    )
    result["sender_gamma_power_fraction"] = {
        c: np.concatenate(v).mean(axis=0).tolist() for c, v in gamma.items()
    }
    for condition in cfg["conditions"]:
        scores, model, predicted = score_readout(
            np.concatenate(features[condition]["R"]), targets, metadata, cfg
        )
        contrasts = np.stack(
            [scores[:, cue, cue] - scores[:, cue, 1 - cue] for cue in range(2)],
            axis=-1,
        )
        cue_gain = np.stack(
            [
                scores[:, sender, sender] - scores[:, 1 - sender, sender]
                for sender in range(2)
            ],
            axis=-1,
        )
        result["conditions"][condition] = dict(
            r2=scores.tolist(),
            mean_r2=scores.mean(axis=0).tolist(),
            selectivity=contrasts.tolist(),
            mean_selectivity=contrasts.mean(axis=0).tolist(),
            cue_gain=cue_gain.tolist(),
            mean_cue_gain=cue_gain.mean(axis=0).tolist(),
        )
        predictions[condition] = predicted
        models.update({f"{condition}_R_{k}": v for k, v in model.items()})
    result["local_encoding_by_condition"] = {}
    for condition in cfg["conditions"]:
        result["local_encoding_by_condition"][condition] = {}
        for sender, population in enumerate(("S1", "S2")):
            scores, model, _ = score_readout(
                np.concatenate(features[condition][population]), targets, metadata, cfg
            )
            local = scores[:, :, sender].mean(axis=1)
            result["local_encoding_by_condition"][condition][population] = dict(
                r2=local.tolist(),
                mean_r2=float(local.mean()),
                r2_by_cue=scores[:, :, sender].tolist(),
            )
            models.update(
                {f"local_{condition}_{population}_{k}": v for k, v in model.items()}
            )
    result["local_encoding"] = result["local_encoding_by_condition"]["coupled"]
    for sender, population in enumerate(("S1", "S2")):
        scores, model, _ = score_readout(
            np.concatenate(input_features[population]), targets, metadata, cfg
        )
        local = scores[:, :, sender].mean(axis=1)
        result["input_encoding"][population] = dict(
            r2=local.tolist(), mean_r2=float(local.mean())
        )
        models.update({f"input_{population}_{k}": v for k, v in model.items()})
    coupled = np.asarray(result["conditions"]["coupled"]["mean_r2"])
    if "zero" not in cfg["conditions"] or "jittered" not in cfg["conditions"]:
        result.update(selective_routing=None, timing_support=None)
        return result, models, dict(targets=targets, **metadata, **predictions)
    zero = np.asarray(result["conditions"]["zero"]["mean_r2"])
    jittered = np.asarray(result["conditions"]["jittered"]["mean_r2"])
    result["selective_routing"] = bool(
        all(v["mean_r2"] > 0 for v in result["local_encoding"].values())
        and all(
            coupled[c, c] > 0
            and coupled[c, c] > coupled[c, 1 - c]
            and coupled[c, c] > coupled[1 - c, c]
            and coupled[c, c] > zero[c, c]
            for c in range(2)
        )
    )
    result["timing_support"] = bool(
        result["selective_routing"]
        and all(
            v["mean_r2"] >= 0.1
            for v in result["local_encoding_by_condition"]["jittered"].values()
        )
        and all(
            coupled[c, c] > jittered[c, c]
            and coupled[c, c] - coupled[1 - c, c] > jittered[c, c] - jittered[1 - c, c]
            for c in range(2)
        )
    )
    return result, models, dict(targets=targets, **metadata, **predictions)


def analyse_attention(repo, identity, *, run_id=None):
    source = source_run(
        repo / ".pingstore", identity, stage="compute", experiment=recipe.SLUG
    )
    cfg = source.record["execution"]["configuration"]
    if cfg.get("schema") != "exp136.attention/v1" or not source.record["execution"].get(
        "paired_parameters_and_inputs_verified"
    ):
        raise ValueError("requires a completed, verified signal-routing run")
    with stage_run(
        repo,
        recipe.SLUG,
        "analyse",
        inputs={"compute": source},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        result, models, predictions = measure_attention(source, cfg)
        write_json_atomic(run.export / "results.json", result)
        np.savez_compressed(run.export / "decoders.npz", **models)
        np.savez_compressed(run.export / "predictions.npz", **predictions)
    return run.run_id
