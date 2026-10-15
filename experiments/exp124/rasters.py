"""Matched single-utterance inference and measurements for the final SHD report."""

import argparse
import sys
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import numpy as np
import torch
from experiments.exp124 import recipe
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run
from snnlab.sim import (
    DenseArrayBinding,
    ExecutionSpec,
    GraphExecutor,
    RecordingSpec,
    SignalRecording,
    infer,
    plan_graph,
)


def record(identities, device, *, batch_encoder=None):
    if batch_encoder is None:
        from experiments.exp124.compute import batch as batch_encoder
    if not identities or len(identities) != 2:
        raise ValueError("sample-raster requires explicit COBA and PING compute sources")
    sources = {}
    for identity in identities:
        source = source_run(REPO / ".pingstore", identity, stage="compute", experiment=recipe.SLUG)
        cfg = source.record["execution"]["configuration"]
        if not cfg.get("full_dataset") or len(cfg["models"]) != 1:
            raise ValueError("expected one full-data condition per source")
        model_name = cfg["models"][0]
        if model_name in sources:
            raise ValueError("duplicate model source")
        sources[model_name] = source
    if set(sources) != set(recipe.MODELS):
        raise ValueError("both trained models are required")
    cfg = dict(sources["coba"].record["execution"]["configuration"])
    other_cfg = sources["ping"].record["execution"]["configuration"]
    if {k:v for k,v in cfg.items() if k != "models"} != {k:v for k,v in other_cfg.items() if k != "models"}:
        raise ValueError("trained scientific configurations differ")
    if sources["coba"].record["execution"]["dataset"] != sources["ping"].record["execution"]["dataset"]:
        raise ValueError("trained dataset selections differ")
    cfg.update(models=list(recipe.MODELS), record_spikes=True, compile_mode=None,
               phase_definition=recipe.PHASES["sample-raster"], inference_noise_seed=cfg["seed"] + 4,
               sample_selection="first validation utterance in source order; not selected by prediction or activity",
               weight_selection="final epoch", optimizer_updates=0)
    with np.load(sources["coba"].file("spikes.npz"), allow_pickle=False) as data:
        selected = int(np.flatnonzero(data["split"] == 1)[0])
        mask = data["event_sample"] == selected
        coordinates = np.unique(np.floor(data["event_time_ms"][mask] / cfg["dt_ms"]).astype(np.int64)
                                * cfg["n_input"] + data["event_channel"][mask])
        cfg.update(sample_class=int(data["labels"][selected]), official_train_index=int(data["official_train_indices"][selected]))
        events = [(coordinates // cfg["n_input"], coordinates % cfg["n_input"])]
    drive = batch_encoder(events, [0], cfg, device, np.random.default_rng(cfg["inference_noise_seed"]))
    arrays = {"input_spikes": drive[:, 0].cpu().numpy().astype(np.uint8)}
    with stage_run(REPO, recipe.SLUG, "compute", phase="sample-raster",
                   inputs={f"{m}_weights": s for m,s in sources.items()}, configuration=cfg) as run:
        with torch.no_grad():
            for model_name in recipe.MODELS:
                directory = sources[model_name].unit(model_name)
                populations = ("e1", "e2", "i1", "i2") if model_name == "ping" else ("e1", "e2")
                if (directory / "manifest.json").exists():
                    result = infer(ExecutionSpec(
                        kind="infer", graph=load_json(directory / "graph.json"), checkpoint=directory,
                        input_bindings=(DenseArrayBinding("drive", drive),),
                        recording=RecordingSpec(signals=tuple(SignalRecording(f"{pop.upper()}.spikes")
                                                             for pop in populations)),
                        seed=cfg["seed"], device=str(device), diagnostics=False,
                        projection_mode=cfg["projection_mode"], extension_validation=cfg["extension_validation"],
                        retain_parameters=False, retain_final_state=False, retain_runtime_state=False))
                    arrays[f"{model_name}_scores"] = result.outputs["class_scores"][0].cpu().numpy()
                    for pop in populations:
                        arrays[f"{model_name}_{pop}"] = result.recorded_signals[f"{pop.upper()}.spikes"][:, 0].numpy().astype(np.uint8)
                    run.record["execution"].setdefault("inference_metrics", {})[model_name] = result.metrics
                else:
                    # Retained pre-migration runs contain NPZ weights rather than native checkpoints.
                    bundle = recipe.author_network(cfg, model_name)
                    model = GraphExecutor(plan_graph(bundle.graph), seed=cfg["seed"],
                                          projection_mode=cfg["projection_mode"], extension_validation=cfg["extension_validation"]).to(device)
                    with np.load(directory / "final_weights.npz", allow_pickle=False) as weights:
                        parameters = model.parameter_map()
                        if set(parameters) != set(weights.files) or any(not np.isfinite(weights[k]).all() for k in weights.files):
                            raise ValueError("invalid trained parameter payload")
                        for name, parameter in parameters.items():
                            parameter.copy_(torch.as_tensor(weights[name], device=device))
                    output = model({"drive": drive}, diagnostics=False, **cfg["forward_retention"]).outputs
                    arrays[f"{model_name}_scores"] = output["class_scores"][0].cpu().numpy()
                    for pop in populations:
                        arrays[f"{model_name}_{pop}"] = output[f"{pop}_spikes"][:, 0].cpu().numpy().astype(np.uint8)
                    del model
        np.savez_compressed(run.export / "rasters.npz", **arrays)
        run.record["execution"].update(device=str(device), snnlab_version=version("snnlab"), optimizer_updates=0)
    return run.run_id


def measure(source, cfg):
    """Measure retained raw inference arrays without allocating a stage."""
    measurements = {"selection": cfg["sample_selection"], "class": cfg["sample_class"],
                    "official_train_index": cfg["official_train_index"], "noise_seed": cfg["inference_noise_seed"],
                    "duration_ms": cfg["duration_ms"], "dt_ms": cfg["dt_ms"], "models": {}}
    with np.load(source.file("rasters.npz"), allow_pickle=False) as data:
        steps = round(cfg["duration_ms"] / cfg["dt_ms"])
        for key in data.files:
            if not np.isfinite(data[key]).all():
                raise ValueError("nonfinite inference output")
            if not key.endswith("scores") and (data[key].shape[0] != steps or not np.isin(data[key], (0,1)).all()):
                raise ValueError("invalid raster")
        for model in recipe.MODELS:
            prediction = int(data[f"{model}_scores"].argmax())
            populations = {}
            for pop in ("e1", "e2", "i1", "i2") if model == "ping" else ("e1", "e2"):
                spikes = data[f"{model}_{pop}"]
                populations[pop] = {"neurons": int(spikes.shape[1]), "spikes": int(spikes.sum()),
                                    "rate_hz": float(spikes.mean() * 1000 / cfg["dt_ms"])}
            measurements["models"][model] = {"prediction": prediction, "correct": prediction == cfg["sample_class"],
                                               "populations": populations}
    return measurements


def analyse(identity):
    source = source_run(REPO / ".pingstore", identity, stage="compute", experiment=recipe.SLUG, phase="sample-raster")
    cfg = source.record["execution"]["configuration"]
    measurements = measure(source, cfg)
    with stage_run(REPO, recipe.SLUG, "analyse", inputs={"inference": source}, configuration=cfg) as run:
        write_json_atomic(run.export / "raster_measurements.json", measurements)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    print(analyse(parser.parse_args().source))
