"""Run the paired input-rate sweep from MNIST and a pinned trained network."""

import argparse
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import numpy as np
import torch
from experiments.exp121 import recipe as R
from experiments.helpers.checkpoints import pinned_checkpoint
from pingstore.stages import source_run, stage_run
from snnlab.sim.execution import (  # noqa: TID251
    GraphExecutor,
    import_legacy_parameters_v1,
    plan_graph,
)
from torchvision.datasets import MNIST


def compute(dataset_root):
    torch.set_num_threads(2)
    dataset = MNIST(str(dataset_root), train=False, download=True)
    labels = dataset.targets.numpy()
    rng = np.random.default_rng(R.CONFIG["sample_seed"])
    indices = np.concatenate(
        [
            rng.choice(
                np.flatnonzero(labels == k), R.CONFIG["images_per_class"], replace=False
            )
            for k in range(10)
        ]
    )
    pixels = (
        dataset.data[indices].numpy().reshape(len(indices), 784).astype(np.float32)
        / 255
    )
    labels = labels[indices]
    bank = source_run(ROOT / ".pingstore", R.BANK, stage="compute", experiment="exp022")
    weights = pinned_checkpoint(
        bank.unit("ping__variable_rate__seed42"),
        role="best_validation",
        filename="weights.pth",
        sha256=R.CHECKPOINT_SHA256,
    )
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    cfg = R.CONFIG
    steps = round(cfg["duration_ms"] / cfg["dt_ms"])
    with stage_run(
        ROOT, "exp121", "compute", inputs={"bank": bank}, configuration=cfg
    ) as run:
        bundle = R.author_network()
        bundle.write(run.export / "network.bundle")
        model = GraphExecutor(plan_graph(bundle.graph), seed=cfg["trained_seed"])
        imported = import_legacy_parameters_v1(bundle.graph, checkpoint)
        with torch.no_grad():
            for name, param in model.parameter_map().items():
                param.copy_(imported.parameters[name])
        output = np.empty((len(R.RATES), len(indices), steps, 10), dtype=np.uint8)
        i_counts = np.empty((len(R.RATES), len(indices), steps), dtype=np.uint16)
        hashes = []
        started = time.perf_counter()
        with torch.inference_mode():
            for trial, index in enumerate(indices):
                generator = torch.Generator().manual_seed(
                    cfg["encoding_seed_offset"] + int(index)
                )
                uniforms = torch.rand(steps, 1, 784, generator=generator)
                trial_hashes = []
                for k, rate in enumerate(R.RATES):
                    probability = (
                        torch.from_numpy(pixels[trial])[None, None]
                        * rate
                        * cfg["dt_ms"]
                        / 1000
                    )
                    encoded = (uniforms < probability).float()
                    trial_hashes.append(
                        hashlib.sha256(encoded.numpy().tobytes()).hexdigest()
                    )
                    result = model(
                        {"input_spikes": encoded},
                        recording_fields=["out_spikes", "i_spikes"],
                    )
                    output[k, trial] = result.recordings["out_spikes"][:, 0].numpy()
                    i_counts[k, trial] = (
                        result.recordings["i_spikes"][:, 0].sum(dim=1).numpy()
                    )
                hashes.append(trial_hashes)
                if (trial + 1) % 10 == 0:
                    print(f"{trial + 1}/{len(indices)} images complete", flush=True)
        np.savez_compressed(
            run.export / "recording.npz",
            indices=indices,
            labels=labels,
            pixels=pixels,
            output_spikes=output,
            i_population_spike_counts=i_counts,
            input_rates_hz=R.RATES,
        )
        run.record["execution"].update(
            executor="snnsim.GraphExecutor",
            checkpoint_sha256=R.CHECKPOINT_SHA256,
            input_sha256=hashes,
            simulation_seconds=time.perf_counter() - started,
            dataset="official MNIST test",
            dataset_root=str(dataset_root.resolve()),
            dataset_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (dataset_root / "MNIST/raw").glob("t10k-*-ubyte")
            },
        )
    return run.run_id


def compute_internal(identity, studies=R.INTERNAL_STUDIES):
    """Reuse a validated input study's samples and baseline; run new conditions."""
    torch.set_num_threads(2)
    source = source_run(
        ROOT / ".pingstore", identity, stage="compute", experiment="exp121"
    )
    cfg = source.record["execution"]["configuration"]
    if cfg != {
        **R.CONFIG,
        "input_rates_hz": list(R.RATES),
        "peak_window_ms": [0, 400],
        "rate_window_ms": [0, 400],
        "sensitivity_thresholds_hz": [10, 25, 50],
    }:
        raise ValueError("Input study configuration does not match the paired protocol")
    reference = source.record["inputs"]["bank"]
    bank = source_run(
        ROOT / ".pingstore",
        reference["run_id"],
        stage="compute",
        experiment="exp022",
        reference=reference,
    )
    weights = pinned_checkpoint(
        bank.unit("ping__variable_rate__seed42"),
        role="best_validation",
        filename="weights.pth",
        sha256=R.CHECKPOINT_SHA256,
    )
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    with np.load(source.file("recording.npz")) as saved:
        indices, labels, pixels = (
            saved[k].copy() for k in ("indices", "labels", "pixels")
        )
        baseline_out = saved["output_spikes"][0].copy()
        baseline_i = saved["i_population_spike_counts"][0].copy()
        if not np.array_equal(saved["input_rates_hz"], R.RATES):
            raise ValueError("Input condition order mismatch")
    config = dict(
        R.CONFIG,
        intervention="internal_parameter_sweeps",
        studies=studies,
        control_values={study: R.CONTROLS[study] for study in studies},
        weight_policy="fixed_peak_except_explicit_I_to_E_scaling",
        inhibitory_strength_projection="I_to_E",
        baseline_e_threshold_mv=-50.0,
    )
    with stage_run(
        ROOT,
        "exp121",
        "compute",
        inputs={"input_study": source, "bank": bank},
        configuration=config,
    ) as run:
        started = time.perf_counter()
        hashes = {}
        for study in studies:
            controls = R.CONTROLS[study]
            models = []
            for scale in controls[1:]:
                bundle = R.author_network(study, scale)
                bundle.write(run.export / f"{study}-{scale:g}.bundle")
                model = GraphExecutor(plan_graph(bundle.graph), seed=42)
                imported = import_legacy_parameters_v1(bundle.graph, checkpoint)
                with torch.no_grad():
                    for name, param in model.parameter_map().items():
                        param.copy_(imported.parameters[name])
                    if study == "inhibition":
                        projection = next(
                            p
                            for p in bundle.graph["projections"]
                            if p["id"] == "I_to_E"
                        )
                        for name in projection["parameters"]:
                            model.parameter_map()[name].mul_(scale)
                models.append(model)
            output = np.empty((len(controls), *baseline_out.shape), dtype=np.uint8)
            population = np.empty((len(controls), *baseline_i.shape), dtype=np.uint16)
            output[0], population[0] = baseline_out, baseline_i
            hashes[study] = []
            with torch.inference_mode():
                for trial, index in enumerate(indices):
                    gen = torch.Generator().manual_seed(
                        R.CONFIG["encoding_seed_offset"] + int(index)
                    )
                    probability = (
                        torch.from_numpy(pixels[trial])[None, None]
                        * 5
                        * R.CONFIG["dt_ms"]
                        / 1000
                    )
                    encoded = (
                        torch.rand(baseline_out.shape[1], 1, 784, generator=gen)
                        < probability
                    ).float()
                    digest = hashlib.sha256(encoded.numpy().tobytes()).hexdigest()
                    if digest != source.record["execution"]["input_sha256"][trial][0]:
                        raise ValueError("Paired input mismatch")
                    hashes[study].append(digest)
                    for k, model in enumerate(models, start=1):
                        result = model(
                            {"input_spikes": encoded},
                            recording_fields=["out_spikes", "i_spikes"],
                        )
                        output[k, trial] = result.recordings["out_spikes"][:, 0].numpy()
                        population[k, trial] = (
                            result.recordings["i_spikes"][:, 0].sum(dim=1).numpy()
                        )
                    if (trial + 1) % 10 == 0:
                        print(f"{study}: {trial + 1}/100 images", flush=True)
            np.savez_compressed(
                run.export / f"{study}--recording.npz",
                indices=indices,
                labels=labels,
                pixels=pixels,
                output_spikes=output,
                i_population_spike_counts=population,
                control_values=controls,
            )
        run.record["execution"].update(
            input_sha256=hashes,
            checkpoint_sha256=R.CHECKPOINT_SHA256,
            baseline_reused=True,
            new_presentations=len(indices)
            * sum(len(R.CONTROLS[s]) - 1 for s in studies),
            simulation_seconds=time.perf_counter() - started,
        )
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--internal-from",
        help="Validated input-rate compute run supplying paired samples and baseline",
    )
    parser.add_argument(
        "--studies", nargs="+", choices=R.INTERNAL_STUDIES, default=R.INTERNAL_STUDIES
    )
    args = parser.parse_args()
    if args.internal_from:
        compute_internal(args.internal_from, tuple(dict.fromkeys(args.studies)))
    else:
        compute(args.dataset_root)
