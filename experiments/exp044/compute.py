"""Evaluate an explicit trained bank through native graphs; never train or publish."""

from __future__ import annotations

import argparse
import os
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
import torch
import torch.nn.functional as F
from experiments.exp044 import evidence, inputs, recipe
from pingstore.contracts import PingstoreError, write_json_atomic
from snnlab.sim.datasets import load_dataset
from snnlab.sim.encoders import encode_images_poisson
from snnlab.sim.execution import GraphExecutor, plan_graph


def evaluate_cell(
    train: Path, checkpoint: dict, cell: dict, common: dict, cfg: dict, export: Path
) -> list[dict]:
    """Bind the verified final checkpoint and evaluate independent presentations."""
    state = torch.load(
        train / checkpoint["filename"], map_location="cpu", weights_only=True
    )
    n_e, n_i = common["n_hidden"], common["n_inh"]
    shapes = {
        "W_ff.0": (common["n_in"], n_e),
        "W_ff.1": (n_e, common["n_out"]),
        "W_ei.1": (n_e, n_i),
        "W_ie.1": (n_i, n_e),
        "W_ee.1": (n_e, n_e),
        "W_ii.1": (n_i, n_i),
    }
    if not isinstance(state, dict) or set(state) != set(shapes):
        raise PingstoreError(
            "exp044 checkpoint must contain exactly the six audit matrices"
        )
    for name, shape in shapes.items():
        value = state[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or tuple(value.shape) != shape
            or not torch.isfinite(value).all()
        ):
            raise PingstoreError(f"invalid checkpoint matrix {name}")
        if name in ("W_ee.1", "W_ii.1") and torch.count_nonzero(value):
            raise PingstoreError(
                f"exp044 requires zero same-population recurrence: {name}"
            )
        if name in ("W_ei.1", "W_ie.1") and (value < 0).any():
            raise PingstoreError(
                f"exp044 requires nonnegative recurrent weights: {name}"
            )

    _, pixels, _, labels = load_dataset(
        "mnist", split=True, evaluation_split="test", evaluation_only=True
    )
    samples = cfg["evaluation_samples"]
    if (
        len(labels) != common["dataset_split"]["official_test_samples"]
        or pixels.shape != (len(labels), common["n_in"])
        or not 0 < samples <= len(labels)
    ):
        raise PingstoreError("invalid official-test partition or sample cap")
    selected = np.random.RandomState(cfg["evaluation_subset_seed"]).choice(
        len(labels), samples, replace=False
    )
    device = torch.device(
        os.environ.get("PINGLAB_DEVICE")
        or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )
    dt = cell["dt_ms"]
    steps = recipe.duration_steps(common["t_ms"], dt)
    duration_s = steps * dt / 1000.0
    modes = ["infer"]
    if cell["seed"] == cfg["raster"]["seed"]:
        modes.append("snapshot")
    trials = []
    for mode in modes:
        traces = mode == "snapshot"
        bundle = recipe.author_network(cfg, cell, common, traces=traces)
        model = GraphExecutor(
            plan_graph(bundle.graph),
            seed=cell["seed"],
            surrogate_slope=common["surrogate_slope"],
        )
        parameters = model.parameter_map()
        if set(parameters) != set(recipe.CHECKPOINT_PARAMETERS):
            raise PingstoreError(
                "exp044 graph parameter roles disagree with the recipe"
            )
        with torch.no_grad():
            for graph_name, checkpoint_name in recipe.CHECKPOINT_PARAMETERS.items():
                # The trained forward pass clamps feedforward weights at zero.
                # Recurrent matrices are copied unchanged; all matrices already
                # have runtime [source, target] orientation and stored scaling.
                value = state[checkpoint_name]
                if checkpoint_name.startswith("W_ff"):
                    value = value.clamp(min=0)
                if parameters[graph_name].shape != value.shape:
                    raise PingstoreError(
                        f"graph/checkpoint shape mismatch: {graph_name}"
                    )
                parameters[graph_name].copy_(value)
        model.to(device).eval()
        destination = export / mode / cell["cell_name"]
        destination.mkdir(parents=True, exist_ok=False)
        generator = torch.Generator().manual_seed(
            cell["seed"] if traces else cfg["evaluation_encoder_seed"]
        )
        started = perf_counter()
        with torch.inference_mode():
            if traces:
                index = cfg["raster"]["sample_index"]
                spikes = encode_images_poisson(
                    torch.from_numpy(pixels[index : index + 1]),
                    steps,
                    dt,
                    common["input_rate"],
                    generator=generator,
                )
                result = model({"drive": spikes.to(device)}, diagnostics=True)
                data = {
                    name: result.diagnostics[name].detach().cpu().numpy()[:, 0]
                    for name in ("spk_e", "spk_i")
                }
                if any(not np.isin(values, (0, 1)).all() for values in data.values()):
                    raise PingstoreError("invalid graph raster spikes")
                np.savez_compressed(
                    destination / "spikes.npz",
                    dt=dt,
                    **{name: values.astype(bool) for name, values in data.items()},
                )
                evidence.snapshot(destination / "spikes.npz", dt, common)
            else:
                correct, ce_sum, counts = 0, 0.0, {"e": 0.0, "i": 0.0}
                for start in range(0, samples, cfg["evaluation_batch_size"]):
                    indices = selected[start : start + cfg["evaluation_batch_size"]]
                    spikes = encode_images_poisson(
                        torch.from_numpy(pixels[indices]),
                        steps,
                        dt,
                        common["input_rate"],
                        generator=generator,
                    )
                    # Each call starts with reset voltage, conductances, delays
                    # and readout state. No runtime state is carried between images.
                    result = model({"drive": spikes.to(device)}, diagnostics=False)
                    scores = result.outputs["class_scores"]
                    if (
                        scores.shape != (len(indices), common["n_out"])
                        or not torch.isfinite(scores).all()
                    ):
                        raise PingstoreError("invalid graph class scores")
                    targets = torch.from_numpy(labels[indices])
                    ce_sum += float(
                        F.cross_entropy(scores, targets.to(device), reduction="sum")
                    )
                    correct += int((scores.argmax(dim=1).cpu() == targets).sum())
                    for label, size in (("e", n_e), ("i", n_i)):
                        values = result.outputs[f"spk_{label}_count"]
                        if (
                            values.shape != (len(indices), size)
                            or not torch.isfinite(values).all()
                            or (values < 0).any()
                            or (values > steps).any()
                            or not torch.equal(values, values.floor())
                        ):
                            raise PingstoreError("invalid graph spike-count reduction")
                        counts[label] += float(values.sum())
                write_json_atomic(
                    destination / "metrics.json",
                    {
                        "config": {
                            "dt": dt,
                            "t_ms": common["t_ms"],
                            "dataset": "mnist",
                            "n_hidden": n_e,
                            "n_inh": n_i,
                            "evaluation_partition": "official_mnist_test",
                            "evaluation_samples": samples,
                            **cell["execution_dynamics"],
                        },
                        "accuracy_pct": 100 * correct / samples,
                        "cross_entropy": ce_sum / samples,
                        "n_correct": correct,
                        "n_total": samples,
                        "rates_hz": {
                            "e": counts["e"] / (samples * n_e * duration_s),
                            "i": counts["i"] / (samples * n_i * duration_s),
                        },
                    },
                )
                evidence.measurement(
                    destination / "metrics.json", cell, common, samples
                )
        trials.append(
            {
                "cell": cell["cell_name"],
                "mode": mode,
                "graph_digest": bundle.manifest["graph_digest"],
                "device": str(device),
                "duration_seconds": perf_counter() - started,
            }
        )
        print(f"[{mode}] {cell['cell_name']}", flush=True)
    return trials


def compute(identity: str, *, run_id: str | None = None) -> str:
    bank = inputs.source(REPO, identity, "compute", experiment="exp022")
    cfg = recipe.configuration(smoke=os.environ.get("PINGLAB_SMOKE") == "1")
    contract = evidence.training_contract(bank.export, cfg)
    checkpoint_rows = evidence.checkpoints(bank.export, contract)
    evidence.histories(bank.export, contract)
    with inputs.execution(
        REPO, "compute", sources={"bank": bank}, run_id=run_id, configuration=cfg
    ) as run:
        run.record["execution"].update(
            environment={
                "PINGLAB_SMOKE": "1" if cfg["profile"] == "smoke" else "0",
                **(
                    {"PINGLAB_DEVICE": os.environ["PINGLAB_DEVICE"]}
                    if os.environ.get("PINGLAB_DEVICE")
                    else {}
                ),
            },
            executor=cfg["executor"],
            snnlab_version=version("snnlab"),
            parameter_binding=dict(recipe.CHECKPOINT_PARAMETERS),
            trials=[],
        )
        write_json_atomic(
            run.export / "evidence.json",
            {
                "schema": "exp044.compute/v2",
                "config": cfg,
                "training_contract": contract,
                "checkpoint_provenance": checkpoint_rows,
            },
        )
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for cell, checkpoint in zip(
                contract["cells"], checkpoint_rows, strict=True
            ):
                run.record["execution"]["trials"].extend(
                    evaluate_cell(
                        bank.unit(cell["cell_name"]),
                        checkpoint,
                        cell,
                        contract["common"],
                        cfg,
                        run.export,
                    )
                )
        finally:
            torch.set_num_threads(previous_threads)
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="completed v4 exp022 compute bank ID"
    )
    parser.add_argument("--run-id", help="unused v4 reservation")
    args = parser.parse_args()
    try:
        compute(args.source, run_id=args.run_id)
    except PingstoreError as exc:
        parser.exit(1, f"exp044 compute: {exc}\n")


if __name__ == "__main__":
    main()
