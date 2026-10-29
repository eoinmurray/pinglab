"""Train and record the introductory classifier using SNNLab lang and sim.

Invoke this stage independently through its CLI; it records configuration,
provenance and scientific outputs in Pingstore. Importing the module does not
launch computation or another stage.
"""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]

import numpy as np
import snnlab
import torch
from experiments.exp127.custom.network_artifacts import model_execution
from experiments.exp127.dataset import dataset
from experiments.exp127.recipe import SLUG, author_network, configuration
from pingstore.library import stage_run, write_json_atomic
from snnlab.sim import (
    CheckpointSelection,
    DenseArrayBinding,
    GraphExecutor,
    Performance,
    SnapshotSpec,
    ValidationSpec,
    plan_graph,
    simulate,
    train,
)


def compute(*, run_id=None):
    cfg = configuration()
    torch.set_num_threads(cfg["torch_threads"])
    bundle = author_network(cfg)
    sets = {
        name: dataset(cfg[f"{name}_samples"], cfg["seed"] + i + 1, cfg)
        for i, name in enumerate(("train", "validation", "test"))
    }
    bindings = {
        name: DenseArrayBinding("inputs", torch.from_numpy(values[0]))
        for name, values in sets.items()
    }
    with stage_run(REPO, SLUG, "compute", configuration=cfg, run_id=run_id) as run:
        path = bundle.write(run.export / "classifier.bundle", visualise=False)
        bundle.visualise(run.export / "circuit.svg", view="circuit")
        run.record["scientific_artifacts"] = {
            "snapshot_stages": {
                "teaching-train": ["end"],
                "train": ["end"],
                "before--simulation": ["start"],
                "after--simulation": ["start"],
            }
        }
        np.savez_compressed(
            run.export / "dataset.npz",
            **{
                f"{name}_{field}": value
                for name, values in sets.items()
                for field, value in zip(
                    ("spikes", "labels", "rates", "poisson_counts"), values, strict=True
                )
            },
        )
        np.savez_compressed(
            run.export / "training_trace.npz",
            **training_trace(
                bundle, path, sets["train"], run.export / "teaching-train", cfg
            ),
        )
        started = time.perf_counter()
        trained = train(
            bundle=path,
            input_bindings=(bindings["train"],),
            targets={"class": torch.from_numpy(sets["train"][1])},
            validation=ValidationSpec(
                input_bindings=(bindings["validation"],),
                targets={"class": torch.from_numpy(sets["validation"][1])},
            ),
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            shuffle=True,
            diagnostics=False,
            seed=cfg["seed"],
            device=cfg["device"],
            checkpoint_selection=CheckpointSelection(include_initial=True),
            save_final_checkpoint=run.export / "final.checkpoint",
            save_selected_checkpoint=run.export / "selected.checkpoint",
            performance=Performance(**cfg["performance"]),
            snapshot=SnapshotSpec(
                path=run.export / "train" / "end", stage="end", parameters=True
            ),
        )
        if trained.selected_checkpoint is None:
            raise ValueError("training did not select a checkpoint")
        training_seconds = time.perf_counter() - started
        write_json_atomic(
            run.export / "learning.json",
            {
                "epochs": trained.metrics["epochs"],
                "selection": trained.selected_checkpoint.selection_record,
            },
        )
        recordings = {}
        selected = trained.selected_checkpoint
        for mode, checkpoint in (
            ("before", None),
            ("after", run.export / "selected.checkpoint"),
        ):
            result = simulate(
                bundle=path,
                checkpoint=checkpoint,
                input_bindings=(bindings["test"],),
                seed=cfg["seed"],
                device=cfg["device"],
                performance=Performance(**cfg["performance"]),
                snapshot=SnapshotSpec(
                    path=run.export / f"{mode}--simulation" / "start",
                    stage="start",
                    parameters=True,
                ),
            )
            for name, tensor in (
                ("scores", result.outputs["scores"]),
                *result.diagnostics.items(),
            ):
                recordings[f"{mode}_{name}"] = tensor.detach().cpu().numpy()
            recordings[f"{mode}_weights"] = (
                next(iter(result.parameters.values())).detach().cpu().numpy()
            )
            print(
                mode,
                "test accuracy",
                float(
                    np.mean(recordings[f"{mode}_scores"].argmax(1) == sets["test"][1])
                ),
                flush=True,
            )
        np.savez_compressed(run.export / "recording.npz", **recordings)
        run.record["execution"].update(
            {
                "device": "cpu",
                "cpu_threads": 1,
                "training_seconds": training_seconds,
                "versions": {
                    "snnlab": snnlab.__version__,
                    "torch": torch.__version__,
                    "numpy": np.__version__,
                },
                "selected_epoch": selected.selection_record["epoch"],
            }
        )
        print(
            "selected epoch",
            selected.selection_record["epoch"],
            "training seconds",
            round(training_seconds, 2),
            flush=True,
        )
    print(run.run_id)
    return run.run_id


def training_trace(bundle, bundle_path, values, artifacts, cfg):
    """Record a separate fixed-batch teaching loop through the native SNNLab graph."""
    spikes, labels, _, _ = values
    model = GraphExecutor(
        plan_graph(bundle.graph),
        seed=cfg["seed"],
        trainable_parameters=bundle.training["resolved_parameters"]["trainable"],
        surrogate=bundle.training["surrogate"],
        compile_mode=None,
    )
    parameter = next(iter(model.parameter_map().values()))
    optimizer = torch.optim.AdamW([parameter], lr=cfg["learning_rate"], weight_decay=0)
    inputs, targets = torch.from_numpy(spikes), torch.from_numpy(labels)
    indices = [int(np.flatnonzero(labels == cls)[0]) for cls in (0, 1)]
    weights = [parameter.detach().numpy().copy()]
    losses, gradients, outputs, voltages = [], [], [], []
    with model_execution(
        model,
        bundle,
        artifacts,
        kind="train",
        seed=cfg["seed"],
        settings=cfg,
        snapshot_stages=("end",),
    ):
        for _ in range(cfg["trace_updates"]):
            optimizer.zero_grad(set_to_none=True)
            result = model({"inputs": inputs}, diagnostics=True)
            loss = torch.nn.functional.cross_entropy(result.outputs["scores"], targets)
            losses.append(float(loss.detach()))
            outputs.append(
                result.diagnostics["output_spikes"][:, indices].detach().numpy()
            )
            voltages.append(
                result.diagnostics["output_voltage"][:, indices].detach().numpy()
            )
            loss.backward()
            gradients.append(parameter.grad.detach().numpy().copy())
            torch.nn.utils.clip_grad_norm_([parameter], cfg["gradient_clip"])
            optimizer.step()
            model.enforce_constraints()
            weights.append(parameter.detach().numpy().copy())
    return dict(
        losses=np.asarray(losses),
        weights=np.asarray(weights),
        gradients=np.asarray(gradients),
        outputs=np.asarray(outputs),
        voltages=np.asarray(voltages),
        indices=np.asarray(indices),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="identity reserved before remote dispatch")
    compute(**vars(parser.parse_args()))
