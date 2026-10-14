"""Train and record the introductory classifier using SNNLab lang and sim."""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
import snnlab
import torch
from experiments.exp127.recipe import CONFIG, SLUG, network
from pingstore.contracts import write_json_atomic
from pingstore.stages import stage_run
from snnlab.sim.checkpoint_selection import CheckpointSelection
from snnlab.sim.execution import (
    DenseArrayBinding,
    ExecutionSpec,
    ValidationSpec,
    build,
    simulate,
    train,
)


def dataset(samples, seed):
    rng = np.random.default_rng(seed)
    labels = np.arange(samples, dtype=np.int64) % 2
    rng.shuffle(labels)
    rates = np.asarray(CONFIG["rate_patterns_hz"])[labels] * rng.uniform(
        *CONFIG["rate_jitter"], size=(samples, 3)
    )
    steps = round(CONFIG["duration_ms"] / CONFIG["dt_ms"])
    counts = rng.poisson(rates * CONFIG["dt_ms"] / 1000, size=(steps, samples, 3))
    spikes = (counts > 0).astype(np.float32)
    return spikes, labels, rates, counts


def training_trace(bundle_path, values):
    """Record a separate fixed-batch teaching loop through the native SNNLab graph."""
    spikes, labels, _, _ = values
    model = build(
        ExecutionSpec(
            kind="train", bundle=bundle_path, seed=CONFIG["seed"], device="cpu"
        )
    ).model
    parameter = next(iter(model.parameter_map().values()))
    optimizer = torch.optim.AdamW(
        [parameter], lr=CONFIG["learning_rate"], weight_decay=0
    )
    inputs, targets = torch.from_numpy(spikes), torch.from_numpy(labels)
    indices = [int(np.flatnonzero(labels == cls)[0]) for cls in (0, 1)]
    weights = [parameter.detach().numpy().copy()]
    losses, gradients, outputs, voltages = [], [], [], []
    for _ in range(CONFIG["trace_updates"]):
        optimizer.zero_grad(set_to_none=True)
        result = model({"inputs": inputs}, diagnostics=True)
        loss = torch.nn.functional.cross_entropy(result.outputs["scores"], targets)
        losses.append(float(loss.detach()))
        outputs.append(result.diagnostics["output_spikes"][:, indices].detach().numpy())
        voltages.append(
            result.diagnostics["output_voltage"][:, indices].detach().numpy()
        )
        loss.backward()
        gradients.append(parameter.grad.detach().numpy().copy())
        torch.nn.utils.clip_grad_norm_([parameter], CONFIG["gradient_clip"])
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


def compute():
    torch.set_num_threads(1)
    bundle = network()
    sets = {
        name: dataset(CONFIG[f"{name}_samples"], CONFIG["seed"] + i + 1)
        for i, name in enumerate(("train", "validation", "test"))
    }
    bindings = {
        name: DenseArrayBinding("inputs", torch.from_numpy(values[0]))
        for name, values in sets.items()
    }
    with stage_run(REPO, SLUG, "compute", configuration=CONFIG) as run:
        path = bundle.write(run.export / "classifier.bundle")
        initial = build(
            ExecutionSpec(kind="build", bundle=path, seed=CONFIG["seed"], device="cpu")
        )
        torch.save(initial.model.state_dict(), run.export / "before.pt")
        np.savez_compressed(
            run.export / "dataset.npz",
            **{
                f"{name}_{field}": value
                for name, values in sets.items()
                for field, value in zip(
                    ("spikes", "labels", "rates", "poisson_counts"), values
                )
            },
        )
        np.savez_compressed(
            run.export / "training_trace.npz", **training_trace(path, sets["train"])
        )
        started = time.perf_counter()
        trained = train(
            ExecutionSpec(
                kind="train",
                bundle=path,
                input_bindings=(bindings["train"],),
                targets={"class": torch.from_numpy(sets["train"][1])},
                validation=ValidationSpec(
                    input_bindings=(bindings["validation"],),
                    targets={"class": torch.from_numpy(sets["validation"][1])},
                ),
                epochs=CONFIG["epochs"],
                batch_size=CONFIG["batch_size"],
                shuffle=True,
                diagnostics=False,
                seed=CONFIG["seed"],
                device="cpu",
                checkpoint_selection=CheckpointSelection(include_initial=True),
                save_final_checkpoint=run.export / "final.checkpoint",
                save_selected_checkpoint=run.export / "selected.checkpoint",
            )
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
            ("before", run.export / "before.pt"),
            ("after", run.export / "selected.checkpoint"),
        ):
            result = simulate(
                ExecutionSpec(
                    kind="simulate",
                    bundle=path,
                    checkpoint=checkpoint,
                    input_bindings=(bindings["test"],),
                    seed=CONFIG["seed"],
                    device="cpu",
                )
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


if __name__ == "__main__":
    compute()
