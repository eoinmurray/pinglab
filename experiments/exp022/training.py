"""Recipe-owned MNIST optimizer loop on the native graph executor."""

import json
import os
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from experiments.helpers.checkpoint_graph import PARAMETERS, initial_draws
from pingstore.contracts import file_sha256, write_json_atomic
from snnlab.analysis import (
    active_fraction,
    firing_rate,
    population_spike_count_cv,
    rhythmicity_metrics,
)
from snnlab.sim.datasets import load_dataset
from snnlab.sim.encoders import encode_images_poisson
from snnlab.sim.execution import GraphExecutor, plan_graph, resolve_device
from snnlab.sim.timing import duration_steps
from torch.utils.data import DataLoader, TensorDataset

from . import recipe


def weight_statistics(value):
    array = value.detach().cpu().numpy()
    return {
        "shape": list(array.shape),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "zero_count": int((array == 0).sum()),
        "zero_fraction": float((array == 0).mean()),
    }


def export_tensors(model, cfg):
    parameters = model.parameter_map()
    return {
        **{
            key: parameters[name].detach().cpu().clone()
            for name, key in PARAMETERS.items()
        },
        "W_ee.1": torch.zeros(cfg["n_hidden"], cfg["n_hidden"]),
        "W_ii.1": torch.zeros(cfg["n_inh"], cfg["n_inh"]),
    }


def reference_measurements(model, spikes, cfg):
    with torch.no_grad():
        result = model({"drive": spikes}, diagnostics=True)
    e = result.diagnostics["spk_e"].cpu().numpy()[:, 0]
    i = result.diagnostics["spk_i"].cpu().numpy()[:, 0]
    rhythm = rhythmicity_metrics(e, cfg["dt"])
    return {
        "rate_e": firing_rate(e, cfg["dt"]),
        "rate_i": firing_rate(i, cfg["dt"]),
        "cv": population_spike_count_cv(e, cfg["dt"]),
        "act": active_fraction(e),
        "contrast": rhythm["contrast"] if rhythm["contrast"] is not None else 0.0,
        "f0_hz": None,
        "lobe_lag_ms": rhythm["lobe_lag"],
        "trough_lag_ms": rhythm["trough_lag"],
        "iei_mode_lag_ms": rhythm["iei_mode_lag"],
        "lobe_to_trough": rhythm["lobe_to_trough"],
    }


def train_cell(cell, directory, samples, epochs):
    """Reset every presentation; select minimum draw-mean CE, then accuracy.

    Training retains the original device encoder stream and shuffled DataLoader;
    validation restarts its three CPU encoder/rate streams every epoch.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    cfg = recipe.training_settings(cell, samples, epochs)
    cfg["execution_digest"] = execution_digest(execution_identity())
    device = torch.device(resolve_device(os.environ.get("PINGLAB_DEVICE", "auto")))
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    x_train, x_validation, y_train, y_validation = load_dataset(
        "mnist", max_samples=samples, split=True
    )
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=cfg["batch_size"],
        shuffle=True,
    )
    validation_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_validation), torch.from_numpy(y_validation)),
        batch_size=cfg["batch_size"],
    )
    bundle = recipe.author_network(cfg, observables=("spikes",))
    trainable = ["input_to_E.weight", "readout_projection.weight"]
    if cfg["trainable_w_ei"]:
        trainable.append("E_to_I.weight")
    if cfg["trainable_w_ie"]:
        trainable.append("I_to_E.weight")
    model = GraphExecutor(
        plan_graph(bundle.graph),
        seed=cfg["seed"],
        trainable_parameters=trainable,
        surrogate_slope=cfg["surrogate_slope"],
    )
    _, initial = initial_draws(cfg)
    with torch.no_grad():
        for name, value in initial.items():
            model.parameter_map()[name].copy_(value)
    model.to(device)
    zero_ee = torch.zeros(cfg["n_hidden"], cfg["n_hidden"])
    zero_ii = torch.zeros(cfg["n_inh"], cfg["n_inh"])
    roles = {
        "W_in": initial["input_to_E.weight"],
        "W_out": initial["readout_projection.weight"],
        "W_EI_1": initial["E_to_I.weight"],
        "W_IE_1": initial["I_to_E.weight"],
        "W_EE_1": zero_ee,
        "W_II_1": zero_ii,
    }
    cfg["weight_initialization"] = {
        role: {
            "distribution": "lower_clamped_normal",
            "zeros_remain_trainable": True,
            "requested_initial_zero_fraction": cfg["w_in_initial_zero_fraction"]
            if role == "W_in"
            else 0.0,
            "statistics": weight_statistics(value),
        }
        for role, value in roles.items()
    }
    cfg["dataset_split"] = {
        "source_train_partition": "official_mnist_train",
        "source_test_partition": "official_mnist_test",
        "validation_fraction": 0.1,
        "split_seed": 42,
        "optimizer_train_samples": len(y_train),
        "validation_samples": len(y_validation),
        "official_test_samples": 10000,
        "checkpoint_selection_partition": "validation",
        "official_test_used_during_training": False,
    }
    cfg["validation_encoder_draws"] = {
        "count": 3,
        "encoder_seeds": list(recipe.VALIDATION_ENCODER_SEEDS),
        "input_rate_seeds": list(recipe.VALIDATION_RATE_SEEDS),
        "aggregation_unit": "validation_sample_then_encoder_draw",
        "checkpoint_selection": "minimum_mean_cross_entropy; tie maximum_mean_accuracy; tie earliest_epoch",
    }
    cfg.update(
        training_run_id=cell["training_run_id"],
        training_cell_name=cell["name"],
        mode="train",
        executor="snnlab.sim.GraphExecutor",
        graph_digest=bundle.manifest["graph_digest"],
        device=str(device),
    )
    write_json_atomic(directory / "config.json", cfg)
    steps = duration_steps(cfg["t_ms"], cfg["dt"])
    seconds = steps * cfg["dt"] / 1000
    _, test_pixels, _, test_labels = load_dataset(
        "mnist", split=True, evaluation_only=True, evaluation_split="test"
    )
    index = int(np.flatnonzero(test_labels == 0)[0])
    torch.manual_seed(recipe.REFERENCE_ENCODER_SEED)
    reference = encode_images_poisson(
        torch.from_numpy(test_pixels[index : index + 1]).to(device),
        steps,
        cfg["dt"],
        cfg["input_rate"],
    )
    initial_metrics = reference_measurements(model, reference, cfg)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    rates = (
        torch.tensor(cfg["input_rates"], dtype=torch.float32)
        if cfg["input_rates"]
        else None
    )
    rate_generator = torch.Generator().manual_seed(
        cfg["seed"] + recipe.TRAINING_RATE_SEED_OFFSET
    )

    def sample_rates(batch_size, generator):
        return (
            cfg["input_rate"]
            if rates is None
            else rates[
                torch.randint(len(rates), (batch_size,), generator=generator)
            ].to(device)
        )

    history = []
    best_loss = float("inf")
    best_accuracy = 0.0
    best_epoch = 0
    best_state = None
    started = perf_counter()
    for epoch in range(1, epochs + 1):
        trained = train_epoch(
            model,
            train_loader,
            optimizer,
            cfg,
            device,
            steps,
            seconds,
            sample_rates,
            rate_generator,
        )
        validated = validate_epoch(
            model, validation_loader, cfg, device, steps, seconds, sample_rates
        )
        validation_loss, accuracy = validated["test_loss"], validated["acc"]
        new_best = validation_loss < best_loss or (
            validation_loss == best_loss and accuracy > best_accuracy
        )
        if new_best:
            best_loss = validation_loss
            best_accuracy = accuracy
            best_epoch = epoch
            best_state = export_tensors(model, cfg)
        record = {
            "ep": epoch,
            "lr": cfg["lr"],
            "new_best": new_best,
            **trained,
            **validated,
            "weight_norms": {
                PARAMETERS[name]: float(value.detach().norm())
                for name, value in model.parameter_map().items()
                if value.requires_grad
            },
            **reference_measurements(model, reference, cfg),
        }
        history.append(record)
        with (directory / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        print(
            f"{cell['name']} epoch {epoch}/{epochs}: CE={validation_loss:g}, accuracy={accuracy:g}",
            flush=True,
        )
    if best_state is None:
        raise ValueError("training never produced a selectable checkpoint")
    final = export_tensors(model, cfg)
    torch.save(best_state, directory / "weights.pth")
    torch.save(final, directory / "weights_final.pth")
    final_roles = {
        role: final[key]
        for role, key in {
            "W_in": "W_ff.0",
            "W_out": "W_ff.1",
            "W_EE_1": "W_ee.1",
            "W_EI_1": "W_ei.1",
            "W_IE_1": "W_ie.1",
            "W_II_1": "W_ii.1",
        }.items()
    }
    write_json_atomic(
        directory / "metrics.json",
        {
            "config": cfg,
            "epochs": history,
            "init": initial_metrics,
            "end": reference_measurements(model, reference, cfg),
            "best_acc": best_accuracy,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "total_elapsed_s": perf_counter() - started,
            "weight_final": {
                role: weight_statistics(value) for role, value in final_roles.items()
            },
            "training_run_id": cell["training_run_id"],
            "training_cell_name": cell["name"],
            "checkpoints": {
                role: {
                    "filename": filename,
                    "epoch": selected_epoch,
                    "sha256": file_sha256(directory / filename),
                }
                for role, filename, selected_epoch in (
                    ("best_validation", "weights.pth", best_epoch),
                    ("final_epoch", "weights_final.pth", epochs),
                )
            },
        },
    )


def execution_identity():
    """Read-only source, backend and raw dataset identity for training retries."""
    import os
    from importlib.metadata import version

    import snnlab
    from experiments.helpers import checkpoint_graph, ping

    source = Path(snnlab.__file__).parent
    raw = Path("/tmp/mnist/MNIST/raw")
    names = (
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    )
    if any(not (raw / name).is_file() for name in names):
        raise ValueError(
            "prepopulate the explicit MNIST cache before freezing a training bank"
        )
    return {
        "dataset_files": {name: file_sha256(raw / name) for name in names},
        "versions": {
            name: version(name)
            for name in ("snnlab", "torch", "numpy", "torchvision", "scikit-learn")
        },
        "source": {
            str(p.relative_to(source)): file_sha256(p)
            for p in sorted(source.rglob("*"))
            if p.is_file() and p.suffix in (".py", ".json")
        },
        "helper_source": {
            module.__name__: file_sha256(Path(module.__file__))
            for module in (checkpoint_graph, ping, recipe)
        },
        "training_source": file_sha256(Path(__file__)),
        "runtime": {
            "requested_device": os.environ.get("PINGLAB_DEVICE", "auto"),
            "cuda": torch.version.cuda,
            "threads": torch.get_num_threads(),
            "default_dtype": str(torch.get_default_dtype()),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        },
    }


def execution_digest(identity):
    import hashlib

    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def train_epoch(
    model, loader, optimizer, cfg, device, steps, seconds, sample_rates, rate_generator
):
    model.train()
    loss_sum = 0.0
    batch_count = 0
    sample_count = 0
    grad_sum = 0.0
    grad_max = 0.0
    skipped = 0
    nan_batches = 0
    for pixels, labels in loader:
        pixels, labels = pixels.to(device), labels.to(device)
        drive = encode_images_poisson(
            pixels, steps, cfg["dt"], sample_rates(len(labels), rate_generator)
        )
        result = model({"drive": drive}, diagnostics=False)
        scores = result.outputs["class_scores"]
        if torch.isnan(scores).any():
            optimizer.zero_grad()
            nan_batches += 1
            continue
        loss = F.cross_entropy(scores, labels)
        if cfg["fr_reg_upper_strength"]:
            sample_rate = result.outputs["spk_e_count"].mean(1) / seconds
            loss = (
                loss
                + cfg["fr_reg_upper_strength"]
                * torch.relu(sample_rate - cfg["fr_reg_upper_target_hz"])
                .square()
                .mean()
            )
        optimizer.zero_grad()
        loss.backward()
        norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        )
        if np.isfinite(norm):
            optimizer.step()
            model.enforce_constraints()
            loss_sum += float(loss.detach())
            batch_count += 1
            grad_sum += norm
            grad_max = max(grad_max, norm)
        else:
            optimizer.zero_grad(set_to_none=True)
            skipped += 1
        sample_count += len(labels)
    return {
        "loss": loss_sum / max(batch_count, 1),
        "samples": sample_count,
        "grad_norm": grad_sum / max(batch_count, 1),
        "grad_norm_max": grad_max,
        "skipped_steps": skipped,
        "nan_forward_batches": nan_batches,
    }


def validate_epoch(model, loader, cfg, device, steps, seconds, sample_rates):
    model.eval()
    correct = 0
    total = 0
    ce_sum = 0.0
    e_sum = 0.0
    i_sum = 0.0
    margin = 0.0
    confidence = 0.0
    scale = 0.0
    draws = []
    output_spikes = 0.0
    output_silent = 0
    output_samples = 0
    class_spikes = torch.zeros(cfg["n_out"], dtype=torch.float64)
    by_rate = {}
    with torch.no_grad():
        for draw_index, (encoder_seed, rate_seed) in enumerate(
            zip(
                recipe.VALIDATION_ENCODER_SEEDS,
                recipe.VALIDATION_RATE_SEEDS,
                strict=True,
            )
        ):
            encoder = torch.Generator().manual_seed(encoder_seed)
            rate_rng = torch.Generator().manual_seed(rate_seed)
            draw_correct = 0
            draw_total = 0
            draw_loss = 0.0
            for pixels, labels in loader:
                pixels, labels = pixels.to(device), labels.to(device)
                sampled_rates = sample_rates(len(labels), rate_rng)
                drive = encode_images_poisson(
                    pixels, steps, cfg["dt"], sampled_rates, generator=encoder
                )
                result = model({"drive": drive}, diagnostics=False)
                scores = result.outputs["class_scores"]
                batch = len(labels)
                loss = float(F.cross_entropy(scores, labels)) * batch
                n_correct = int((scores.argmax(1) == labels).sum())
                ce_sum += loss
                draw_loss += loss
                correct += n_correct
                draw_correct += n_correct
                total += batch
                draw_total += batch
                e_sum += (
                    float(result.outputs["spk_e_count"].sum())
                    / cfg["n_hidden"]
                    / seconds
                )
                i_sum += (
                    float(result.outputs["spk_i_count"].sum()) / cfg["n_inh"] / seconds
                )
                if cfg["readout_mode"] == "spike-count":
                    out = scores.detach().cpu()
                    per_sample = out.sum(1)
                    output_spikes += float(per_sample.sum())
                    output_silent += int((per_sample == 0).sum())
                    output_samples += batch
                    class_spikes += out.sum(0).double()
                    realized = torch.as_tensor(sampled_rates).detach().cpu()
                    if realized.ndim == 0:
                        realized = realized.expand(batch)
                    for rate in torch.unique(realized).tolist():
                        mask = realized == rate
                        bucket = by_rate.setdefault(
                            float(rate),
                            {
                                "n_samples": 0,
                                "n_correct": 0,
                                "n_silent": 0,
                                "n_spikes": 0.0,
                            },
                        )
                        bucket["n_samples"] += int(mask.sum())
                        bucket["n_correct"] += int(
                            (out.argmax(1)[mask] == labels.detach().cpu()[mask]).sum()
                        )
                        bucket["n_silent"] += int((per_sample[mask] == 0).sum())
                        bucket["n_spikes"] += float(per_sample[mask].sum())
                z_true = scores.gather(1, labels[:, None]).squeeze(1)
                other = scores.clone()
                other.scatter_(1, labels[:, None], float("-inf"))
                margin += float((z_true - other.max(1).values).sum())
                confidence += float(scores.softmax(1).gather(1, labels[:, None]).sum())
                scale += float(scores.abs().mean(1).sum())
            draws.append(
                {
                    "draw": draw_index,
                    "encoder_seed": encoder_seed,
                    "input_rate_seed": rate_seed,
                    "n_correct": draw_correct,
                    "n_total": draw_total,
                    "acc": 100 * draw_correct / draw_total,
                    "test_loss": draw_loss / draw_total,
                }
            )
    validation_loss = ce_sum / total
    accuracy = 100 * correct / total
    return {
        "acc": accuracy,
        "test_loss": validation_loss,
        "test_rate_e": e_sum / total,
        "test_rate_i": i_sum / total,
        "test_margin": margin / total,
        "test_confidence": confidence / total,
        "test_logit_scale": scale / total,
        "validation_draws": draws,
        "test_output_spikes_per_sample": output_spikes / output_samples
        if output_samples
        else None,
        "test_output_silent_fraction": output_silent / output_samples
        if output_samples
        else None,
        "test_output_class_spike_fraction": (class_spikes / class_spikes.sum()).tolist()
        if class_spikes.sum()
        else [0.0] * cfg["n_out"]
        if output_samples
        else None,
        "test_output_by_input_rate": [
            {
                "rate_hz": rate,
                "n_samples": bucket["n_samples"],
                "accuracy_pct": 100 * bucket["n_correct"] / bucket["n_samples"],
                "spikes_per_sample": bucket["n_spikes"] / bucket["n_samples"],
                "silent_fraction": bucket["n_silent"] / bucket["n_samples"],
            }
            for rate, bucket in sorted(by_rate.items())
        ],
    }
