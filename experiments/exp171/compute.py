"""Calibrate a frozen PING teacher and fine-tune COBA for the configured epoch count."""

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from shutil import copyfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "libraries")]
import numpy as np
import torch
import torch.nn.functional as F
from experiments.exp171 import recipe
from experiments.exp171.custom.inhibition import history_state
from experiments.exp171.custom.native import capture
from pingstore.library import file_sha256, source_run, stage_run
from scipy.signal import lfilter
from sklearn.model_selection import train_test_split
from snnlab import extensions, sim
from snnlab.sim import GraphExecutor, NetworkSnapshot, plan_graph, save_runtime_state
from torchvision.datasets import MNIST


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rng(cfg, domain, *keys):
    return np.random.default_rng(
        np.random.SeedSequence([cfg[domain + "_seed"], *map(int, keys)])
    )


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def source_indices(labels):
    pool = np.random.RandomState(42).choice(len(labels), 7000, replace=False)
    return train_test_split(pool, test_size=700, random_state=42, stratify=labels[pool])


def select_indices(labels, cfg):
    train, validation = source_indices(labels)
    selection = rng(cfg, "selection")
    selected = []
    for partition, count in [
        (train, cfg["train_per_digit"]),
        (validation, cfg["validation_per_digit"]),
    ]:
        selected.append(
            np.concatenate(
                [
                    selection.choice(
                        partition[labels[partition] == digit], count, replace=False
                    )
                    for digit in range(10)
                ]
            )
        )
    return *selected, train, validation


def sensory(pixels, original_index, cfg):
    return (
        rng(cfg, "sensory", original_index).random(
            (cfg["steps"], cfg["n_input"]), dtype=np.float32
        )
        < pixels[None, :] * cfg["input_rate_hz"] * cfg["dt_ms"] / 1000
    ).astype(np.float32)


def filtered(events, cfg):
    delayed = np.concatenate([np.zeros_like(events[:1]), events[:-1]], axis=0)
    return lfilter(
        [1.0], [1.0, -np.exp(-cfg["dt_ms"] / cfg["tau_gaba_ms"])], delayed, axis=0
    )


def calibrate(target, cfg, original_index):
    mu, sd = target.mean(0), target.std(0)
    if np.any(mu <= 0) or np.any(sd <= 0):
        raise ValueError("Nonpositive teacher conductance moments")
    cv = sd / mu
    gen = rng(cfg, "calibration", original_index)

    def fit(indices):
        history = cfg.get("inhibition_prehistory_steps", 0)
        u = gen.random((history + cfg["steps"], len(indices)))
        if cfg.get("scaled"):
            # Each GABA filter walks one channel over time.
            u = np.asfortranarray(u)
        low = np.full(len(indices), cfg["probability_bounds"][0])
        high = np.full(len(indices), cfg["probability_bounds"][1])
        error, p = np.full(len(indices), np.inf), low.copy()
        for _ in range(cfg["calibration_bisections"]):
            mid = (low + high) / 2
            h = filtered(u < mid, cfg)[history:]
            realised = h.std(0) / np.maximum(h.mean(0), 1e-20)
            e = np.abs(realised / cv[indices] - 1)
            better = e < error
            error[better], p[better] = e[better], mid[better]
            low, high = (
                np.where(realised > cv[indices], mid, low),
                np.where(realised <= cv[indices], mid, high),
            )
        h = filtered(u < p, cfg)[history:]
        q = mu[indices] / h.mean(0)
        return p, q, error, h * q

    p, q, error, g = fit(np.arange(cfg["n_e"]))
    for _ in range(cfg["calibration_retries"]):
        indices = np.flatnonzero(error > cfg["calibration_sd_target"])
        if not len(indices):
            break
        pp, qq, ee, gg = fit(indices)
        good = ee < error[indices]
        chosen = indices[good]
        p[chosen], q[chosen], error[chosen], g[:, chosen] = (
            pp[good],
            qq[good],
            ee[good],
            gg[:, good],
        )
    if not np.isfinite(q).all() or error.max() >= cfg["calibration_std_tolerance"]:
        raise ValueError(
            f"Calibration failed for image {original_index}: {error.max()}"
        )
    return (
        p.astype(np.float32),
        q.astype(np.float32),
        mu,
        sd,
        g.mean(0),
        g.std(0),
        float(error.max()),
    )


def artificial(probabilities, amplitudes, indices, cfg, *, epoch=None, draw=None):
    domain, key = ("training", epoch) if epoch is not None else ("validation", draw)
    if key is None:
        raise ValueError("Explicit epoch or draw required")
    return np.stack(
        [
            (
                rng(cfg, domain, key, index).random(
                    (
                        cfg.get("inhibition_prehistory_steps", 0) + cfg["steps"],
                        cfg["n_e"],
                    ),
                    dtype=np.float32,
                )
                < p[None, :]
            ).astype(np.float32)
            * q[None, :]
            for p, q, index in zip(probabilities, amplitudes, indices, strict=True)
        ],
        axis=1,
    )


def make_model(bundle, parameters, cfg, *, trainable=False):
    if "artificial_to_E.weight" in parameters:
        expected = (
            torch.ones(1, cfg["n_e"]) if cfg.get("scaled") else torch.eye(cfg["n_e"])
        )
        if not torch.equal(parameters["artificial_to_E.weight"].cpu(), expected):
            raise ValueError(
                "Private prehistory requires the frozen identity projection"
            )
    model = GraphExecutor(
        plan_graph(bundle.graph),
        seed=cfg["seed"],
        trainable_parameters=recipe.TRAINABLE if trainable else (),
        surrogate_slope=cfg["surrogate_slope"],
        compile_mode=None,
        precision=cfg["precision"],
    )
    if set(model.parameter_map()) != set(parameters):
        raise ValueError("Unexpected native parameter membership")
    with torch.no_grad():
        for name, value in model.parameter_map().items():
            value.copy_(parameters[name])
    return model.to(cfg["device"])


def forward(model, drive, inhibition=None):
    device = next(iter(model.parameter_map().values())).device
    inputs = {"drive": torch.from_numpy(drive).to(device)}
    state = None
    if inhibition is not None:
        history = len(inhibition) - len(drive)
        if history < 0 or inhibition.shape[1] != drive.shape[1]:
            raise ValueError("Inhibitory input does not cover the scored sensory trial")
        if history:
            state = history_state(model, inhibition[:history])
        inputs["artificial"] = torch.from_numpy(inhibition[history:]).to(device)
    return model(
        inputs,
        runtime_state=state,
        diagnostics=False,
        retain_parameters=False,
        retain_final_state=False,
        retain_runtime_state=False,
    ).outputs


def load_source(cfg):
    bank = source_run(
        ROOT / ".pingstore",
        cfg["source_run"],
        experiment="exp022",
        stage="compute",
        reference={"run_id": cfg["source_run"], "payload_digest": cfg["source_digest"]},
    )
    params, configurations, pins = {}, {}, {}
    for kind in ("coba", "ping"):
        cell = cfg[kind + "_cell"]
        conf = json.loads(bank.file(cell, "config.json").read_text())
        expected = {
            "dt": 0.1,
            "t_ms": 200.0,
            "max_samples": 7000,
            "input_rate": 25.0,
            "fr_reg_upper_strength": 0.0,
            "readout_mode": "mem-mean",
            "seed": 42,
            "n_in": 784,
            "n_hidden": 1024,
            "n_inh": 256,
            "n_out": 10,
            "tau_gaba_ms": 6.0,
            "tau_ampa_ms": 2.0,
            "v_grad_dampen": 1000.0,
        }
        if any(conf[k] != v for k, v in expected.items()):
            raise ValueError("Source settings do not match this pilot")
        pin = bank.record["bank_reuse"]["diagnostic_provenance"][cell]["checkpoint"]
        path = bank.file(cell, pin["filename"])
        if (
            pin["role"] != cfg["checkpoint_role"]
            or pin["epoch"] != 50
            or file_sha256(path) != pin["sha256"]
        ):
            raise ValueError("Invalid final-epoch checkpoint")
        weights = torch.load(path, map_location="cpu", weights_only=True)
        params[kind] = {
            name: weights[key].clamp(min=0).clone()
            for name, key in recipe.MAPPING.items()
        }
        configurations[kind], pins[kind] = conf, pin
    for name in ("E_to_I.weight", "I_to_E.weight"):
        if torch.count_nonzero(params["coba"][name]):
            raise ValueError("COBA feedback is not disabled")
    cfg["source_checkpoints"], cfg["source_configurations"] = pins, configurations
    return bank, params


def evaluate(model, drives, p, q, indices, labels, cfg):
    score_draws, rate_draws, hashes, moments = [], [], [], []
    model.eval()
    with torch.no_grad():
        for draw in range(cfg["validation_draws"]):
            ss, rr = [], []
            for start in range(0, len(indices), cfg["batch_size"]):
                take = np.arange(start, min(start + cfg["batch_size"], len(indices)))
                ev = artificial(p[take], q[take], indices[take], cfg, draw=draw)
                hashes.append(digest(ev))
                g = filtered(ev, cfg)[cfg.get("inhibition_prehistory_steps", 0) :]
                moments.append(np.stack([g.mean(0), g.std(0)], axis=1))
                out = forward(model, drives[:, take], ev)
                ss.append(out["scores"].detach().cpu().numpy())
                rr.append(
                    out["e_count"].detach().cpu().numpy().mean(1)
                    / (cfg["duration_ms"] / 1000)
                )
            score_draws.append(np.concatenate(ss))
            rate_draws.append(np.concatenate(rr))
    scores, rates = np.stack(score_draws), np.stack(rate_draws)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite validation score")
    losses = (
        F.cross_entropy(
            torch.from_numpy(scores.reshape(-1, cfg["n_output"])),
            torch.from_numpy(np.tile(labels, cfg["validation_draws"])),
            reduction="none",
        )
        .numpy()
        .reshape(scores.shape[:2])
    )
    return (
        dict(
            cross_entropy=float(losses.mean()),
            accuracy_pct=float(100 * (scores.argmax(-1) == labels).mean()),
            e_rate_hz=float(rates.mean()),
        ),
        scores,
        rates,
        losses,
        hashes,
        np.concatenate(moments),
    )


def train_one_epoch(model, optimizer, drives, p, q, indices, labels, cfg, epoch):
    nt = cfg["train_per_digit"] * 10
    training_records, training_hashes, training_moments = [], [], []
    norms = []
    tick = time.perf_counter()
    model.train()
    order = rng(cfg, "shuffle", epoch).permutation(nt)
    batch_losses = []
    for start in range(0, nt, cfg["batch_size"]):
        take = order[start : start + cfg["batch_size"]]
        ev = artificial(p[take], q[take], indices[take], cfg, epoch=epoch)
        training_hashes.append(digest(ev))
        g = filtered(ev, cfg)[cfg.get("inhibition_prehistory_steps", 0) :]
        training_moments.append(np.stack([g.mean(0), g.std(0)], axis=1))
        out = forward(model, drives[:, take], ev)
        loss = F.cross_entropy(
            out["scores"],
            torch.from_numpy(labels[indices[take]]).to(out["scores"].device),
        )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss; no skipped updates allowed")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        )
        if not np.isfinite(norm):
            raise ValueError("Nonfinite gradient; no skipped updates allowed")
        optimizer.step()
        model.enforce_constraints()
        norms.append(norm)
        batch_losses.append(float(loss.detach()))
        training_records.append(
            dict(
                epoch=epoch,
                original_indices=indices[take].tolist(),
                loss=batch_losses[-1],
                grad_norm=norm,
            )
        )
        print(
            f"epoch {epoch} update {len(batch_losses)}/10 loss={batch_losses[-1]:.6f} norm={norm:.6f}",
            flush=True,
        )
    training_loss = float(np.mean(batch_losses))
    return (
        training_loss,
        norms,
        time.perf_counter() - tick,
        training_records,
        training_hashes,
        training_moments,
    )


def verify_changes(model, initial):
    changes = {}
    for name, value in model.parameter_map().items():
        changes[name] = float((value.detach().cpu() - initial[name]).abs().max())
        if not torch.isfinite(value).all() or (
            name not in recipe.TRAINABLE and changes[name] != 0
        ):
            raise ValueError("Nonfinite or changed frozen parameter")
    if any(changes[name] == 0 for name in recipe.TRAINABLE):
        raise ValueError("A trainable weight matrix did not change")
    return changes


def cuda_preflight(model, teacher, drives, labels, cfg):
    """Check full-batch accelerator forward/backward without an optimizer step."""
    if cfg["device"] != "cuda":
        return None
    started = time.perf_counter()
    batch = cfg["batch_size"]
    with torch.no_grad():
        reference = forward(teacher, drives[:, : cfg["calibration_batch_size"]])
        if not torch.isfinite(reference["g_ping"]).all():
            raise ValueError("Nonfinite accelerator teacher output")
    result = forward(
        model,
        drives[:, :batch],
        np.zeros(
            (cfg["inhibition_prehistory_steps"] + cfg["steps"], batch, cfg["n_e"]),
            dtype=np.float32,
        ),
    )
    loss = F.cross_entropy(
        result["scores"], torch.from_numpy(labels[:batch]).to(cfg["device"])
    )
    loss.backward()
    for name in recipe.TRAINABLE:
        gradient = model.parameter_map()[name].grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise ValueError("Invalid accelerator backward pass")
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    result = dict(
        seconds=time.perf_counter() - started,
        loss=float(loss.detach()),
        device=torch.cuda.get_device_name(),
        peak_memory_bytes=torch.cuda.max_memory_allocated(),
        optimizer_steps=0,
    )
    print("CUDA_PREFLIGHT", json.dumps(result), flush=True)
    return result


def compute(run_id=None, *, epochs=None):
    started = time.perf_counter()
    cfg = recipe.configuration(
        device=os.environ.get("SNNLAB_DEVICE", "cpu"),
        **({} if epochs is None else {"epochs": epochs}),
    )
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    torch.set_num_threads(cfg["threads"])
    torch.manual_seed(cfg["seed"])
    bank, params = load_source(cfg)
    dataset = MNIST(
        root=os.environ.get("PINGLAB_DATASET_CACHE", "/tmp/mnist"),
        train=True,
        download=True,
    )
    labels = dataset.targets.numpy().astype(np.int64)
    pixels = dataset.data.numpy().reshape(-1, cfg["n_input"]).astype(np.float32) / 255
    train_idx, val_idx, source_train, source_val = select_indices(labels, cfg)
    indices = np.concatenate([train_idx, val_idx])
    cfg["dataset_sha256"] = {
        "pixels": digest(dataset.data.numpy()),
        "labels": digest(labels),
    }
    cfg["source_code_sha256"] = {
        str(p.relative_to(Path(__file__).parent)): file_sha256(p)
        for p in Path(__file__).parent.rglob("*.py")
    }
    with stage_run(
        ROOT,
        recipe.SLUG,
        "compute",
        inputs={"bank": bank},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        teacher_bundle, bundle = (
            recipe.author_network(cfg, teacher=True),
            recipe.author_network(cfg),
        )
        for name, authored in [("teacher", teacher_bundle), ("network", bundle)]:
            authored.write(run.export / (name + ".bundle"), visualise=False)
            authored.visualise(run.export / (name + "-circuit.svg"), view="circuit")
        teacher = make_model(teacher_bundle, params["ping"], cfg)
        capture(
            teacher,
            teacher_bundle,
            run.export / "teacher",
            cfg,
            kind="simulate",
            stage="start",
        )
        params["coba"]["artificial_to_E.weight"] = torch.eye(cfg["n_e"])
        model = make_model(bundle, params["coba"], cfg, trainable=True)
        capture(
            model,
            bundle,
            run.export / "initial",
            cfg,
            kind="simulate",
            stage="start",
            epoch=0,
        )
        # Sensory inputs are compact on disk; the full tensor stays in scratch RAM.
        drives = np.stack(
            [sensory(pixels[index], index, cfg) for index in indices], axis=1
        )
        np.savez_compressed(
            run.export / "dataset.npz",
            indices=indices,
            labels=labels[indices],
            source_train=source_train,
            source_validation=source_val,
            sensory_packed=np.packbits(drives.astype(bool), axis=0),
        )
        run.record["execution"]["cuda_preflight"] = cuda_preflight(
            model, teacher, drives, labels[indices], cfg
        )
        cal_start = time.perf_counter()
        calibration = []
        with torch.no_grad():
            for start in range(0, len(indices), cfg["calibration_batch_size"]):
                stop = min(start + cfg["calibration_batch_size"], len(indices))
                out = forward(teacher, drives[:, start:stop])
                conductance = out["g_ping"].detach().cpu().numpy()
                for position in range(start, stop):
                    g = conductance[:, position - start].astype(np.float64)
                    calibration.append(calibrate(g, cfg, indices[position]))
                if stop % 10 == 0:
                    print(
                        f"calibration {stop}/{len(indices)}; {time.perf_counter() - cal_start:.1f}s",
                        flush=True,
                    )
        p, q, mu, sd, fitted_mu, fitted_sd = [
            np.stack([row[k] for row in calibration]) for k in range(6)
        ]
        np.savez_compressed(
            run.export / "calibration.npz",
            probability=p,
            amplitude_us=q,
            target_mean_us=mu,
            target_sd_us=sd,
            fitted_mean_us=fitted_mu,
            fitted_sd_us=fitted_sd,
            max_sd_error=np.array([row[6] for row in calibration]),
        )
        timings = {
            "calibration_seconds": time.perf_counter() - cal_start,
            "training_seconds": 0.0,
            "evaluation_seconds": 0.0,
        }
        # Verify native weighted-event filtering and deterministic state resets once.
        replay_start = time.perf_counter()
        ev = artificial(p[:1], q[:1], indices[:1], cfg, draw=0)
        history_steps = cfg["inhibition_prehistory_steps"]
        if history_steps:
            save_runtime_state(
                run.export / "initial-inhibition-state",
                history_state(model, ev[:history_steps]),
            )
        with torch.no_grad():
            first = forward(model, drives[:, :1], ev)
            second = forward(model, drives[:, :1], ev)
            torch.testing.assert_close(
                first["scores"], second["scores"], rtol=0, atol=0
            )
        filter_error = float(
            np.max(
                np.abs(
                    first["g_artificial"].detach().cpu().numpy()
                    - filtered(ev, cfg)[history_steps:]
                )
            )
        )
        if filter_error > 2e-5:
            raise ValueError(f"Native GABA filtering mismatch: {filter_error}")
        timings["epoch_zero_replay_seconds"] = time.perf_counter() - replay_start
        nt = len(train_idx)
        history, all_scores, all_rates, all_losses = [], [], [], []
        expected_hashes, validation_moments = None, None
        best, best_state, best_epoch = (float("inf"), float("inf")), None, None
        optimizer = torch.optim.AdamW(
            [v for v in model.parameters() if v.requires_grad],
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )
        training_records, training_hashes, training_moments = [], [], []
        for epoch in range(cfg["epochs"] + 1):
            training_loss, norms = None, []
            if epoch:
                training_loss, norms, elapsed, records, hashes, moments = (
                    train_one_epoch(
                        model, optimizer, drives, p, q, indices, labels, cfg, epoch
                    )
                )
                timings["training_seconds"] += elapsed
                training_records.extend(records)
                training_hashes.extend(hashes)
                training_moments.extend(moments)
            tick = time.perf_counter()
            measured, scores, rates, losses, hashes, moments = evaluate(
                model, drives[:, nt:], p[nt:], q[nt:], val_idx, labels[val_idx], cfg
            )
            timings["evaluation_seconds"] += time.perf_counter() - tick
            if expected_hashes is not None and hashes != expected_hashes:
                raise ValueError("Validation inputs changed across epochs")
            expected_hashes, validation_moments = hashes, moments
            row = dict(
                epoch=epoch,
                training_loss=training_loss,
                grad_norm_mean=float(np.mean(norms)) if norms else None,
                grad_norm_max=max(norms) if norms else None,
                invalid_updates=0,
                **measured,
            )
            history.append(row)
            all_scores.append(scores)
            all_rates.append(rates)
            all_losses.append(losses)
            rank = (measured["cross_entropy"], -measured["accuracy_pct"])
            if rank < best:
                best, best_epoch = rank, epoch
                best_state = {
                    name: value.detach().clone()
                    for name, value in model.parameter_map().items()
                }
            write_json(
                run.scratch / "progress.json", dict(history=history, timings=timings)
            )
            print(json.dumps(row), flush=True)
        changes = verify_changes(model, params["coba"])
        capture(
            model,
            bundle,
            run.export / "final-training",
            cfg,
            kind="train",
            stage="end",
            epoch=cfg["epochs"],
        )
        with torch.no_grad():
            for name, value in model.parameter_map().items():
                value.copy_(best_state[name])
        capture(
            model,
            bundle,
            run.export / "best-validation",
            cfg,
            kind="train",
            stage="end",
            epoch=best_epoch,
        )
        np.savez_compressed(
            run.export / "validation.npz",
            scores=np.stack(all_scores),
            e_rates=np.stack(all_rates),
            losses=np.stack(all_losses),
            validation_conductance_moments=validation_moments,
            training_conductance_moments=np.concatenate(training_moments),
        )
        write_json(
            run.export / "metrics.json",
            dict(
                history=history,
                updates=training_records,
                best_epoch=best_epoch,
                parameter_changes=changes,
                native_filter_max_abs_error_us=filter_error,
                epoch_zero_replay_exact=True,
                validation_input_hashes=expected_hashes,
                training_input_hashes=training_hashes,
                inhibition_replay=dict(
                    original_index=int(indices[0]),
                    domain="validation",
                    draw=0,
                    prehistory_steps=history_steps,
                    scored_steps=cfg["steps"],
                ),
            ),
        )
        timings["compute_seconds"] = time.perf_counter() - started
        write_json(run.export / "timings.json", timings)
        run.record["scientific_artifacts"] = {
            "snapshot_stages": {
                "teacher": ["start"],
                "initial": ["start"],
                "final-training": ["end"],
                "best-validation": ["end"],
            }
        }
        run.record["execution"].update(
            elapsed_seconds=timings["compute_seconds"], timings=timings
        )
        bank.check_unchanged()
    print(run.run_id, flush=True)
    return run.run_id


def raster(source_id, *, run_id=None):
    """Replay the first validation batch at the retained lowest-CE checkpoint."""
    started = time.perf_counter()
    source = source_run(
        ROOT / ".pingstore", source_id, experiment=recipe.SLUG, stage="compute"
    )
    cfg = dict(source.record["execution"]["configuration"])
    cfg["training_code_sha256"] = cfg.pop("source_code_sha256")
    cfg["source_code_sha256"] = {
        str(p.relative_to(Path(__file__).parent)): file_sha256(p)
        for p in Path(__file__).parent.rglob("*.py")
    }
    cfg.update(
        device="cpu",
        threads=1,
        raster_validation_draw=0,
        raster_selection="First validation image, without activity or correctness selection",
        raster_retention="<25 MB; one trial recording, batch scores, native bundle/start snapshot and initial inhibitory state",
    )
    torch.set_num_threads(1)
    metrics = json.loads(source.file("metrics.json").read_text())
    epoch = metrics["best_epoch"]
    if epoch != min(
        range(len(metrics["history"])),
        key=lambda e: (
            metrics["history"][e]["cross_entropy"],
            -metrics["history"][e]["accuracy_pct"],
            e,
        ),
    ):
        raise ValueError("Retained checkpoint is not the best validation checkpoint")
    cfg["raster_checkpoint_epoch"] = epoch
    start = cfg["train_per_digit"] * 10
    stop = start + min(cfg["batch_size"], cfg["validation_per_digit"] * 10)
    with np.load(source.file("dataset.npz")) as z:
        indices, labels = (
            z["indices"][start:stop].copy(),
            z["labels"][start:stop].copy(),
        )
        drive = np.unpackbits(
            z["sensory_packed"][:, start:stop], axis=0, count=cfg["steps"]
        ).astype(np.float32)
    with np.load(source.file("calibration.npz")) as z:
        p, q = z["probability"][start:stop].copy(), z["amplitude_us"][start:stop].copy()
    events = artificial(p, q, indices, cfg, draw=cfg["raster_validation_draw"])
    weights = NetworkSnapshot.read(
        source.unit("best-validation", "end"), parameters=True
    ).load_parameters()
    bundle = recipe.author_network(cfg, record_spikes=True)
    model = make_model(bundle, weights, cfg)
    model.eval()
    with stage_run(
        ROOT,
        recipe.SLUG,
        "compute",
        phase="raster",
        inputs={"training": source},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        bundle.write(run.export / "network.bundle", visualise=False)
        bundle.visualise(run.export / "circuit.svg", view="circuit")
        capture(
            model,
            bundle,
            run.export / "execution",
            cfg,
            kind="simulate",
            stage="start",
            epoch=epoch,
        )
        history = cfg.get("inhibition_prehistory_steps", 0)
        if history:
            save_runtime_state(
                run.export / "initial-inhibition-state",
                history_state(model, events[:history]),
            )
        tick = time.perf_counter()
        with torch.no_grad():
            out = forward(model, drive, events)
        simulation_seconds = time.perf_counter() - tick
        scores = out["scores"].cpu().numpy()
        with np.load(source.file("validation.npz")) as z:
            expected = z["scores"][epoch, cfg["raster_validation_draw"], : stop - start]
            expected_rates = z["e_rates"][
                epoch, cfg["raster_validation_draw"], : stop - start
            ]
        np.testing.assert_allclose(scores, expected, rtol=1e-6, atol=1e-6)
        rates = out["e_count"].cpu().numpy().mean(1) / (cfg["duration_ms"] / 1000)
        np.testing.assert_allclose(rates, expected_rates, rtol=1e-6, atol=1e-5)
        for name, parameter in model.parameter_map().items():
            np.testing.assert_array_equal(
                parameter.detach().cpu().numpy(), np.asarray(weights[name])
            )
        e, i = out["E_spikes"][:, 0].cpu().numpy(), out["I_spikes"][:, 0].cpu().numpy()
        if i.any():
            raise ValueError("Disabled biological I population spiked")
        g = out["g_artificial"][:, 0].cpu().numpy()
        error = float(np.max(np.abs(g - filtered(events[:, :1], cfg)[history:, 0])))
        if error > 2e-5:
            raise ValueError("Raster inhibition does not match the prehistory filter")
        np.savez_compressed(
            run.export / "raster.npz",
            E_spikes=e.astype(bool),
            I_spikes=i.astype(bool),
            artificial_events=events[history:, 0] > 0,
            prehistory_events=events[:history, 0] > 0,
            g_artificial=g,
            amplitude_us=q[0],
            probability=p[0],
            batch_scores=scores,
            batch_indices=indices,
            batch_labels=labels,
            batch_rates_hz=rates,
            sensory_packed=np.packbits(drive[:, 0].astype(bool), axis=0),
        )
        summary = dict(
            epoch=epoch,
            checkpoint_selection="Lowest validation cross-entropy; accuracy then earliest epoch break ties",
            original_index=int(indices[0]),
            label=int(labels[0]),
            predicted=int(scores[0].argmax()),
            validation_draw=cfg["raster_validation_draw"],
            replayed_batch_size=len(indices),
            e_rate_hz=float(rates[0]),
            actual_I_spikes=int(i.sum()),
            replay_max_score_error=float(np.abs(scores - expected).max()),
            native_filter_max_abs_error_us=error,
            simulation_seconds=simulation_seconds,
            elapsed_seconds=time.perf_counter() - started,
        )
        write_json(run.export / "raster.json", summary)
        run.record["scientific_artifacts"] = {
            "snapshot_stages": {"execution": ["start"]}
        }
        run.record["execution"].update(
            no_simulation=False,
            simulation_seconds=simulation_seconds,
            elapsed_seconds=time.perf_counter() - started,
        )
        source.check_unchanged()
    print(run.run_id, flush=True)
    return run.run_id


def encode_native(arrays, selected, *, dt_ms, channels, config, seed, context):
    cfg = config["recipe"]
    indices = np.asarray(context.sample_ids)
    drive = np.stack(
        [
            sensory(arrays["features"][row] / np.float32(255), index, cfg)
            for row, index in zip(selected, indices, strict=True)
        ],
        axis=1,
    )
    coordinate = (
        {"epoch": context.epoch + 1}
        if context.phase == "train"
        else {"draw": context.draw_seed}
    )
    events = artificial(
        arrays["probability"][selected],
        arrays["amplitude"][selected],
        indices,
        cfg,
        **coordinate,
    )
    history = cfg["inhibition_prehistory_steps"]
    conductance = filtered(events[:history], cfg)[-1].astype(np.float32)
    return sim.PreparedBatch(
        {
            "drive": torch.from_numpy(drive),
            "artificial": torch.from_numpy(events[history:]),
        },
        sim.PresentationState(
            signature=config["signature"],
            conductances={"artificial_to_E": torch.from_numpy(conductance)},
            input_histories={
                "artificial": torch.from_numpy(events[history - 1 : history])
            },
            provenance={"policy": "private_gaba_only", "history_steps": history},
        ),
    )


def calibrate_native(teacher, pixels, indices, cfg):
    rows = []
    batch = cfg["calibration_batch_size"]
    with ThreadPoolExecutor(max_workers=cfg["threads"]) as pool, torch.no_grad():
        for start in range(0, len(indices), batch):
            ids = indices[start : start + batch]
            drives = np.stack(
                [sensory(pixels[index] / np.float32(255), index, cfg) for index in ids],
                axis=1,
            )
            g = forward(teacher, drives)["g_ping"].cpu().numpy()
            rows.extend(
                pool.map(
                    lambda pair: calibrate(pair[1].astype(np.float64), cfg, pair[0]),
                    zip(ids, np.moveaxis(g, 1, 0), strict=True),
                )
            )
            print(f"calibration {len(rows)}/{len(indices)}", flush=True)
    return [np.stack(column) for column in zip(*rows, strict=True)]


def native_parity(bundle, binding, params, cfg, performance):
    """Compare one native update with unequal microbatches against an eager update."""
    binding = replace(binding, sample_indices=(0, 1, 2, 3, 4))
    prepared, targets = sim.resolve_dataset_snapshot_binding(bundle.graph, binding)
    tensors = {key: value.to(cfg["device"]) for key, value in prepared.tensors.items()}
    state = prepared.tensors.initial_state.to(cfg["device"])
    recorded = recipe.author_network(cfg, record_spikes=True)
    dense_cfg = {**cfg, "scaled": False}
    dense = recipe.author_network(cfg, record_spikes=True, dense_private=True)
    dense_model = make_model(
        dense,
        {**params, "artificial_to_E.weight": torch.eye(cfg["n_e"])},
        dense_cfg,
    )
    compact_model = make_model(recorded, params, cfg)
    with torch.no_grad():
        expected = dense_model(
            tensors,
            initial_state=replace(
                state, signature=sim.runtime_state_signature(dense_model.plan)
            ),
        ).outputs
        actual = compact_model(
            tensors,
            initial_state=replace(
                state, signature=sim.runtime_state_signature(compact_model.plan)
            ),
        ).outputs
    (target,) = targets
    labels = target.value.to(cfg["device"])
    common = dict(
        bundle=bundle,
        input_bindings=[binding],
        epochs=1,
        batch_size=5,
        weights=sim.WeightInitialization(params),
        device=cfg["device"],
        seed=cfg["shuffle_seed"],
        metrics=sim.TrainingMetrics(initial=()),
        checkpoint_selection=sim.CheckpointSelection.disabled(),
        retain_final_state=False,
        retain_runtime_state=False,
        diagnostics=False,
    )
    reference = sim.train(
        **common,
        performance=sim.Performance(
            compile_mode=None,
            precision=cfg["precision"],
            fused_optimizer="off",
        ),
    )
    preflight = {
        **common,
        "metrics": sim.TrainingMetrics(train="evaluate"),
        "validation": sim.ValidationSpec(
            input_bindings=[
                replace(binding, split="validation", sample_indices=(5, 6))
            ],
            encoding_seeds=(0, 1, 2),
        ),
        "observations": sim.EpochObservations(population_rates=("E",)),
    }
    accumulated = sim.train(
        **preflight,
        batching=sim.Batching(microbatch_size=3),
        performance=performance,
    )
    errors = {}

    def compare(label, actual, expected):
        actual = torch.as_tensor(actual).detach().cpu()
        expected = torch.as_tensor(expected).detach().cpu()
        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=2e-5)
        errors[label] = float((actual - expected).abs().max())

    compare("conductance", actual["g_artificial"], expected["g_artificial"])
    compare("scores", actual["scores"], expected["scores"])
    compare(
        "loss",
        F.cross_entropy(actual["scores"], labels),
        F.cross_entropy(expected["scores"], labels),
    )
    for name in recipe.TRAINABLE:
        compare(
            f"gradient:{name}", accumulated.gradients[name], reference.gradients[name]
        )
        compare(
            f"update:{name}",
            accumulated.parameters[name] - params[name].to(cfg["device"]),
            reference.parameters[name] - params[name].to(cfg["device"]),
        )
        for key, value in reference.optimizer_state[name].items():
            compare(
                f"adamw:{name}:{key}", accumulated.optimizer_state[name][key], value
            )
    verify_changes(accumulated.model, params)
    return errors


def scaled_compute(
    mode,
    *,
    run_id=None,
    benchmark_id=None,
    benchmark_digest=None,
    compile_mode=None,
    microbatch_size=16,
    calibration_id=None,
    calibration_digest=None,
):
    started = time.perf_counter()
    if os.environ.get("SNNLAB_DEVICE") != "cuda" or torch.cuda.device_count() != 1:
        raise RuntimeError("Scaled exp171 requires exactly one CUDA GPU")
    cfg = recipe.configuration(
        scaled=True,
        device="cuda",
        threads=int(os.environ.get("OMP_NUM_THREADS", "4")),
        compile_mode=compile_mode,
        microbatch_size=microbatch_size,
    )
    torch.set_num_threads(cfg["threads"])
    bank, parameters = load_source(cfg)
    compatibility = dict(
        bank=bank.reference,
        configuration=cfg,
        snnlab=version("snnlab"),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        files={
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in sorted(Path(__file__).parent.rglob("*.py"))
            if path.name not in {"analyse.py", "present.py", "test.py"}
        },
        lock=file_sha256(ROOT / "uv.lock"),
    )
    performance = sim.Performance(
        compile_mode=cfg["compile_mode"],
        compile_block_steps="auto",
        precision=cfg["precision"],
        cuda_graphs="auto",
        fused_optimizer="auto",
    )
    compatibility["performance"] = asdict(performance)
    inputs = {"bank": bank}
    if mode == "full":
        benchmark = source_run(
            ROOT / ".pingstore",
            benchmark_id,
            experiment=recipe.SLUG,
            stage="compute",
            reference={"run_id": benchmark_id, "payload_digest": benchmark_digest},
        )
        report = json.loads(benchmark.file("benchmark.json").read_text())
        if report["status"] != "passed" or report["compatibility"] != compatibility:
            raise ValueError(
                "Full mode requires a reviewed, compatible passing benchmark"
            )
        inputs["benchmark"] = benchmark

    cfg = {
        **cfg,
        "mode": mode,
        "compatibility": compatibility,
        "retention_estimate": "Up to 40 MB benchmark or 350 MB full; compact dataset and calibration arrays, native bundles, snapshots and two checkpoints.",
    }
    dataset = MNIST(
        root=os.environ.get("PINGLAB_DATASET_CACHE", "/tmp/mnist"),
        train=True,
        download=True,
    )
    labels = dataset.targets.numpy().astype(np.int64)
    pixels = dataset.data.numpy().reshape(-1, cfg["n_input"])
    train, val = source_indices(labels)
    if mode == "benchmark":
        train = train[: cfg["benchmark_train_size"]]
        val = val[: cfg["benchmark_validation_size"]]
    ids = np.concatenate([train, val])
    cfg["dataset_sha256"] = {"pixels": digest(pixels), "labels": digest(labels)}
    calibration_source = None
    if calibration_id:
        calibration_source = source_run(
            ROOT / ".pingstore",
            calibration_id,
            experiment=recipe.SLUG,
            stage="compute",
            reference={"run_id": calibration_id, "payload_digest": calibration_digest},
        )
        source_cfg = calibration_source.record["execution"]["configuration"]
        if (
            source_cfg.get("mode") != "benchmark"
            or calibration_source.record["inputs"].get("bank") != bank.reference
            or source_cfg.get("dataset_sha256") != cfg["dataset_sha256"]
            or any(
                source_cfg.get(key) != cfg[key]
                for key in (
                    "dt_ms",
                    "steps",
                    "n_input",
                    "n_e",
                    "tau_gaba_ms",
                    "tau_ampa_ms",
                    "inhibition_prehistory_steps",
                    "input_rate_hz",
                    "sensory_seed",
                    "calibration_seed",
                    "probability_bounds",
                    "calibration_bisections",
                    "calibration_retries",
                    "calibration_sd_target",
                    "calibration_std_tolerance",
                    "refractory_timing",
                    "precision",
                )
            )
        ):
            raise ValueError("Incompatible retained benchmark calibration")
        inputs["calibration"] = calibration_source

    print(
        f"{mode}: {len(train)} train, {len(val)} validation, {compatibility['gpu']}",
        flush=True,
    )
    with stage_run(
        ROOT,
        recipe.SLUG,
        "compute",
        inputs=inputs,
        configuration=cfg,
        run_id=run_id,
        operation=mode,
    ) as run:
        teacher_bundle = recipe.author_network(cfg, teacher=True)
        bundle = recipe.author_network(cfg)
        for name, authored in [("teacher", teacher_bundle), ("network", bundle)]:
            authored.write(run.export / f"{name}.bundle", visualise=False)
            authored.visualise(run.export / f"{name}-circuit.svg", view="circuit")
        teacher = make_model(teacher_bundle, parameters["ping"], cfg)
        capture(
            teacher,
            teacher_bundle,
            run.export / "teacher",
            cfg,
            kind="simulate",
            stage="start",
        )
        params = {
            **parameters["coba"],
            "artificial_to_E.weight": torch.ones(1, cfg["n_e"]),
        }
        encoder_cfg = {
            key: cfg[key]
            for key in (
                "steps",
                "n_input",
                "n_e",
                "dt_ms",
                "tau_gaba_ms",
                "input_rate_hz",
                "inhibition_prehistory_steps",
                "sensory_seed",
                "training_seed",
                "validation_seed",
            )
        }
        extensions.register_encoder("pinglab.exp171/v1", encode_native)
        encoder = sim.DatasetEncoder(
            "custom",
            definition="pinglab.exp171/v1",
            rng_policy="sample_v1",
            duration_ms=cfg["duration_ms"],
            config={
                "recipe": encoder_cfg,
                "signature": sim.runtime_state_signature(sim.plan_graph(bundle.graph)),
            },
        )

        def binding(path, split, positions):
            return sim.DatasetSnapshotBinding(
                path,
                "drive",
                "exp171-mnist",
                split,
                encoder,
                target_id="digit",
                sample_ids_key="ids",
                sample_indices=tuple(positions),
            )

        def write_dataset(path, ids, probability, amplitude):
            np.savez_compressed(
                path,
                ids=ids,
                features=pixels[ids],
                labels=labels[ids],
                probability=probability,
                amplitude=amplitude,
            )

        parity_start = None
        if calibration_source is not None:
            path = run.export / "dataset.npz"
            cached_dataset = calibration_source.file("dataset.npz")
            cached_calibration = calibration_source.file("calibration.npz")
            with (
                np.load(cached_dataset) as retained,
                np.load(cached_calibration) as moments,
            ):
                np.testing.assert_array_equal(retained["ids"], ids)
                np.testing.assert_array_equal(moments["ids"], ids)
                np.testing.assert_array_equal(retained["features"], pixels[ids])
                np.testing.assert_array_equal(retained["labels"], labels[ids])
            copyfile(cached_dataset, path)
            copyfile(cached_calibration, run.export / "calibration.npz")
            calibration_s = 0.0
            calibration_estimate_s = json.loads(
                calibration_source.file("benchmark.json").read_text()
            )["calibration_seconds"]
            parity_start = time.perf_counter()
            parity = native_parity(
                bundle, binding(path, "train", range(5)), params, cfg, performance
            )
            print(
                f"retained calibration verified; numerical parity passed: {parity}",
                flush=True,
            )
        else:
            parity = {}
            calibration_s = 0.0
            if mode == "benchmark":
                first_batch = cfg["calibration_batch_size"]
                tick = time.perf_counter()
                first = calibrate_native(teacher, pixels, ids[:first_batch], cfg)
                calibration_s += time.perf_counter() - tick
                parity_path = run.scratch / "parity.npz"
                write_dataset(parity_path, ids[:first_batch], *first[:2])
                parity = native_parity(
                    bundle,
                    binding(parity_path, "train", range(5)),
                    params,
                    cfg,
                    performance,
                )
                print(f"numerical parity passed: {parity}", flush=True)
            else:
                first_batch = 0
            tick = time.perf_counter()
            calibration = calibrate_native(teacher, pixels, ids[first_batch:], cfg)
            calibration_s += time.perf_counter() - tick
            if first_batch:
                calibration = [
                    np.concatenate((start, rest))
                    for start, rest in zip(first, calibration, strict=True)
                ]
            probability, amplitude, mu, sd, fitted_mu, fitted_sd, error = calibration
            del teacher
            print(f"calibration completed in {calibration_s:.2f}s", flush=True)
            np.savez_compressed(
                run.export / "calibration.npz",
                ids=ids,
                target_mean=mu,
                target_sd=sd,
                fitted_mean=fitted_mu,
                fitted_sd=fitted_sd,
                max_relative_sd_error=error,
            )
            path = run.export / "dataset.npz"
            write_dataset(path, ids, probability, amplitude)
            calibration_estimate_s = calibration_s
        parity_s = None if parity_start is None else time.perf_counter() - parity_start
        training = binding(path, "train", range(len(train)))
        print("starting measured native training", flush=True)
        epochs = cfg["benchmark_epochs"] if mode == "benchmark" else cfg["epochs"]
        result = sim.train(
            bundle=bundle,
            weights=sim.WeightInitialization(params),
            input_bindings=[training],
            validation=sim.ValidationSpec(
                input_bindings=[
                    binding(path, "validation", range(len(train), len(ids)))
                ],
                encoding_seeds=tuple(range(cfg["validation_draws"])),
            ),
            epochs=epochs,
            batch_size=cfg["batch_size"],
            batching=sim.Batching(
                order="shuffle", microbatch_size=cfg["microbatch_size"]
            ),
            seed=cfg["shuffle_seed"],
            device=cfg["device"],
            performance=performance,
            training_timing=True,
            diagnostics=False,
            metrics=sim.TrainingMetrics(train="evaluate"),
            observations=sim.EpochObservations(population_rates=("E",)),
            checkpoint_selection=sim.CheckpointSelection(
                tie_break=(sim.SelectionMetric("accuracy", direction="max"),),
                include_initial=True,
            ),
            save_final_checkpoint=run.export / "final.checkpoint",
            save_selected_checkpoint=run.export / "best.checkpoint",
            snapshot=sim.SnapshotSpec(run.export / "training", stage="both"),
            retain_final_state=False,
            retain_runtime_state=False,
            on_epoch_end=lambda row: print(
                f"epoch {row.epoch}, updates {row.updates}", flush=True
            ),
        )
        changes = verify_changes(result.model, params)
        best = result.selected_checkpoint
        best_model = make_model(bundle, best.parameters, cfg)
        capture(
            best_model,
            bundle,
            run.export / "best-validation",
            cfg,
            kind="train",
            stage="end",
            epoch=best.completed_updates
            // ((len(train) + cfg["batch_size"] - 1) // cfg["batch_size"]),
        )
        counts = result.metrics["batch_counts"]
        expected_updates = epochs * (
            (len(train) + cfg["batch_size"] - 1) // cfg["batch_size"]
        )
        if counts["optimizer_updates"] != expected_updates or counts[
            "examples"
        ] != epochs * len(train):
            raise ValueError(
                "Native trainer did not complete the expected presentations and updates"
            )
        metrics = {
            key: result.metrics[key]
            for key in (
                "epochs",
                "batch_counts",
                "training_timing",
                "performance",
                "selection_record",
            )
        }
        metrics["parameter_changes"] = changes
        write_json(run.export / "native-metrics.json", metrics)
        run.record["scientific_artifacts"] = {
            "snapshot_stages": {
                "teacher": ["start"],
                "training": ["start", "end"],
                "best-validation": ["end"],
            }
        }
        elapsed = time.perf_counter() - started
        run.record["execution"].update(
            elapsed_seconds=elapsed, calibration_seconds=calibration_s
        )
        if mode == "benchmark":
            timing = metrics["training_timing"]
            warmed = timing["epochs"][-1]
            estimate = (
                calibration_estimate_s * 7000 / len(ids)
                + timing["setup_s"]
                + (warmed["training_s"] + warmed["train_evaluation_s"])
                * 20
                * 6300
                / len(train)
                + warmed["validation_s"] * 21 * 700 / len(val)
            )
            write_json(
                run.export / "benchmark.json",
                dict(
                    status="passed",
                    compatibility=compatibility,
                    parity=parity,
                    parity_seconds=parity_s,
                    calibration_estimate_seconds=calibration_estimate_s,
                    calibration_seconds=calibration_s,
                    elapsed_seconds=elapsed,
                    full_estimate_seconds=estimate,
                    gpu_usd_per_second=cfg["benchmark_gpu_usd_per_second"],
                    full_estimate_gpu_usd=estimate
                    * cfg["benchmark_gpu_usd_per_second"],
                ),
            )
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--benchmark", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument(
        "--benchmark-id", help="Explicitly reviewed passing benchmark run"
    )
    parser.add_argument("--benchmark-digest", help="Reviewed benchmark payload digest")
    parser.add_argument(
        "--raster-source",
        help="Replay a retained training run's best-validation checkpoint",
    )
    parser.add_argument(
        "--epochs", type=int, help="Total fine-tuning epochs; default from recipe"
    )
    parser.add_argument("--compile-mode", choices=("eager", "default"), default="eager")
    parser.add_argument("--microbatch-size", choices=(16, 32, 64), type=int, default=16)
    parser.add_argument(
        "--calibration-id", help="Benchmark run with the identical calibrated subset"
    )
    parser.add_argument("--calibration-digest")
    args = parser.parse_args()
    if bool(args.calibration_id) != bool(args.calibration_digest) or (
        args.calibration_id and not args.benchmark
    ):
        parser.error("Calibration reuse requires --benchmark and both calibration pins")
    if not (args.benchmark or args.full) and (
        args.compile_mode != "eager" or args.microbatch_size != 16
    ):
        parser.error("Execution policy overrides require --benchmark or --full")
    if args.full != bool(args.benchmark_id and args.benchmark_digest) or (
        not args.full and (args.benchmark_id or args.benchmark_digest)
    ):
        parser.error("--full requires --benchmark-id and --benchmark-digest")
    if args.benchmark or args.full:
        if args.epochs is not None or args.raster_source:
            parser.error(
                "Scaled modes cannot be combined with pilot or raster arguments"
            )
        scaled_compute(
            "full" if args.full else "benchmark",
            run_id=args.run_id,
            benchmark_id=args.benchmark_id,
            benchmark_digest=args.benchmark_digest,
            compile_mode=None if args.compile_mode == "eager" else args.compile_mode,
            microbatch_size=args.microbatch_size,
            calibration_id=args.calibration_id,
            calibration_digest=args.calibration_digest,
        )
    elif args.raster_source:
        if args.epochs is not None:
            parser.error("--epochs cannot be combined with --raster-source")
        raster(args.raster_source, run_id=args.run_id)
    else:
        compute(args.run_id, epochs=args.epochs)
