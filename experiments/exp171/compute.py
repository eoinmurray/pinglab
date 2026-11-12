"""Calibrate private inhibition, then fine-tune through SNNLab's native trainer."""

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "libraries")]
import numpy as np
import torch
from experiments.exp171 import recipe
from pingstore.library import file_sha256, source_run, stage_run
from scipy.signal import lfilter
from sklearn.model_selection import train_test_split
from snnlab import extensions, lang, sim
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


def generator_key(cfg):
    """Archived runs retain their recorded Bernoulli generator semantics."""
    if cfg.get("inhibition_method") == recipe.POISSON_METHOD:
        return "poisson_rate_hz"
    return (
        "timing_interpolation"
        if cfg.get("inhibition_method") in recipe.COUNT_METHODS
        else "probability"
    )


def moment_keys(cfg):
    return (
        ("expected_mean", "expected_sd")
        if cfg.get("inhibition_method") == recipe.POISSON_METHOD
        else ("fitted_mean", "fitted_sd")
    )


def poisson_parameters(target, cfg):
    """Exp170's discrete Poisson moment fit, independent of any sampled draw."""
    if target.ndim != 2 or not np.isfinite(target).all() or np.any(target < 0):
        raise ValueError("Invalid PING conductance target")
    mu, variance = target.mean(0), target.var(0)
    active = mu > 0
    if np.any(active & (variance == 0)):
        raise ValueError(
            "Constant positive conductance cannot be represented by finite-rate Poisson input"
        )
    decay = np.exp(-cfg["dt_ms"] / cfg["tau_gaba_ms"])
    amplitude = np.zeros_like(mu, dtype=np.float32)
    rates = np.zeros_like(mu, dtype=np.float64)
    amplitude[active] = ((1 + decay) * variance[active] / mu[active]).astype(np.float32)
    rates[active] = (
        mu[active] ** 2
        * (1 - decay)
        / ((cfg["dt_ms"] / 1000) * (1 + decay) * variance[active])
    )
    validate_poisson_parameters(rates, amplitude)
    if np.any(amplitude[active] <= 0):
        raise ValueError("Unrepresentable Poisson calibration")
    return rates, amplitude


def validate_poisson_parameters(rates, amplitudes):
    if (
        rates.shape != amplitudes.shape
        or not np.isfinite(rates).all()
        or not np.isfinite(amplitudes).all()
        or np.any(rates < 0)
        or np.any(amplitudes < 0)
        or np.any((rates == 0) != (amplitudes == 0))
    ):
        raise ValueError("Invalid Poisson rates or amplitudes")


def poisson_expected_moments(rates, amplitude, cfg):
    decay = np.exp(-cfg["dt_ms"] / cfg["tau_gaba_ms"])
    increments = rates * cfg["dt_ms"] / 1000
    return (
        increments * amplitude / (1 - decay),
        np.sqrt(increments * amplitude.astype(np.float64) ** 2 / (1 - decay**2)),
    )


def teacher_event_counts(counts, cfg):
    """One half-up rounded population-mean I count per image, never a floor of one."""
    counts = np.asarray(counts)
    if (
        counts.ndim != 2
        or counts.shape[1] != cfg["n_i"]
        or not np.isfinite(counts).all()
        or np.any(counts < 0)
        or np.any(counts > cfg["steps"])
        or not np.equal(counts, np.rint(counts)).all()
    ):
        raise ValueError("Invalid teacher spike counts for event-count extraction")
    result = np.floor(counts.mean(axis=1, dtype=np.float64) + 0.5).astype(np.int32)
    if np.any(result >= cfg["steps"]):
        raise ValueError(
            "Teacher-derived count exceeds distinct deliverable event bins"
        )
    return result


def event_counts(cfg, size, values=None):
    """Validate compact CPU metadata before generating any device tensors."""
    if cfg.get("inhibition_method") == recipe.TEACHER_COUNT_METHOD:
        if values is None:
            raise ValueError(
                "Teacher-rate inhibition requires saved per-image event_count"
            )
        counts = np.asarray(values)
        if (
            counts.shape != (size,)
            or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < 0)
            or np.any(counts >= cfg["steps"])
        ):
            raise ValueError("Invalid per-image event counts")
        return counts
    return np.full(size, cfg["target_events_per_source"], dtype=np.int32)


def dataset_event_counts(data, cfg, selected):
    return (
        data["event_count"][selected]
        if cfg.get("inhibition_method") == recipe.TEACHER_COUNT_METHOD
        else None
    )


def relative_moment_error(actual, target):
    """Exact silence is valid; nonzero output against a zero target is not."""
    actual, target = np.asarray(actual), np.asarray(target)
    relative = np.full_like(actual, np.inf, dtype=np.float64)
    np.divide(actual, target, out=relative, where=target != 0)
    relative = np.abs(relative - 1)
    return np.where((target == 0) & (actual == 0), 0.0, relative)


def timing_endpoints(gen, steps, channels, count):
    regular = (
        (np.arange(count)[:, None] + gen.random(channels)[None, :])
        * (steps - 1)
        / count
    )
    random = np.column_stack(
        [np.sort(gen.choice(steps - 1, count, replace=False)) for _ in range(channels)]
    )
    return regular, random


def timing_events(times, steps, *, pending=False, half_up=False):
    last = steps - 1 if pending else steps - 2
    rounded = np.floor(times + 0.5 + 1e-9) if half_up else np.rint(times)
    bins = np.minimum(rounded.astype(int), last)
    if np.any(np.diff(bins, axis=0) <= 0) or bins.min() < 0 or bins.max() > last:
        raise ValueError("Timing interpolation lost distinct deliverable events")
    events = np.zeros((steps, bins.shape[1]), dtype=np.float32)
    events[bins, np.arange(bins.shape[1])[None, :]] = 1
    return events


def clustered_endpoint(gen, steps, channels, count):
    """A private random-phase burst of K distinct consecutive bins."""
    start = gen.integers(0, steps - count, size=channels)
    return start[None, :] + np.arange(count)[:, None]


def timing_positions(regular, random, alpha, clustered=None):
    upper = 1 if clustered is None else 2
    if not np.isfinite(alpha).all() or np.any(alpha < 0) or np.any(alpha > upper):
        raise ValueError("Invalid timing interpolation")
    first = np.minimum(alpha, 1)
    second = np.maximum(alpha - 1, 0)
    times = regular * (1 - first) + random * (first - second)
    return times if clustered is None else times + clustered * second


def timing_trial(regular, random, alpha, cfg, history_endpoints=None, clusters=None):
    """Exactly K scored events plus the tail of an independent preceding draw."""
    events = timing_events(
        timing_positions(
            regular, random, alpha, None if clusters is None else clusters[0]
        ),
        cfg["steps"],
        half_up=clusters is not None,
    )
    history = cfg.get("inhibition_prehistory_steps", 0)
    if history:
        hr, hc = history_endpoints
        past = timing_events(
            timing_positions(hr, hc, alpha, None if clusters is None else clusters[1]),
            cfg["steps"],
            pending=True,
            half_up=clusters is not None,
        )
        events = np.concatenate((past[-history:], events))
    return events


def calibrate_count_matched(target, cfg, original_index):
    """Exp170 timing fit, independently implemented for per-image generators."""
    steps, channels = target.shape
    mu, sd = target.mean(0), target.std(0)
    if np.any(mu <= 0) or np.any(sd <= 0) or not np.isfinite(target).all():
        raise ValueError("Invalid teacher conductance moments")
    cv = sd / mu
    gen = rng(cfg, "calibration", original_index)
    regular, random = timing_endpoints(
        gen, steps, channels, cfg["target_events_per_source"]
    )

    history = cfg.get("inhibition_prehistory_steps", 0)
    history_endpoints = (
        timing_endpoints(gen, steps, channels, cfg["target_events_per_source"])
        if history
        else None
    )

    extended = cfg.get("inhibition_method") in recipe.EXTENDED_METHODS
    clusters = (
        tuple(
            clustered_endpoint(gen, steps, channels, cfg["target_events_per_source"])
            for _ in range(2)
        )
        if extended
        else None
    )

    def moments(alpha, indices=None):
        selected = np.arange(channels) if indices is None else indices
        past = tuple(a[:, selected] for a in history_endpoints) if history else None
        events = timing_trial(
            regular[:, selected],
            random[:, selected],
            alpha,
            cfg,
            past,
            tuple(a[:, selected] for a in clusters) if extended else None,
        )
        h = filtered(events, cfg)[history:]
        return h, h.std(0) / h.mean(0)

    _, regular_cv = moments(np.zeros(channels))
    if np.any(regular_cv >= cv):
        raise ValueError(
            f"Regular timing endpoint exceeds teacher CV for image {original_index}"
        )
    bracketed = np.zeros(channels, dtype=bool)
    # Equivalent endpoint moments may differ at float64 roundoff after scaling.
    upper_target = (
        cv * (1 - 1e-12)
        if cfg.get("inhibition_method") == recipe.TEACHER_COUNT_METHOD
        else cv
    )
    attempts = 1 if extended else cfg["bracket_attempts"]
    for attempt in range(attempts):
        indices = np.flatnonzero(~bracketed)
        if not len(indices):
            break
        if attempt:
            random[:, indices] = np.column_stack(
                [
                    np.sort(
                        gen.choice(
                            steps - 1, cfg["target_events_per_source"], replace=False
                        )
                    )
                    for _ in indices
                ]
            )
            if history:
                history_endpoints[1][:, indices] = np.column_stack(
                    [
                        np.sort(
                            gen.choice(
                                steps - 1,
                                cfg["target_events_per_source"],
                                replace=False,
                            )
                        )
                        for _ in indices
                    ]
                )
        _, candidate_cv = moments(np.ones(len(indices)), indices)
        bracketed[indices] = candidate_cv >= upper_target[indices]
    upper = np.ones(channels)
    if extended:
        indices = np.flatnonzero(~bracketed)
        if len(indices):
            _, clustered_cv = moments(np.full(len(indices), 2.0), indices)
            bracketed[indices] = clustered_cv >= upper_target[indices]
            upper[indices] = 2
    if not bracketed.all():
        missing = np.flatnonzero(~bracketed)
        raise ValueError(
            f"Could not bracket teacher CV for image {original_index}: "
            f"{len(missing)}/{channels} channels after {attempts} random draws"
            f"{' plus clustered endpoint' if extended else ''}; "
            f"unbracketed target CV range {cv[missing].min():.6g}..{cv[missing].max():.6g}; "
            f"first channel IDs {missing[:8].tolist()}"
        )
    lo, hi = np.zeros(channels), upper
    error, alpha = np.full(channels, np.inf), np.zeros(channels)
    for _ in range(cfg["timing_bisections"]):
        mid = (lo + hi) / 2
        _, candidate_cv = moments(mid)
        candidate_error = np.abs(candidate_cv / cv - 1)
        improved = candidate_error < error
        error[improved], alpha[improved] = candidate_error[improved], mid[improved]
        lo = np.where(candidate_cv < cv, mid, lo)
        hi = np.where(candidate_cv >= cv, mid, hi)
    # Certify the values that are actually persisted, including timing quantization.
    alpha = alpha.astype(np.float32)
    h, _ = moments(alpha)
    q = (mu / h.mean(0)).astype(np.float32)
    g = h * q
    fitted_mean, fitted_sd = g.mean(0), g.std(0)
    error = np.abs(fitted_sd / sd - 1)
    mean_error = np.max(np.abs(fitted_mean / mu - 1))
    if (
        not np.isfinite(q).all()
        or np.any(q <= 0)
        or not np.isfinite(error).all()
        or not np.isfinite(mean_error)
        or error.max() >= cfg["calibration_std_tolerance"]
        or mean_error > 1e-6
    ):
        raise ValueError(
            f"Saved-parameter calibration failed for image {original_index}: "
            f"mean error {mean_error}, SD error {error.max()}"
        )
    return alpha, q, mu, sd, fitted_mean, fitted_sd, float(error.max())


def calibrate(target, cfg, original_index, *, event_count=None):
    if cfg.get("inhibition_method") == recipe.POISSON_METHOD:
        rates, amplitude = poisson_parameters(target, cfg)
        mean, sd = poisson_expected_moments(rates, amplitude, cfg)
        mu, target_sd = target.mean(0), target.std(0)
        mean_error = relative_moment_error(mean, mu).max()
        sd_error = relative_moment_error(sd, target_sd).max()
        if max(mean_error, sd_error) > 1e-6:
            raise ValueError(f"Poisson expectation mismatch for image {original_index}")
        return rates, amplitude, mu, target_sd, mean, sd, float(sd_error)
    if cfg.get("inhibition_method") == recipe.TEACHER_COUNT_METHOD:
        count = int(event_counts(cfg, 1, np.asarray([event_count]))[0])
        if count == 0:
            if not np.isfinite(target).all() or np.any(target != 0):
                raise ValueError(
                    f"Image {original_index}: teacher I count rounds to zero but conductance is nonzero; cannot match with zero artificial events"
                )
            zero = np.zeros(target.shape[1], dtype=np.float32)
            return tuple(zero.copy() for _ in range(7))
        cfg = dict(cfg, target_events_per_source=count)
    return calibrate_count_matched(target, cfg, original_index)


def artificial(
    parameters, amplitudes, indices, cfg, *, epoch=None, draw=None, counts=None
):
    domain, key = ("training", epoch) if epoch is not None else ("validation", draw)
    if key is None:
        raise ValueError("Explicit epoch or draw required")
    if cfg.get("inhibition_method") == recipe.POISSON_METHOD:
        validate_poisson_parameters(parameters, amplitudes)
        return np.stack(
            [
                rng(cfg, domain, key, index)
                .poisson(
                    rate * cfg["dt_ms"] / 1000,
                    size=(
                        cfg["steps"] + cfg["inhibition_prehistory_steps"],
                        cfg["n_e"],
                    ),
                )
                .astype(np.float32)
                * q
                for rate, q, index in zip(parameters, amplitudes, indices, strict=True)
            ],
            axis=1,
        )
    if generator_key(cfg) == "timing_interpolation":
        draws = []
        resolved = event_counts(cfg, len(indices), counts)
        for alpha, q, index, count in zip(
            parameters, amplitudes, indices, resolved, strict=True
        ):
            if count == 0:
                draws.append(
                    np.zeros(
                        (cfg["steps"] + cfg["inhibition_prehistory_steps"], cfg["n_e"]),
                        dtype=np.float32,
                    )
                )
                continue
            gen = rng(cfg, domain, key, index)
            regular, random = timing_endpoints(gen, cfg["steps"], cfg["n_e"], count)
            past = (
                timing_endpoints(gen, cfg["steps"], cfg["n_e"], count)
                if cfg.get("inhibition_prehistory_steps", 0)
                else None
            )
            clusters = (
                tuple(
                    clustered_endpoint(gen, cfg["steps"], cfg["n_e"], count)
                    for _ in range(2)
                )
                if cfg.get("inhibition_method") in recipe.EXTENDED_METHODS
                else None
            )
            draws.append(timing_trial(regular, random, alpha, cfg, past, clusters) * q)
        return np.stack(draws, axis=1)
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
            for p, q, index in zip(parameters, amplitudes, indices, strict=True)
        ],
        axis=1,
    )


def device_generator(cfg, domain, *keys, device):
    """Versioned per-image streams; never consume the process-global RNG."""
    seed = np.random.SeedSequence([cfg[domain + "_seed"], *map(int, keys)])
    value = int(seed.generate_state(1, dtype=np.uint64)[0]) % (2**63 - 1)
    return torch.Generator(device=device).manual_seed(value)


def sensory_device(pixels, indices, cfg, device):
    peak, dt = cfg["input_rate_hz"], cfg["dt_ms"]
    if (
        not math.isfinite(peak)
        or peak < 0
        or not math.isfinite(dt)
        or dt <= 0
        or peak * dt / 1000 > 1
    ):
        raise ValueError("Invalid sensory rate or timestep")
    pixels = (torch.as_tensor(pixels, device=device, dtype=torch.float32) / 255).clamp(
        0, 1
    )
    # Match the native encoder's arithmetic and draw shape without per-image
    # CUDA tensor-to-Python validation; scalar settings are checked above.
    probability = pixels * (
        torch.as_tensor(peak, device=device, dtype=pixels.dtype) * dt / 1000
    )
    drive = torch.empty(
        (cfg["steps"], len(indices), cfg["n_input"]), device=device, dtype=torch.float32
    )
    for row, index in enumerate(indices):
        gen = device_generator(cfg, "sensory", index, device=device)
        values = torch.rand(
            (cfg["steps"], 1, cfg["n_input"]), generator=gen, device=device
        )
        drive[:, row] = (values < probability[row : row + 1].unsqueeze(0)).float()[:, 0]
    return drive


def timing_endpoints_device(gen, cfg, device):
    count, channels, steps = cfg["target_events_per_source"], cfg["n_e"], cfg["steps"]
    regular = (
        torch.arange(count, device=device, dtype=torch.float64)[:, None]
        + torch.rand(channels, generator=gen, device=device, dtype=torch.float64)[None]
    ) * ((steps - 1) / count)
    random = (
        torch.multinomial(
            torch.ones((channels, steps - 1), device=device),
            count,
            replacement=False,
            generator=gen,
        )
        .sort(dim=1)
        .values.T
    )
    return regular, random


def timing_bins_device(endpoints, alpha, cluster, cfg, *, pending=False):
    first, second = alpha.clamp(max=1), alpha.clamp(min=1) - 1
    times = endpoints[0] * (1 - first) + endpoints[1] * (first - second)
    if cluster is not None:
        times = times + cluster * second
    rounded = torch.floor(times + 0.5 + 1e-9) if cluster is not None else times.round()
    return rounded.long().clamp(max=cfg["steps"] - (1 if pending else 2))


def poisson_device(
    parameters,
    amplitudes,
    indices,
    cfg,
    device,
    *,
    epoch=None,
    draw=None,
    retain_history=False,
):
    """Exact Poisson bin counts; GPU preparation stays outside native graph replay."""
    domain, key = ("training", epoch) if epoch is not None else ("validation", draw)
    if key is None:
        raise ValueError("Explicit epoch or draw required")
    validate_poisson_parameters(parameters, amplitudes)
    steps, history, channels = (
        cfg["steps"],
        cfg["inhibition_prehistory_steps"],
        cfg["n_e"],
    )
    if parameters.shape != (len(indices), channels):
        raise ValueError("Poisson parameters do not match the image/channel batch")
    rates = torch.as_tensor(parameters, device=device, dtype=torch.float64)
    amplitude = torch.as_tensor(amplitudes, device=device, dtype=torch.float32)
    offset = history if retain_history else 0
    events = torch.zeros(
        (steps + offset, len(indices), channels), device=device, dtype=torch.float32
    )
    conductance = torch.zeros(
        (len(indices), channels), device=device, dtype=torch.float32
    )
    pending = torch.zeros(
        (1, len(indices), channels), device=device, dtype=torch.float32
    )
    ages = torch.arange(
        max(0, history - 1) - 1, -1, -1, device=device, dtype=torch.float64
    )
    decay_weights = torch.exp(-ages * cfg["dt_ms"] / cfg["tau_gaba_ms"])
    for row, index in enumerate(indices):
        gen = device_generator(cfg, domain, key, index, device=device)
        raw = torch.poisson(
            (rates[row] * cfg["dt_ms"] / 1000).expand(steps + history, -1),
            generator=gen,
        )
        events[offset:, row] = raw[history:].float() * amplitude[row]
        if history:
            conductance[row] = (
                (raw[: history - 1] * decay_weights[:, None]).sum(0)
                * amplitude[row].double()
            ).float()
            pending[0, row] = raw[history - 1].float() * amplitude[row]
            if retain_history:
                events[:history, row] = raw[:history].float() * amplitude[row]
    return events, conductance, pending


def artificial_device(
    parameters,
    amplitudes,
    indices,
    cfg,
    device,
    *,
    epoch=None,
    draw=None,
    retain_history=False,
    counts=None,
):
    """Generate dense events and sparse-history initial state on the target device."""
    if cfg.get("inhibition_method") == recipe.POISSON_METHOD:
        return poisson_device(
            parameters,
            amplitudes,
            indices,
            cfg,
            device,
            epoch=epoch,
            draw=draw,
            retain_history=retain_history,
        )
    domain, key = ("training", epoch) if epoch is not None else ("validation", draw)
    if key is None:
        raise ValueError("Explicit epoch or draw required")
    steps, history, channels = (
        cfg["steps"],
        cfg["inhibition_prehistory_steps"],
        cfg["n_e"],
    )
    alpha = torch.as_tensor(parameters, device=device, dtype=torch.float64)
    amplitude = torch.as_tensor(amplitudes, device=device, dtype=torch.float32)
    offset = history if retain_history else 0
    events = torch.zeros(
        (steps + offset, len(indices), channels), device=device, dtype=torch.float32
    )
    conductance = torch.zeros(
        (len(indices), channels), device=device, dtype=torch.float32
    )
    pending = torch.zeros(
        (1, len(indices), channels), device=device, dtype=torch.float32
    )
    resolved = event_counts(cfg, len(indices), counts)
    for row, (index, count) in enumerate(zip(indices, resolved, strict=True)):
        count = int(count)
        if count == 0:
            continue
        local_cfg = dict(cfg, target_events_per_source=count)
        gen = device_generator(cfg, domain, key, index, device=device)
        scored = timing_endpoints_device(gen, local_cfg, device)
        past = timing_endpoints_device(gen, local_cfg, device) if history else None
        clusters = [None, None]
        if cfg["inhibition_method"] in recipe.EXTENDED_METHODS:
            clusters = [
                torch.randint(steps - count, (channels,), generator=gen, device=device)[
                    None
                ]
                + torch.arange(count, device=device)[:, None]
                for _ in range(2)
            ]
        bins = timing_bins_device(scored, alpha[row], clusters[0], cfg)
        events[offset:, row].scatter_(0, bins, amplitude[row].expand(count, -1))
        if past is not None:
            bins = timing_bins_device(past, alpha[row], clusters[1], cfg, pending=True)
            age = steps - 2 - bins
            active = (bins >= steps - history) & (age >= 0)
            weights = torch.exp(
                -age.clamp(min=0).double() * cfg["dt_ms"] / cfg["tau_gaba_ms"]
            )
            conductance[row] = (weights * active * amplitude[row].double()).sum(dim=0)
            pending[0, row] = (bins == steps - 1).any(dim=0) * amplitude[row]
            if retain_history:
                # The extra discarded row receives events preceding the history window.
                tail = torch.zeros(
                    (history + 1, channels), device=device, dtype=torch.float32
                )
                positions = (bins - (steps - history)).clamp(min=-1) + 1
                tail.scatter_(0, positions, amplitude[row].expand(count, -1))
                events[:history, row] = tail[1:]
    return events, conductance, pending


def validate_source_configuration(conf, cfg):
    expected = {
        "dt": 0.1,
        "t_ms": 200.0,
        "max_samples": 7000,
        "input_rate": 25.0,
        "fr_reg_upper_strength": cfg["rate_penalty"],
        "fr_reg_upper_target_hz": cfg.get("rate_target_hz") or 0.0,
        "readout_mode": "mem-mean",
        "seed": cfg["seed"],
        "n_in": 784,
        "n_hidden": 1024,
        "n_inh": 256,
        "n_out": 10,
        "tau_gaba_ms": 6.0,
        "tau_ampa_ms": 2.0,
        "v_grad_dampen": 1000.0,
    }
    if any(conf[k] != v for k, v in expected.items()):
        raise ValueError("Source settings do not match the ceiling and seed")


def source_bank(cfg, *, root=None):
    """Resolve a full bank or compact export and its original bank reference."""
    bank = source_run(
        (ROOT if root is None else root) / ".pingstore",
        cfg["source_run"],
        experiment="exp022",
        reference={
            "run_id": cfg["source_run"],
        },
    )
    if bank.details["stage"] == "analyse":
        if bank.details["execution"]["operation"] != "export-finetuning-inputs":
            raise ValueError("Source must be a fine-tuning input export")
        if set(bank.details["inputs"]) != {"bank"}:
            raise ValueError("Compact source must pin exactly one original bank")
        original = bank.details["inputs"]["bank"]
        if not original["run_id"].startswith("exp022-") or not original[
            "run_id"
        ].endswith("-compute"):
            raise ValueError("Compact source must pin an exp022 compute bank")
    elif bank.details["stage"] == "compute":
        original = bank.reference
    else:
        raise ValueError("Unsupported source stage")
    return bank, original


def load_source(cfg):
    bank, _ = source_bank(cfg)
    provenance = (
        bank.details["execution"]["configuration"]["diagnostic_provenance"]
        if bank.details["stage"] == "analyse"
        else bank.details["bank_reuse"]["diagnostic_provenance"]
    )
    params, configurations, pins = {}, {}, {}
    for kind in ("coba", "ping"):
        cell = cfg[kind + "_cell"]
        conf = json.loads(bank.file(cell, "config.json").read_text())
        validate_source_configuration(conf, cfg)
        pin = provenance[cell]["checkpoint"]
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


CALIBRATION_KEYS = (
    "source_run",
    "source_digest",
    "ping_cell",
    "seed",
    "checkpoint_role",
    "source_checkpoints",
    "dataset_sha256",
    "encoding_method",
    "sensory_seed",
    "calibration_seed",
    "teacher_statistics",
    "train_images",
    "validation_images",
    "dt_ms",
    "duration_ms",
    "steps",
    "input_rate_hz",
    "tau_gaba_ms",
    "tau_ampa_ms",
    "n_input",
    "n_e",
    "n_i",
    "n_output",
    "voltage_grad_dampen",
    "surrogate_slope",
    "refractory_timing",
    "inhibition_method",
    "target_events_per_source",
    "timing_bisections",
    "bracket_attempts",
    "calibration_std_tolerance",
    "inhibition_prehistory_steps",
    "device",
)


def check_calibration_configuration(previous, cfg):
    for key in CALIBRATION_KEYS:
        if key not in previous or previous[key] != cfg[key]:
            raise ValueError(f"Calibration reuse mismatch: {key}")


def reuse_calibration(source, cfg, ids, pixels, labels, snapshot):
    """Reuse only an explicit, authenticated and scientifically compatible input."""
    check_calibration_configuration(source.details["execution"]["configuration"], cfg)
    with np.load(source.file("dataset.npz")) as data:
        for key, expected in (
            ("ids", ids),
            ("features", pixels[ids]),
            ("labels", labels[ids]),
        ):
            np.testing.assert_array_equal(data[key], expected)
        alpha = data[generator_key(cfg)].copy()
        amplitude = data["amplitude"].copy()
        saved_counts = dataset_event_counts(data, cfg, slice(None))
    with np.load(source.file("calibration.npz")) as cal:
        np.testing.assert_array_equal(cal["ids"], ids)
        arrays = [
            cal[key].copy()
            for key in (
                "target_mean",
                "target_sd",
                *moment_keys(cfg),
                "max_relative_sd_error",
                "teacher_i_spike_count",
                "teacher_i_rate_hz",
            )
        ]
        if cfg["inhibition_method"] == recipe.TEACHER_COUNT_METHOD:
            expected_counts = teacher_event_counts(arrays[-2], cfg)
            np.testing.assert_array_equal(saved_counts, expected_counts)
            np.testing.assert_array_equal(cal["event_count"], expected_counts)
    if cfg["inhibition_method"] == recipe.POISSON_METHOD:
        validate_poisson_parameters(alpha, amplitude)
        expected_mu, expected_sd = poisson_expected_moments(alpha, amplitude, cfg)
        np.testing.assert_allclose(expected_mu, arrays[0], rtol=1e-6, atol=0)
        np.testing.assert_allclose(expected_sd, arrays[1], rtol=1e-6, atol=0)
        np.testing.assert_allclose(expected_mu, arrays[2], rtol=1e-6, atol=0)
        np.testing.assert_allclose(expected_sd, arrays[3], rtol=1e-6, atol=0)
    shape = (len(ids), cfg["n_e"])
    if (
        any(
            a.shape != shape or not np.isfinite(a).all()
            for a in [alpha, amplitude, *arrays[:4]]
        )
        or (
            cfg["inhibition_method"] != recipe.POISSON_METHOD
            and np.any((alpha < 0) | (alpha > 2))
        )
        or np.any(amplitude < 0)
        or any(np.any(a < 0) for a in arrays[:4])
        or np.max(relative_moment_error(arrays[2], arrays[0])) > 1e-6
        or np.max(relative_moment_error(arrays[3], arrays[1]))
        >= cfg["calibration_std_tolerance"]
    ):
        raise ValueError("Invalid cached calibration")
    if cfg["inhibition_method"] == recipe.TEACHER_COUNT_METHOD:
        if np.any((amplitude == 0) != (saved_counts[:, None] == 0)):
            raise ValueError("Cached amplitudes disagree with per-image silence")
        silent = saved_counts == 0
        if any(np.any(a[silent] != 0) for a in arrays[:4]):
            raise ValueError("Zero-count cached calibration has nonzero moments")
    elif cfg["inhibition_method"] != recipe.POISSON_METHOD and np.any(amplitude == 0):
        raise ValueError("Invalid zero cached amplitude")
    counts, rates = arrays[-2:]
    if (
        counts.shape != (len(ids), cfg["n_i"])
        or rates.shape != counts.shape
        or not np.issubdtype(counts.dtype, np.integer)
        or np.any(counts < 0)
        or np.any(counts > cfg["steps"])
    ):
        raise ValueError("Invalid cached teacher activity")
    np.testing.assert_allclose(rates, counts / (cfg["duration_ms"] / 1000), rtol=1e-6)
    source.restore_bundle("teacher/start", snapshot)
    source.check_unchanged()
    return [alpha, amplitude, *arrays]


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


ENCODING_PROFILE = []


def profile_preparation(cfg, context, name, device, prepare):
    """Opt-in synchronized encoding measurements outside native graph replay."""
    if not cfg.get("profile_encoding", False):
        return prepare()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    tick = time.perf_counter()
    result = prepare()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    ENCODING_PROFILE.append(
        dict(
            phase=context.phase,
            epoch=context.epoch,
            component=name,
            images=len(context.sample_ids),
            seconds=time.perf_counter() - tick,
        )
    )
    return result


def test_sensory(arrays, selected, cfg, device):
    """Restore manuscript CPU RNG states, including time-major batch ordering."""
    selected = np.asarray(selected)
    batch = cfg["test_batch_size"]
    if not len(selected):
        raise ValueError("Empty test encoding batch")
    start = int(selected[0]) // batch * batch
    stop = min(start + batch, len(arrays["features"]))
    if not len(selected) or np.any(selected < start) or np.any(selected >= stop):
        raise ValueError("Test encoding must stay inside one manuscript batch")
    gen = torch.Generator().set_state(
        torch.from_numpy(arrays["test_rng_states"][start // batch].copy())
    )
    images = torch.from_numpy(arrays["features"][start:stop].copy()).float() / 255
    drive = sim.encode_images_poisson(
        images, cfg["steps"], cfg["dt_ms"], cfg["input_rate_hz"], generator=gen
    )
    return drive[:, selected - start].to(device)


def encode_native(
    arrays, selected, *, dt_ms, channels, config, seed, context, device="cpu"
):
    cfg = config["recipe"]
    indices = np.asarray(context.sample_ids)
    if cfg.get("encoding_method") in {"torch_device_v1", "test_protocol_v1"}:
        coordinate = (
            {"epoch": context.epoch + 1}
            if context.phase == "train"
            else {
                "draw": cfg.get("final_training_draw", 0)
                if context.phase == "inference"
                else context.draw_seed
            }
        )
        events, conductance, pending = profile_preparation(
            cfg,
            context,
            "artificial",
            device,
            lambda: artificial_device(
                arrays[generator_key(cfg)][selected],
                arrays["amplitude"][selected],
                indices,
                cfg,
                device,
                counts=dataset_event_counts(arrays, cfg, selected),
                **coordinate,
            ),
        )
        return sim.PreparedBatch(
            {
                "drive": profile_preparation(
                    cfg,
                    context,
                    "sensory",
                    device,
                    lambda: (
                        test_sensory(arrays, selected, cfg, device)
                        if cfg["encoding_method"] == "test_protocol_v1"
                        else sensory_device(
                            arrays["features"][selected], indices, cfg, device
                        )
                    ),
                ),
                "artificial": events,
            },
            sim.PresentationState(
                signature=config["signature"],
                conductances={"artificial_to_E": conductance},
                input_histories={"artificial": pending},
                provenance={
                    "policy": "private_gaba_only",
                    "history_steps": cfg["inhibition_prehistory_steps"],
                    "encoding_method": cfg["encoding_method"],
                },
            ),
        )
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
        else {
            "draw": cfg.get("final_training_draw", 0)
            if context.phase == "inference"
            else context.draw_seed
        }
    )
    events = artificial(
        arrays[generator_key(cfg)][selected],
        arrays["amplitude"][selected],
        indices,
        cfg,
        counts=dataset_event_counts(arrays, cfg, selected),
        **coordinate,
    )
    history = cfg["inhibition_prehistory_steps"]
    conductance = (
        filtered(events[:history], cfg)[-1].astype(np.float32)
        if history
        else np.zeros((len(indices), cfg["n_e"]), dtype=np.float32)
    )
    return sim.PreparedBatch(
        {
            "drive": torch.from_numpy(drive),
            "artificial": torch.from_numpy(events[history:]),
        },
        sim.PresentationState(
            signature=config["signature"],
            conductances={"artificial_to_E": torch.from_numpy(conductance)},
            input_histories={
                "artificial": torch.from_numpy(
                    events[history - 1 : history]
                    if history
                    else np.zeros_like(events[:1])
                )
            },
            provenance={"policy": "private_gaba_only", "history_steps": history},
        ),
    )


def performance(cfg, *, calibration=False):
    return sim.Performance(
        compile_mode=cfg["compile_mode"],
        precision="highest" if calibration else cfg["precision"],
    )


def parameter_file(bundle, parameters, cfg, destination):
    """Adapt the pinned legacy weight names to a native inference checkpoint."""
    built = sim.build(
        bundle=bundle,
        seed=cfg["seed"],
        device=cfg["device"],
        performance=performance(cfg),
    )
    if set(built.model.parameter_map()) != set(parameters):
        raise ValueError("Unexpected native parameter membership")
    with torch.no_grad():
        for name, value in built.model.parameter_map().items():
            value.copy_(torch.as_tensor(parameters[name], device=value.device))
    torch.save(built.model.state_dict(), destination)


def dataset_encoder(bundle, cfg):
    name = "pinglab.exp171/v1"
    registered = next(
        (row for row in extensions.definitions("encoder") if row.name == name), None
    )
    if registered is None:
        extensions.register_encoder(name, encode_native)
    elif registered.function is not encode_native:
        raise ValueError(
            f"Encoder definition already belongs to another function: {name}"
        )
    keys = (
        "inhibition_method",
        "encoding_method",
        "target_events_per_source",
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
        "final_training_draw",
        "profile_encoding",
    )
    if cfg.get("encoding_method") == "test_protocol_v1":
        keys += ("test_batch_size",)
    return sim.DatasetEncoder(
        "custom",
        definition=name,
        rng_policy="sample_v1",
        duration_ms=cfg["duration_ms"],
        config={
            "recipe": {
                key: cfg.get(key, False) if key == "profile_encoding" else cfg[key]
                for key in keys
            },
            "signature": sim.runtime_state_signature(sim.plan_graph(bundle.graph)),
        },
    )


def dataset_binding(path, encoder, split, positions):
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


def calibrate_dataset(
    bundle, checkpoint, pixels, indices, cfg, snapshot, timing=None, drive_encoder=None
):
    rows, i_counts = [], []
    model = None
    pending = deque()
    batch = cfg["calibration_batch_size"]
    timing = timing if timing is not None else {}
    timing.update(teacher_batches=[], fit_seconds=[], pipeline_wait_seconds=0.0)

    def fit_image(index, trace, count):
        tick = time.perf_counter()
        kwargs = (
            {"event_count": int(count)}
            if cfg["inhibition_method"] == recipe.TEACHER_COUNT_METHOD
            else {}
        )
        values = calibrate(trace.astype(np.float64), cfg, index, **kwargs)
        return values, time.perf_counter() - tick

    def collect():
        tick = time.perf_counter()
        for future in pending.popleft():
            values, seconds = future.result()
            rows.append(values)
            timing["fit_seconds"].append(seconds)
        timing["pipeline_wait_seconds"] += time.perf_counter() - tick
        print(f"calibration {len(rows)}/{len(indices)}", flush=True)

    with ThreadPoolExecutor(max_workers=cfg["threads"]) as pool:
        for start in range(0, len(indices), batch):
            if len(pending) >= cfg["calibration_inflight_batches"]:
                collect()
            tick = time.perf_counter()
            ids = indices[start : start + batch]
            drive = (
                drive_encoder(pixels[ids], ids, start)
                if drive_encoder
                else (
                    sensory_device(pixels[ids], ids, cfg, cfg["device"])
                    if cfg.get("encoding_method") == "torch_device_v1"
                    else torch.from_numpy(
                        np.stack(
                            [
                                sensory(pixels[index] / np.float32(255), index, cfg)
                                for index in ids
                            ],
                            axis=1,
                        )
                    )
                )
            )
            if model is None:
                result = sim.simulate(
                    bundle=bundle,
                    checkpoint=checkpoint,
                    input_bindings=[sim.DenseArrayBinding("drive", drive)],
                    seed=cfg["seed"],
                    device=cfg["device"],
                    performance=performance(cfg, calibration=True),
                    snapshot=sim.SnapshotSpec(snapshot, stage="start"),
                    diagnostics=False,
                    retain_parameters=False,
                    retain_final_state=False,
                    retain_runtime_state=False,
                )
                model = result.model
            else:
                result = model(
                    {"drive": drive},
                    diagnostics=False,
                    retain_parameters=False,
                    retain_final_state=False,
                    retain_runtime_state=False,
                    batch_offset=start,
                )
            conductance = result.outputs["g_ping"].detach().cpu().numpy()
            counts = result.outputs["i_count"].detach().cpu().numpy()
            timing["teacher_batches"].append(
                dict(
                    images=len(ids),
                    first_image=start,
                    seconds=time.perf_counter() - tick,
                    scope="encoding, teacher execution and output transfer; first batch includes load/snapshot",
                )
            )
            if (
                counts.shape != (len(ids), cfg["n_i"])
                or not np.isfinite(counts).all()
                or np.any(counts < 0)
                or np.any(counts > cfg["steps"])
                or not np.equal(counts, np.rint(counts)).all()
            ):
                raise ValueError("Invalid teacher inhibitory spike counts")
            i_counts.append(counts.astype(np.int32))
            per_image_count = (
                teacher_event_counts(counts, cfg)
                if cfg["inhibition_method"] == recipe.TEACHER_COUNT_METHOD
                else np.zeros(len(ids), dtype=np.int32)
            )
            pending.append(
                [
                    pool.submit(fit_image, index, trace, count)
                    for index, trace, count in zip(
                        ids,
                        np.moveaxis(conductance, 1, 0),
                        per_image_count,
                        strict=True,
                    )
                ]
            )
        while pending:
            collect()
    counts = np.concatenate(i_counts)
    return [np.stack(column) for column in zip(*rows, strict=True)] + [
        counts,
        counts.astype(np.float32) / (cfg["duration_ms"] / 1000),
    ]


def record_selected(
    run, parameters, dataset, cfg, epoch, *, position=0, selection_policy=None
):
    """Native selected-weight snapshot and one fixed illustrative validation trial."""
    observed = recipe.author_network(cfg, record_spikes=True)
    standalone = getattr(run, "record", {}).get("phase") == "raster"
    if standalone:
        observed.write(run.export / "network.bundle", visualise=False)
        observed.visualise(run.export / "network-circuit.svg", view="circuit")
    start = cfg["train_images"]
    stop = start + min(cfg["replay_batch_size"], cfg["validation_images"])
    with np.load(dataset) as z:
        indices, labels = z["ids"][start:stop], z["labels"][start:stop]
        if type(position) is not int or not 0 <= position < len(indices):
            raise ValueError("Raster position is outside the retained batch")
        if cfg.get("encoding_method") == "torch_device_v1":
            drive = (
                sensory_device(z["features"][start:stop], indices, cfg, cfg["device"])
                .cpu()
                .numpy()
            )
            events = (
                artificial_device(
                    z[generator_key(cfg)][start:stop],
                    z["amplitude"][start:stop],
                    indices,
                    cfg,
                    cfg["device"],
                    draw=0,
                    retain_history=True,
                    counts=dataset_event_counts(z, cfg, slice(start, stop)),
                )[0]
                .cpu()
                .numpy()
            )
        else:
            drive = np.stack(
                [
                    sensory(row / np.float32(255), index, cfg)
                    for row, index in zip(
                        z["features"][start:stop], indices, strict=True
                    )
                ],
                axis=1,
            )
            events = artificial(
                z[generator_key(cfg)][start:stop],
                z["amplitude"][start:stop],
                indices,
                cfg,
                draw=0,
                counts=dataset_event_counts(z, cfg, slice(start, stop)),
            )
        if cfg.get("inhibition_method") == recipe.POISSON_METHOD:
            q = z["amplitude"][None, start:stop]
            unscaled = np.rint(
                np.divide(events, q, out=np.zeros_like(events), where=q > 0)
            )
            if (
                not np.isfinite(unscaled).all()
                or np.any(unscaled < 0)
                or np.any(unscaled > np.iinfo(np.uint32).max)
            ):
                raise ValueError("Unrepresentable retained Poisson counts")
            retained_events = unscaled.astype(np.uint32)
        else:
            retained_events = events > 0
    checkpoint = run.scratch / "selected.pt"
    parameter_file(observed, parameters, cfg, checkpoint)
    history = cfg["inhibition_prehistory_steps"]
    encoder = dataset_encoder(observed, {**cfg, "final_training_draw": 0})
    binding = dataset_binding(dataset, encoder, "validation", range(start, stop))
    tick = time.perf_counter()
    result = sim.simulate(
        bundle=observed,
        checkpoint=checkpoint,
        input_bindings=[binding],
        batch_size=stop - start,
        seed=cfg["seed"],
        device=cfg["device"],
        performance=performance(cfg),
        snapshot=sim.SnapshotSpec(
            run.export / "best-validation"
            if standalone
            else run.export / "best-validation" / "end",
            stage="both" if standalone else "end",
        ),
        diagnostics=False,
        retain_parameters=False,
        retain_final_state=False,
        retain_runtime_state=False,
    )
    out = {key: value.detach().cpu().numpy() for key, value in result.outputs.items()}
    if out["I_spikes"].any():
        raise ValueError("Disabled biological I population spiked")
    error = float(
        np.max(
            np.abs(
                out["g_artificial"][:, position]
                - filtered(events[:, position : position + 1], cfg)[history:, 0]
            )
        )
    )
    if error > 2e-5:
        raise ValueError("Native selected replay disagrees with the inhibitory filter")
    if selection_policy == "first_correct_in_retained_validation_batch" and (
        int(out["scores"][position].argmax()) != int(labels[position])
    ):
        raise ValueError("Selected replay is not correctly classified")
    rate = out["e_count"].mean(1) / (cfg["duration_ms"] / 1000)
    np.savez_compressed(
        run.export / "raster.npz",
        E_spikes=out["E_spikes"][:, position].astype(bool),
        I_spikes=out["I_spikes"][:, position].astype(bool),
        artificial_events=retained_events[history:, position],
        prehistory_events=retained_events[:history, position],
        g_artificial=out["g_artificial"][:, position],
        batch_indices=indices,
        batch_labels=labels,
        batch_scores=out["scores"],
        batch_rates_hz=rate,
        sensory_packed=np.packbits(drive[:, position].astype(bool), axis=0),
        **(
            {"batch_artificial_counts": retained_events}
            if cfg.get("inhibition_method") == recipe.POISSON_METHOD
            else {"batch_artificial_packed": np.packbits(retained_events, axis=0)}
        ),
    )
    summary = dict(
        epoch=epoch,
        batch_position=position,
        selection_policy=selection_policy
        or (
            "first_validation_image"
            if position == 0
            else "fixed_validation_batch_position"
        ),
        original_index=int(indices[position]),
        label=int(labels[position]),
        predicted=int(out["scores"][position].argmax()),
        validation_draw=0,
        precision_policy=cfg["precision"],
        replayed_batch_size=len(indices),
        e_rate_hz=float(rate[position]),
        native_filter_max_abs_error_us=error,
        simulation_seconds=time.perf_counter() - tick,
        checkpoint_selection="Lowest validation cross-entropy; accuracy then earliest epoch break ties",
        replay_scope="Fixed first validation batch; illustrative replay, not a new validation measurement",
    )
    if cfg["inhibition_method"] == recipe.TEACHER_COUNT_METHOD:
        with np.load(dataset) as z:
            count = int(z["event_count"][start + position])
        summary.update(
            event_count=count, artificial_rate_hz=count / (cfg["duration_ms"] / 1000)
        )
    if cfg.get("inhibition_method") == recipe.POISSON_METHOD:
        with np.load(dataset) as z:
            rates = z["poisson_rate_hz"][start + position]
        summary.update(
            artificial_rate_hz=float(rates.mean()),
            realized_artificial_rate_hz=float(
                retained_events[history:, position].sum()
                / cfg["n_e"]
                / (cfg["duration_ms"] / 1000)
            ),
            realized_conductance_mean_us=float(out["g_artificial"][:, position].mean()),
            realized_mean_temporal_sd_us=float(
                out["g_artificial"][:, position].std(axis=0).mean()
            ),
        )
    write_json(run.export / "raster.json", summary)
    return summary


def evaluate_final_training(bundle, checkpoint, dataset, encoder, cfg):
    """One fixed draw over the full training split at final-epoch weights."""
    tick = time.perf_counter()
    binding = dataset_binding(dataset, encoder, "train", range(cfg["train_images"]))
    with torch.no_grad():
        result = sim.simulate(
            bundle=bundle,
            checkpoint=checkpoint,
            input_bindings=[binding],
            batch_size=cfg["microbatch_size"] or cfg["batch_size"],
            seed=cfg["shuffle_seed"],
            device=cfg["device"],
            performance=performance(cfg),
            diagnostics=False,
            retain_parameters=False,
            retain_final_state=False,
            retain_runtime_state=False,
        )
    scores = result.outputs["scores"].detach().cpu().float()
    counts = result.outputs["e_count"].detach().cpu().float()
    with np.load(dataset) as data:
        labels = torch.from_numpy(data["labels"][: cfg["train_images"]])
    if not torch.isfinite(scores).all() or not torch.isfinite(counts).all():
        raise ValueError("Nonfinite final training evaluation")
    return dict(
        checkpoint="final.checkpoint",
        epoch=cfg["epochs"],
        examples=len(labels),
        draw=cfg["final_training_draw"],
        cross_entropy=float(torch.nn.functional.cross_entropy(scores, labels)),
        accuracy_pct=float(100 * (scores.argmax(-1) == labels).float().mean()),
        e_rate_hz=float(counts.mean() / (cfg["duration_ms"] / 1000)),
        seconds=time.perf_counter() - tick,
        performance=result.metrics["performance"],
    )


def compute(
    run_id=None,
    *,
    epochs=None,
    train_images=None,
    compile_mode="auto",
    microbatch_size=None,
    rate_target_hz=None,
    seed=42,
    calibration_source=None,
    source=None,
    source_digest=None,
    profile=False,
):
    started = time.perf_counter()
    overrides = {
        key: value
        for key, value in dict(
            epochs=epochs, train_images=train_images, microbatch_size=microbatch_size
        ).items()
        if value is not None
    }
    if (source is None) != (source_digest is None):
        raise ValueError("Source identity and digest must be supplied together")
    if source is not None:
        overrides.update(source_run=source, source_digest=source_digest)
    cfg = recipe.configuration(
        device=os.environ.get("SNNLAB_DEVICE", "cpu"),
        threads=int(os.environ.get("OMP_NUM_THREADS", "1")),
        compile_mode=compile_mode,
        rate_target_hz=rate_target_hz,
        seed=seed,
        **overrides,
    )
    cfg["profile_encoding"] = profile
    ENCODING_PROFILE.clear()
    torch.set_num_threads(cfg["threads"])
    bank, parameters = load_source(cfg)
    dataset = MNIST(
        root=os.environ.get("PINGLAB_DATASET_CACHE", "/tmp/mnist"),
        train=True,
        download=True,
    )
    labels = dataset.targets.numpy().astype(np.int64)
    pixels = dataset.data.numpy().reshape(-1, cfg["n_input"])
    source_train, source_validation = source_indices(labels)
    train, validation = (
        source_train[: cfg["train_images"]],
        source_validation[: cfg["validation_images"]],
    )
    ids = np.concatenate((train, validation))
    cfg["dataset_sha256"] = {"pixels": digest(pixels), "labels": digest(labels)}
    cfg["source_code_sha256"] = {
        str(p.relative_to(Path(__file__).parent)): file_sha256(p)
        for p in Path(__file__).parent.rglob("*.py")
    }
    inputs = {"bank": bank}
    cached = None
    if calibration_source:
        cached = source_run(
            ROOT / ".pingstore",
            calibration_source,
            experiment=recipe.SLUG,
            stage="compute",
        )
        check_calibration_configuration(
            cached.details["execution"]["configuration"], cfg
        )
        inputs["calibration"] = cached
    with stage_run(
        ROOT,
        recipe.SLUG,
        "compute",
        inputs=inputs,
        configuration=cfg,
        run_id=run_id,
    ) as run:
        bundle, teacher = (
            recipe.author_network(cfg),
            recipe.author_network(cfg, teacher=True),
        )
        for name, authored in (("network", bundle), ("teacher", teacher)):
            authored.write(run.export / f"{name}.bundle", visualise=False)
            authored.visualise(run.export / f"{name}-circuit.svg", view="circuit")
        teacher_weights = run.scratch / "teacher.pt"
        parameter_file(teacher, parameters["ping"], cfg, teacher_weights)
        tick = time.perf_counter()
        calibration_timing = {}
        alpha, amplitude, mu, sd, fitted_mu, fitted_sd, error, i_count, i_rate = (
            reuse_calibration(
                cached, cfg, ids, pixels, labels, run.export / "teacher" / "start"
            )
            if cached
            else calibrate_dataset(
                teacher,
                teacher_weights,
                pixels,
                ids,
                cfg,
                run.export / "teacher" / "start",
                calibration_timing,
            )
        )
        calibration_seconds = time.perf_counter() - tick
        calibration_timing["total_seconds"] = calibration_seconds
        calibration_timing["reused"] = cached is not None
        write_json(run.export / "calibration-timing.json", calibration_timing)
        count_arrays = (
            {"event_count": teacher_event_counts(i_count, cfg)}
            if cfg["inhibition_method"] == recipe.TEACHER_COUNT_METHOD
            else {}
        )
        path = run.export / "dataset.npz"
        np.savez_compressed(
            path,
            ids=ids,
            features=pixels[ids],
            labels=labels[ids],
            source_train=source_train,
            source_validation=source_validation,
            **{generator_key(cfg): alpha},
            amplitude=amplitude,
            **count_arrays,
        )
        np.savez_compressed(
            run.export / "calibration.npz",
            ids=ids,
            teacher_i_spike_count=i_count,
            teacher_i_rate_hz=i_rate,
            target_mean=mu,
            target_sd=sd,
            **dict(zip(moment_keys(cfg), (fitted_mu, fitted_sd), strict=True)),
            max_relative_sd_error=error,
            **count_arrays,
        )
        params = {
            **parameters["coba"],
            "artificial_to_E.weight": torch.ones(1, cfg["n_e"]),
        }
        encoder = dataset_encoder(bundle, cfg)
        training = dataset_binding(path, encoder, "train", range(len(train)))
        validation_binding = dataset_binding(
            path, encoder, "validation", range(len(train), len(ids))
        )
        result = sim.train(
            bundle=bundle,
            weights=sim.WeightInitialization(params),
            input_bindings=[training],
            validation=sim.ValidationSpec(
                input_bindings=[validation_binding],
                encoding_seeds=tuple(range(cfg["validation_draws"])),
            ),
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            batching=sim.Batching(
                order="shuffle", microbatch_size=cfg["microbatch_size"]
            ),
            seed=cfg["shuffle_seed"],
            device=cfg["device"],
            performance=performance(cfg),
            training_timing=True,
            diagnostics=False,
            metrics=sim.TrainingMetrics(train="none"),
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
        counts = result.metrics["batch_counts"]
        if counts["optimizer_updates"] != cfg["epochs"] * int(
            np.ceil(len(train) / cfg["batch_size"])
        ) or counts["examples"] != cfg["epochs"] * len(train):
            raise ValueError("Native training did not complete the requested schedule")
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
        metrics["final_training"] = evaluate_final_training(
            bundle, run.export / "final.checkpoint", path, encoder, cfg
        )
        metrics["parameter_changes"] = changes
        write_json(run.export / "native-metrics.json", metrics)
        best_epoch = int(result.selected_checkpoint.data_state["epoch"])
        replay = record_selected(
            run, result.selected_checkpoint.parameters, path, cfg, best_epoch
        )
        timings = dict(
            calibration_seconds=calibration_seconds,
            training_seconds=result.metrics["training_timing"]["total_s"],
            final_training_evaluation_seconds=metrics["final_training"]["seconds"],
            selected_replay_seconds=replay["simulation_seconds"],
            compute_seconds=time.perf_counter() - started,
        )
        write_json(run.export / "timings.json", timings)
        if profile:
            write_json(
                run.export / "encoding-profile.json",
                {
                    "scope": "synchronized CUDA encoding; includes allocation, transfers and RNG; profiling changes overlap",
                    "batches": ENCODING_PROFILE,
                    "host_peak_rss_bytes": resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss
                    * (1 if sys.platform == "darwin" else 1024),
                },
            )
        run.details["scientific_artifacts"] = {
            "snapshot_stages": {
                "training": ["start", "end"],
                "teacher": ["start"],
                "best-validation": ["end"],
            }
        }
        run.details["execution"].update(
            no_simulation=False,
            elapsed_seconds=timings["compute_seconds"],
            timings=timings,
        )
        bank.check_unchanged()
        if cached:
            cached.check_unchanged()
    print(run.run_id, flush=True)
    return run.run_id


def evaluate_test_inputs(identity, run_id=None):
    """Evaluate frozen final students; PING runs only to calibrate their input."""
    started = time.perf_counter()
    package = source_run(
        ROOT / ".pingstore", identity, experiment=recipe.SLUG, stage="analyse"
    )
    spec = json.loads(package.file("test-inputs.json").read_text())
    protocol = recipe.test_configuration()
    if spec["protocol"] != protocol:
        raise ValueError("Unsupported test protocol")
    device = os.environ.get("SNNLAB_DEVICE", "cpu")
    threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    torch.set_num_threads(threads)
    dataset = MNIST(
        root=os.environ.get("PINGLAB_DATASET_CACHE", "/tmp/mnist"),
        train=False,
        download=True,
    )
    pixels = dataset.data.numpy().reshape(-1, 784)
    labels = dataset.targets.numpy().astype(np.int64)
    if pixels.shape != (10000, 784) or labels.shape != (10000,):
        raise ValueError("Expected untouched official MNIST test partition")
    indices = np.random.RandomState(protocol["subset_seed"]).choice(
        10000, protocol["samples"], replace=False
    )
    configuration = dict(
        protocol=protocol,
        checkpoint_role="final_epoch",
        device=device,
        dataset_sha256=dict(pixels=digest(pixels), labels=digest(labels)),
        training_runs=[c["training"] for c in spec["cases"]],
    )
    cases = []
    with stage_run(
        ROOT,
        recipe.SLUG,
        "compute",
        inputs={"test_inputs": package},
        configuration=configuration,
        operation="test-final-checkpoints",
        run_id=run_id,
    ) as run:
        np.savez_compressed(
            run.export / "test-subset.npz",
            indices=indices,
            labels=labels[indices],
            features=pixels[indices],
        )
        snapshot_stages = {}
        for case in spec["cases"]:
            label = case["label"]
            out = run.export / label
            out.mkdir()
            cfg = dict(
                case["configuration"],
                device=device,
                threads=threads,
                precision="highest",
                compile_mode="default" if device == "cuda" else None,
                calibration_batch_size=protocol["batch_size"],
                profile_encoding=False,
                encoding_method="test_protocol_v1",
                test_batch_size=protocol["batch_size"],
                validation_seed=protocol["inhibition_seed"],
                final_training_draw=0,
            )
            if cfg["inhibition_method"] != recipe.POISSON_METHOD:
                raise ValueError(
                    "Endpoint evaluation requires the Poisson-trained students"
                )
            teacher = lang.load_bundle(
                package.restore_bundle(
                    label + "/teacher.bundle", run.scratch / label / "teacher.bundle"
                )
            )
            student = lang.load_bundle(
                package.restore_bundle(
                    label + "/network.bundle", run.scratch / label / "network.bundle"
                )
            )
            teacher.write(out / "teacher.bundle", visualise=False)
            student.write(out / "network.bundle", visualise=False)
            if not cases:
                teacher.visualise(run.export / "teacher-circuit.svg", view="circuit")
                student.visualise(run.export / "network-circuit.svg", view="circuit")
            teacher_state = sim.NetworkSnapshot.read(
                package.restore_bundle(
                    label + "/teacher/start", run.scratch / label / "teacher/start"
                ),
                parameters=True,
            ).load_parameters()
            teacher_checkpoint = run.scratch / label / "teacher.pt"
            parameter_file(teacher, teacher_state, cfg, teacher_checkpoint)
            final_path = package.restore_bundle(
                label + "/final.checkpoint", run.scratch / label / "final.checkpoint"
            )
            final = sim.load_training_checkpoint(final_path)
            if final.data_state["epoch"] != case["configuration"]["epochs"]:
                raise ValueError("Test weights must be the final training epoch")
            states, hashes = [], []
            gen = torch.Generator().manual_seed(protocol["encoder_seed"])

            def encode(
                images, ids, start, *, states=states, gen=gen, cfg=cfg, hashes=hashes
            ):
                states.append(gen.get_state().numpy().copy())
                drive = sim.encode_images_poisson(
                    torch.from_numpy(images).float() / 255,
                    cfg["steps"],
                    cfg["dt_ms"],
                    cfg["input_rate_hz"],
                    generator=gen,
                )
                hashes.append(digest(drive.numpy()))
                return drive.to(device)

            timing = {}
            tick = time.perf_counter()
            rates, amplitude, mu, sd, _, _, error, i_counts, _ = calibrate_dataset(
                teacher,
                teacher_checkpoint,
                pixels,
                indices,
                cfg,
                out / "teacher/start",
                timing,
                drive_encoder=encode,
            )
            calibration_seconds = time.perf_counter() - tick
            dataset_path = out / "dataset.npz"
            np.savez_compressed(
                dataset_path,
                ids=indices,
                labels=labels[indices],
                features=pixels[indices],
                poisson_rate_hz=rates,
                amplitude=amplitude,
                test_rng_states=np.stack(states),
            )
            np.savez_compressed(
                out / "calibration.npz",
                target_mean=mu,
                target_sd=sd,
                max_relative_sd_error=error,
                teacher_i_spike_count=i_counts,
            )
            encoder = dataset_encoder(student, cfg)
            tick = time.perf_counter()
            with torch.no_grad():
                evaluated = sim.simulate(
                    bundle=student,
                    checkpoint=final_path,
                    input_bindings=[
                        dataset_binding(
                            dataset_path, encoder, "test", range(len(indices))
                        )
                    ],
                    batch_size=protocol["batch_size"],
                    seed=protocol["encoder_seed"],
                    device=device,
                    performance=performance(cfg),
                    snapshot=sim.SnapshotSpec(out / "student/start", stage="start"),
                    diagnostics=False,
                    retain_parameters=False,
                    retain_final_state=False,
                    retain_runtime_state=False,
                )
            scores = evaluated.outputs["scores"].detach().cpu().float().numpy()
            counts = evaluated.outputs["e_count"].detach().cpu().float().numpy()
            np.savez_compressed(
                out / "measurements.npz", scores=scores, e_counts=counts
            )
            summary = dict(
                label=label,
                target_hz=cfg["rate_target_hz"],
                training=case["training"],
                checkpoint_epoch=final.data_state["epoch"],
                samples=len(indices),
                correct=int((scores.argmax(-1) == labels[indices]).sum()),
                accuracy_pct=float(100 * (scores.argmax(-1) == labels[indices]).mean()),
                e_rate_hz=float(counts.mean() / (cfg["duration_ms"] / 1000)),
                sensory_sha256=hashes,
                calibration_seconds=calibration_seconds,
                evaluation_seconds=time.perf_counter() - tick,
                performance=evaluated.metrics["performance"],
            )
            cases.append(summary)
            snapshot_stages.update(
                {label + "/teacher": ["start"], label + "/student": ["start"]}
            )
            print(
                f"test {label}: {summary['accuracy_pct']:.2f}%, {summary['e_rate_hz']:.3f} Hz",
                flush=True,
            )
            del evaluated, final
        write_json(
            run.export / "test-results.json", dict(protocol=protocol, cases=cases)
        )
        run.details["scientific_artifacts"] = {"snapshot_stages": snapshot_stages}
        run.details["execution"].update(
            no_simulation=False, elapsed_seconds=time.perf_counter() - started
        )
    return run.run_id


def first_correct_position(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if (
        labels.ndim != 1
        or scores.ndim != 2
        or len(scores) != len(labels)
        or not np.isfinite(scores).all()
    ):
        raise ValueError("Invalid retained classification scores")
    correct = np.flatnonzero(scores.argmax(axis=1) == labels)
    if not len(correct):
        raise ValueError(
            "No correctly classified image in the retained validation batch"
        )
    return int(correct[0])


def replay_correct(identity, run_id=None):
    """Record the first previously correct validation image at selected weights."""
    started = time.perf_counter()
    training = source_run(
        ROOT / ".pingstore", identity, experiment=recipe.SLUG, stage="compute"
    )
    cfg = dict(training.details["execution"]["configuration"])
    if cfg.get("execution_backend") != "snnlab.sim.train":
        raise ValueError("Correct-image replay requires native training evidence")
    cfg.update(
        device=os.environ.get("SNNLAB_DEVICE", cfg["device"]),
        threads=int(os.environ.get("OMP_NUM_THREADS", cfg["threads"])),
    )
    torch.set_num_threads(cfg["threads"])
    with np.load(training.file("raster.npz")) as saved:
        position = first_correct_position(saved["batch_labels"], saved["batch_scores"])
        expected_predictions = saved["batch_scores"].argmax(axis=1)
    cfg["raster_selection"] = "first_correct_in_retained_validation_batch"
    cfg["raster_batch_position"] = position
    cfg["raster_source_code_sha256"] = {
        p.name: file_sha256(p) for p in Path(__file__).parent.glob("*.py")
    }
    with stage_run(
        ROOT,
        recipe.SLUG,
        "compute",
        inputs={"training": training},
        configuration=cfg,
        run_id=run_id,
        phase="raster",
    ) as run:
        checkpoint = sim.load_training_checkpoint(
            training.restore_bundle("best.checkpoint", run.scratch / "best.checkpoint")
        )
        record_selected(
            run,
            checkpoint.parameters,
            training.file("dataset.npz"),
            cfg,
            int(checkpoint.data_state["epoch"]),
            position=position,
            selection_policy=cfg["raster_selection"],
        )
        with np.load(run.export / "raster.npz") as replay:
            np.testing.assert_array_equal(
                replay["batch_scores"].argmax(axis=1), expected_predictions
            )
        run.details["scientific_artifacts"] = {
            "snapshot_stages": {"best-validation": ["start", "end"]}
        }
        run.details["execution"].update(
            no_simulation=False, elapsed_seconds=time.perf_counter() - started
        )
        training.check_unchanged()
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", action="store_true", help="record synchronized encoding timings"
    )
    parser.add_argument(
        "--test-inputs",
        help="evaluate final checkpoints from an explicit compact test-input run",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--source", help="explicit compact exp022 input run")
    parser.add_argument("--source-digest", help="pinned payload digest of --source")
    parser.add_argument("--seed", type=int, choices=recipe.SOURCE_SEEDS)
    parser.add_argument(
        "--rate-target-hz", choices=["off", "25", "10", "5", "2.5", "1"]
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run all six ceilings in isolated sequential processes",
    )
    parser.add_argument(
        "--calibration-source",
        help="Reuse calibration from an explicit compatible compute run",
    )
    parser.add_argument(
        "--raster-source",
        help="Replay the first correct retained validation image from this training run",
    )
    parser.add_argument("--epochs", type=int, help="Training epochs; default 10")
    parser.add_argument(
        "--train-images", type=int, help="Source-training images; default 6300"
    )
    parser.add_argument(
        "--compile-mode", choices=("auto", "eager", "default"), default="auto"
    )
    parser.add_argument(
        "--microbatch-size",
        type=int,
        help="Optional physical batch cap; default uses full batches",
    )
    args = parser.parse_args()
    if args.test_inputs:
        if (
            any(
                [
                    args.sweep,
                    args.raster_source,
                    args.source,
                    args.source_digest,
                    args.calibration_source,
                    args.profile,
                    args.seed,
                    args.rate_target_hz,
                    args.epochs,
                    args.train_images,
                    args.microbatch_size,
                ]
            )
            or args.compile_mode != "auto"
        ):
            parser.error("Test evaluation uses only its pinned input package")
        evaluate_test_inputs(args.test_inputs, args.run_id)
        sys.exit(0)
    if args.sweep:
        if (
            args.run_id
            or args.source
            or args.source_digest
            or args.profile
            or args.raster_source
            or args.calibration_source
            or args.rate_target_hz
        ):
            parser.error(
                "Sweep allocates one run per ceiling; single-run inputs are not allowed"
            )
        # Check all source pairs before launching any scientific execution.
        for cfg in recipe.sweep_configurations(seed=args.seed or 42):
            bank, _ = load_source(cfg)
            bank.check_unchanged()
        for target in recipe.RATE_TARGETS_HZ:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--rate-target-hz",
                "off" if target is None else f"{target:g}",
                "--seed",
                str(args.seed or 42),
                "--compile-mode",
                args.compile_mode,
            ]
            for option, value in (
                ("--epochs", args.epochs),
                ("--train-images", args.train_images),
                ("--microbatch-size", args.microbatch_size),
            ):
                if value is not None:
                    command.extend([option, str(value)])
            subprocess.run(command, check=True)
        sys.exit(0)
    if args.raster_source:
        if (
            args.epochs is not None
            or args.train_images is not None
            or args.microbatch_size is not None
            or args.compile_mode != "auto"
            or args.rate_target_hz is not None
            or args.seed is not None
            or args.calibration_source is not None
            or args.source is not None
            or args.source_digest is not None
            or args.profile
        ):
            parser.error("Raster replay uses the source execution configuration")
        replay_correct(args.raster_source, args.run_id)
    else:
        compute(
            args.run_id,
            epochs=args.epochs,
            train_images=args.train_images,
            compile_mode=None if args.compile_mode == "eager" else args.compile_mode,
            microbatch_size=args.microbatch_size,
            rate_target_hz=None
            if args.rate_target_hz in (None, "off")
            else float(args.rate_target_hz),
            seed=args.seed or 42,
            calibration_source=args.calibration_source,
            source=args.source,
            source_digest=args.source_digest,
            profile=args.profile,
        )
