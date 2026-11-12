"""Replay retained sensory encoding and prepare frozen private inhibition."""

import hashlib

import numpy as np
import torch
from snnlab import sim


def sensory_batch(arrays, selected, cfg):
    selected = np.asarray(selected)
    batch = cfg["manuscript_batch_size"]
    parts = []
    for group in np.unique(selected // batch):
        start = int(group) * batch
        stop = min(start + batch, len(arrays["features"]))
        generator = torch.Generator().set_state(
            torch.from_numpy(arrays["test_rng_states"][group].copy())
        )
        full = sim.encode_images_poisson(
            torch.from_numpy(arrays["features"][start:stop].copy()).float() / 255,
            cfg["steps"],
            cfg["dt_ms"],
            cfg["input_rate_hz"],
            generator=generator,
        )
        parts.append((selected[(selected // batch) == group], full))
    output = torch.empty((cfg["steps"], len(selected), cfg["n_input"]))
    for indices, full in parts:
        for index in indices:
            output[:, int(np.flatnonzero(selected == index)[0])] = full[
                :, int(index % batch)
            ]
    return output


def private_batch(arrays, selected, cfg):
    """Use original seed derivation on CPU; GPU source event draws are not bit-identical."""
    history, steps = cfg["inhibition_prehistory_steps"], cfg["steps"]
    values, initial, pending, counts = [], [], [], []
    ages = torch.arange(history - 2, -1, -1, dtype=torch.float64)
    decay = torch.exp(-ages * cfg["dt_ms"] / cfg["tau_gaba_ms"])
    for row in selected:
        index = int(arrays["ids"][row])
        seed = int(
            np.random.SeedSequence([cfg["inhibition_seed"], 0, index]).generate_state(
                1, dtype=np.uint64
            )[0]
        ) % (2**63 - 1)
        generator = torch.Generator().manual_seed(seed)
        rates = torch.from_numpy(arrays["poisson_rate_hz"][row].copy()).double()
        amplitude = torch.from_numpy(arrays["amplitude"][row].copy()).float()
        raw = torch.poisson(
            (rates * cfg["dt_ms"] / 1000).expand(steps + history, -1),
            generator=generator,
        )
        values.append(raw[history:].float() * amplitude)
        initial.append(
            ((raw[: history - 1] * decay[:, None]).sum(0) * amplitude.double()).float()
        )
        pending.append(raw[history - 1].float() * amplitude)
        counts.append(raw[history:].sum(-1).to(torch.int32).numpy())
    return (
        torch.stack(values, 1),
        torch.stack(initial),
        torch.stack(pending)[None],
        np.stack(counts, 1),
    )


def encode(arrays, selected, *, dt_ms, channels, config, seed, context, device="cpu"):
    cfg = config["recipe"]
    drive = sensory_batch(arrays, selected, cfg).to(device)
    if config["kind"] != "artificial":
        return sim.PreparedBatch({"drive": drive})
    events, conductance, pending, _ = private_batch(arrays, selected, cfg)
    return sim.PreparedBatch(
        {"drive": drive, "artificial": events.to(device)},
        sim.PresentationState(
            signature=config["signature"],
            conductances={"artificial_to_E": conductance.to(device)},
            input_histories={"artificial": pending.to(device)},
            provenance={
                "history_steps": cfg["inhibition_prehistory_steps"],
                "policy": "private_gaba_only",
            },
        ),
    )


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
