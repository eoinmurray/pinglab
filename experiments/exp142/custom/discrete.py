"""Replay frozen COBA-LIF weights on a native SNNLab simulation clock.

This adapter builds the native model and injects validated source matrices before
its forward execution. The public simulation entry point cannot accept a raw
parameter mapping, so the shared model_execution context owns native snapshots.
Input coordinates are audited before execution and weights checked afterward."""

from pathlib import Path

import numpy as np
import torch
from experiments.exp142.custom.inputs import event_digest
from experiments.exp142.custom.network_artifacts import model_execution
from experiments.exp142.recipe import author_network
from snnlab.sim import (
    EventStreamBinding,
    Performance,
    build,
    resolve_event_stream_bindings,
)


def execute(events, weights, mode, cfg, dt_ns, directory, *, matched, diagram=False):
    duration = cfg["duration_ns"]
    sample = cfg["sample_dt_ns"] if matched else cfg["native_sample_dt_ns"]
    if dt_ns <= 0 or sample % dt_ns or duration % sample:
        raise ValueError("nondivisible recording clock")
    times, channels = events["time_ns"], events["channels"]
    if (
        np.any(times % dt_ns)
        or np.any(times < 0)
        or np.any(times >= duration)
        or np.any(events["batches"] != 0)
        or np.any(channels < 0)
        or np.any(channels >= cfg["n_input"])
    ):
        raise ValueError("unrepresentable input schedule")
    bundle = author_network(mode, cfg, dt_ns, matched=matched)
    built = build(
        bundle=bundle,
        device="cpu",
        seed=cfg["seed"],
        performance=Performance(**cfg["performance"]),
    )
    model = built.model
    native = model.parameter_map()
    if set(native) != set(weights):
        raise ValueError("source parameter inventory mismatch")
    with torch.no_grad():
        for name, value in weights.items():
            if tuple(native[name].shape) != value.shape:
                raise ValueError("source weight shape mismatch")
            native[name].copy_(torch.from_numpy(value))
            native[name].requires_grad_(False)
    model.eval()
    binding = EventStreamBinding(
        "drive",
        steps=torch.from_numpy(times // dt_ns),
        batches=torch.from_numpy(events["batches"]),
        channels=torch.from_numpy(channels),
        steps_count=duration // dt_ns,
        batch_size=1,
        source={"kind": "saved_exp141_sample", "time_unit": "ns"},
    )
    resolved = resolve_event_stream_bindings(
        bundle.graph, bindings=(binding,), device="cpu"
    )
    actual = np.column_stack(np.nonzero(resolved.tensors["drive"].numpy())).astype(
        np.int64
    )
    actual[:, 0] *= dt_ns
    np.testing.assert_array_equal(
        actual, np.column_stack((times, events["batches"], channels))
    )
    directory = Path(directory)
    with (
        torch.inference_mode(),
        model_execution(
            model,
            bundle,
            directory,
            seed=cfg["seed"],
            protocol=resolved.protocol,
            settings={
                "dt_ns": dt_ns,
                "matched_physical_arrivals": matched,
                "frozen": True,
            },
            snapshot_stages=("start",),
        ),
    ):
        result = model(resolved.tensors, diagnostics=True)
        for name, expected in weights.items():
            np.testing.assert_array_equal(native[name].detach().numpy(), expected)
    if diagram:
        bundle.visualise(directory / "circuit.svg", view="circuit")
    raw = {key: value[:, 0].cpu().numpy() for key, value in result.diagnostics.items()}
    populations = ["E"] + (["I"] if mode == "ping" else []) + ["readout"]
    starts = np.arange(0, duration // dt_ns, sample // dt_ns)
    ends = starts + sample // dt_ns - 1
    spikes = np.concatenate([raw[f"{name}_spikes"] for name in populations], axis=1)
    steps, neurons = np.nonzero(spikes)
    ni = cfg["n_i"] if mode == "ping" else 0
    ge = [raw["input_to_E_conductance"]]
    if ni:
        ge.append(raw["E_to_I_conductance"])
    ge.append(raw["readout_projection_conductance"])
    gi = np.zeros_like(np.concatenate(ge, axis=1))
    if ni:
        gi[:, : cfg["n_e"]] = raw["I_to_E_conductance"]
    output_steps, output_neurons = np.nonzero(raw["readout_spikes"])
    scores = result.outputs["class_scores"][0].cpu().numpy()
    np.testing.assert_array_equal(
        scores, np.bincount(output_neurons, minlength=cfg["n_output"])
    )
    recording = dict(
        g_time_ns=starts * dt_ns,
        v_time_ns=(ends + 1) * dt_ns,
        g_exc_ns=np.concatenate(ge, axis=1)[starts] * 1000,
        g_inh_ns=gi[starts] * 1000,
        voltage_mv=np.concatenate(
            [raw[f"{name}_voltage"] for name in populations], axis=1
        )[ends],
        spike_time_ns=(steps + 1) * dt_ns,
        spike_neurons=neurons.astype(np.int64),
        output_spike_time_ns=(output_steps + 1) * dt_ns,
        output_spike_neurons=output_neurons.astype(np.int64),
        readout_voltage=raw["readout_voltage"][ends],
        readout_scores=np.cumsum(raw["readout_spikes"], axis=0)[ends],
        class_scores=scores,
    )
    return recording, dict(
        backend="snnlab CPU",
        build_metrics=built.metrics,
        execution_metrics=result.metrics,
        input_events=len(times),
        input_equivalent=True,
        weights_unchanged=True,
        dt_ns=dt_ns,
        sample_dt_ns=sample,
        matched_physical_arrivals=matched,
        input_event_sha256=event_digest(
            dict(zip(("time_ns", "batches", "channels"), actual.T, strict=True))
        ),
        max_input_time_error_ns=0,
        projection_delay_steps={
            row.id: row.delay_steps for row in model.plan.projections
        },
        refractory_timing=bundle.graph["refractory_timing"],
    )
