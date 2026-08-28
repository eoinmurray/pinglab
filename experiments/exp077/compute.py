from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools"), str(REPO / "tools/snnsim")]

import time
import tracemalloc

import config
import models as M  # noqa: TID251 - explicit legacy-versus-graph executor parity gate
import numpy as np
import torch
from execution import ExecutionSpec, GraphExecutor, build, simulate
from experiments.exp077 import recipe
from experiments.helpers import snnlang_stages as stages
from pingstore.contracts import PingstoreError, write_json_atomic
from snnlang.examples.build_examples import ping_classifier


def acquire_parity(directory: Path) -> dict:
    graph = ping_classifier().graph
    M.N_IN, M.N_OUT = recipe.PARITY_MODEL["n_input"], recipe.PARITY_MODEL["n_classes"]
    config.set_sim_dt(recipe.PARITY["dt_ms"], recipe.PARITY["t_ms"])
    M.T_steps = recipe.PARITY["steps"]
    torch.manual_seed(recipe.PARITY["seed"])
    legacy = config.build_net(
        "ping",
        **{
            key: value
            for key, value in recipe.PARITY_MODEL.items()
            if key not in {"n_input", "n_classes"}
        },
    )
    native = build(
        ExecutionSpec(
            kind="build", executor="graph", graph=graph, seed=recipe.PARITY["seed"]
        )
    ).model
    assert isinstance(native, GraphExecutor)
    x = (
        torch.rand(
            recipe.PARITY["steps"],
            recipe.PARITY["batch"],
            recipe.PARITY_MODEL["n_input"],
        )
        < recipe.PARITY["input_probability"]
    ).float()
    legacy.recording = True
    legacy_out = legacy(input_spikes=x)
    native_out = native({"image": x}, record=True)
    parameter_map = {
        "sensory_ping_input.weight": legacy.W_ff[0],
        "classifier_projection.weight": legacy.W_ff[1],
        "sensory_ping_E_to_E.weight": legacy.W_ee["1"],
        "sensory_ping_E_to_I.weight": legacy.W_ei["1"],
        "sensory_ping_I_to_E.weight": legacy.W_ie["1"],
        "sensory_ping_I_to_I.weight": legacy.W_ii["1"],
    }
    tensors = {
        "input": x,
        "legacy_output": legacy_out,
        "native_output": native_out.outputs["class_logits"],
        "legacy_e": legacy.spike_record["hid"],
        "legacy_i": legacy.spike_record["inh"],
        "native_e": native_out.recordings["cell_0"],
        "native_i": native_out.recordings["cell_1"],
    }
    for key, value in parameter_map.items():
        tensors["legacy_parameter__" + key] = value
        tensors["native_parameter__" + key] = native.parameter_map()[key]
    checkpoint = {k: v.detach().clone() for k, v in native.state_dict().items()}
    replay = build(
        ExecutionSpec(
            kind="build",
            executor="graph",
            graph=graph,
            seed=recipe.PARITY["replay_seed"],
        )
    ).model
    assert isinstance(replay, GraphExecutor)
    replay.load_state_dict(checkpoint)
    replay_out = replay({"image": x}, record=False).outputs["class_logits"]
    tensors["replay_output"] = replay_out
    torch.save(checkpoint, directory / "checkpoint.pth")

    legacy.recording = False
    for _ in range(recipe.PARITY["warmups"]):
        legacy(input_spikes=x)
        native({"image": x}, record=False)
    timings = {"legacy": [], "graph": []}
    peaks = {}
    for _ in range(recipe.PARITY["repetitions"]):
        t = time.perf_counter()
        legacy(input_spikes=x)
        timings["legacy"].append(time.perf_counter() - t)
        t = time.perf_counter()
        native({"image": x}, record=False)
        timings["graph"].append(time.perf_counter() - t)
    for name, call in (
        ("legacy", lambda: legacy(input_spikes=x)),
        ("graph", lambda: native({"image": x}, record=False)),
    ):
        tracemalloc.start()
        call()
        _, peaks[name] = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    class OutputOnly(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, values):
            return self.model({"image": values}, record=False).outputs["class_logits"]

    compile_started = time.perf_counter()
    compiled = torch.compile(OutputOnly(native), dynamic=False)
    compile_setup_s = time.perf_counter() - compile_started
    compile_x = x[: recipe.PARITY["compile_steps"], : recipe.PARITY["compile_batch"]]
    first_started = time.perf_counter()
    compiled_first = compiled(compile_x)
    compile_first_s = time.perf_counter() - first_started
    compiled_times = []
    for _ in range(recipe.PARITY["compiled_repetitions"]):
        t = time.perf_counter()
        compiled_warm = compiled(compile_x)
        compiled_times.append(time.perf_counter() - t)
    tensors["compiled_first"] = compiled_first
    tensors["compiled_warm"] = compiled_warm
    np.savez_compressed(
        directory / "parity.npz",
        **{
            key: tensor.detach().cpu().numpy().copy() for key, tensor in tensors.items()
        },
    )
    return {
        "timings": timings,
        "peaks": peaks,
        "compiled_times": compiled_times,
        "compile_backend": "torch.compile Inductor on CPU",
        "compile_workload_steps": int(compile_x.shape[0]),
        "compile_workload_batch": int(compile_x.shape[1]),
        "compile_setup_s": compile_setup_s,
        "compile_first_s": compile_first_s,
    }


def compute(*, run_id=None):
    with stages.execution(REPO, recipe, "compute", run_id=run_id) as run:
        drives = recipe.independent_inputs()
        variants_dir = run.export / "variants"
        variants_dir.mkdir()
        np.savez_compressed(
            run.export / "inputs.npz",
            **{key: value.numpy() for key, value in drives.items()},
        )
        for name, delay_steps in recipe.VARIANTS:
            bundle = recipe.author_variant(name, delay_steps)
            bundle.write(variants_dir / f"{name}.bundle", visualise=False)
            result = simulate(
                ExecutionSpec(
                    kind="simulate",
                    executor="graph",
                    graph=bundle.graph,
                    inputs=drives,
                    seed=recipe.SEED,
                )
            )
            np.savez_compressed(
                variants_dir / f"{name}-recordings.npz",
                **{
                    key: value.detach().cpu().numpy()
                    for key, value in result.recordings.items()
                },
            )
        parity_dir = run.export / "parity"
        parity_dir.mkdir()
        write_json_atomic(parity_dir / "acquisition.json", acquire_parity(parity_dir))
        write_json_atomic(
            run.export / "delay-gates.json",
            stages.test_evidence(REPO, run, "delay-gates", recipe.DELAY_TESTS),
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="unused source-neutral reservation")
    args = parser.parse_args()
    try:
        compute(run_id=args.run_id)
    except (PingstoreError, OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
