"""Current production timing diagnosis; temporary finite-input ablations only."""

import ast
import contextlib
import inspect
import random
import textwrap
import time
from importlib.metadata import version

import numpy as np
import torch
from experiments.exp126.compute import REPO
from experiments.exp126.compute_parts.comparison_compute import (
    encode,
    optimizer,
    restore,
    step,
)
from experiments.exp126.compute_parts.matched_models import make_model
from experiments.exp126.compute_parts.profiling import DiagnosticTransform, cpu_profile
from experiments.exp126.recipe import CURRENT_PROFILE_CONFIGURATION, SLUG
from pingstore.contracts import write_json_atomic
from pingstore.stages import source_run, stage_run
from snnlab.sim.execution import GraphExecutor
from torchvision.datasets import MNIST


class TrimSnapshots(DiagnosticTransform):
    def visit_Assign(self, node):
        node = super().visit_Assign(node)
        if any(
            isinstance(t, ast.Name) and t.id == "next_runtime_state"
            for t in node.targets
        ):
            node.value = ast.Constant(None)
        return node

    def visit_Call(self, node):
        node = super().visit_Call(node)
        if isinstance(node.func, ast.Name) and node.func.id == "ExecutionResult":
            for keyword in node.keywords:
                if keyword.arg in ("parameters", "final_state"):
                    keyword.value = ast.Dict([], [])
        return node


def patched(original, zero, snapshots):
    transformer = (
        TrimSnapshots(instantaneous=zero)
        if snapshots
        else DiagnosticTransform(instantaneous=zero)
    )
    tree = transformer.visit(ast.parse(textwrap.dedent(inspect.getsource(original))))
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree, original.__code__.co_filename, "exec"), namespace)
    return namespace[original.__name__]


def compute(source_id, device_name, cpu_threads, run_id=None):
    if version("snnlab") != "0.10.1" or device_name != "cuda":
        raise ValueError("This diagnosis requires snnlab 0.10.1 and CUDA")
    source = source_run(
        REPO / ".pingstore", source_id, stage="compute", experiment=SLUG
    )
    cfg = dict(source.record["execution"]["configuration"]["protocol"], seed=42)
    torch.set_num_threads(cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    unit = source.unit("snntorch-seed-42")

    def weights(role):
        with np.load(unit / role) as data:
            return {k: torch.from_numpy(data[k].copy()) for k in data.files}

    states = {
        "initial": weights("initial_weights.npz"),
        "trained": weights("final_weights.npz"),
    }
    with np.load(unit / "dataset.npz") as data:
        indices = data["validation_indices"][:128].copy()
    data = MNIST("/tmp/pinglab-mnist", train=True, download=True)
    labels = data.targets[indices].cuda()
    events = encode(data.data[indices].float().reshape(128, 784).cuda() / 255, 642, cfg)
    models = {
        name: make_model(name, cfg, states["initial"], "cuda")
        for name in ("snntorch", "snnlab")
    }
    original = GraphExecutor._forward
    variants = {
        "snntorch": original,
        "snnlab": original,
        "zero_decay": patched(original, True, False),
        "no_snapshots": patched(original, False, True),
        "combined": patched(original, True, True),
    }

    @contextlib.contextmanager
    def variant(name):
        GraphExecutor._forward = variants[name]
        try:
            yield models["snntorch" if name == "snntorch" else "snnlab"]
        finally:
            GraphExecutor._forward = original

    agreement = {}
    timings = {}
    profiles = {}
    rng = random.Random(20261014)
    with stage_run(
        REPO,
        SLUG,
        "compute",
        run_id=run_id,
        inputs={"comparison": source},
        configuration=CURRENT_PROFILE_CONFIGURATION,
    ) as run:
        for state, values in states.items():
            timings[state] = {
                name: {op: [] for op in ("forward", "forward_backward", "steady_step")}
                for name in variants
            }
            agreement[state] = {}
            reference = None
            grads = None
            for name in variants:
                with variant(name) as model:
                    restore(model, values)
                    model.zero_grad(set_to_none=True)
                    scores, _ = model(events)
                    torch.nn.functional.cross_entropy(scores, labels).backward()
                    if name == "snntorch":
                        reference = scores.detach().clone()
                        grads = {
                            k: v.grad.detach().clone()
                            for k, v in model.canonical().items()
                        }
                    agreement[state][name] = dict(
                        score_max_abs=float((scores - reference).abs().max().detach()),
                        gradient_max_abs=max(
                            float((v.grad - grads[k]).abs().max())
                            for k, v in model.canonical().items()
                        ),
                        gradients_close=all(
                            torch.allclose(v.grad, grads[k], atol=1e-5, rtol=1e-4)
                            for k, v in model.canonical().items()
                        ),
                    )
                    if (
                        agreement[state][name]["score_max_abs"] != 0
                        or not agreement[state][name]["gradients_close"]
                    ):
                        raise ValueError(f"Changed computation: {state}/{name}")
            opts = {}
            for name in variants:
                with variant(name) as model:
                    restore(model, values)
                    opts[name] = optimizer(model, cfg)
                    step(model, events, labels, opts[name], cfg)

            def operation(name, op):
                with variant(name) as model:
                    if op == "steady_step":
                        step(model, events, labels, opts[name], cfg)
                    else:
                        model.zero_grad(set_to_none=True)
                        if op == "forward":
                            with torch.inference_mode():
                                model(events)
                        else:
                            scores, _ = model(events)
                            torch.nn.functional.cross_entropy(scores, labels).backward()

            for repeat in range(
                CURRENT_PROFILE_CONFIGURATION["warmups"]
                + CURRENT_PROFILE_CONFIGURATION["repeats"]
            ):
                work = [(n, o) for n in variants for o in timings[state][n]]
                rng.shuffle(work)
                for name, op in work:
                    restore(
                        models["snntorch" if name == "snntorch" else "snnlab"], values
                    )
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    operation(name, op)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    if repeat >= CURRENT_PROFILE_CONFIGURATION["warmups"]:
                        timings[state][name][op].append(elapsed)
            profiles[state] = {}
            for name in variants:
                restore(models["snntorch" if name == "snntorch" else "snnlab"], values)
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    operation(name, "steady_step")
                    torch.cuda.synchronize()
                profiles[state][name] = {
                    "operators": [
                        dict(
                            name=r.key,
                            calls=r.count,
                            self_cpu_us=r.self_cpu_time_total,
                            self_device_us=r.self_device_time_total,
                        )
                        for r in prof.key_averages()
                    ],
                    "python": cpu_profile(lambda: operation(name, "steady_step")),
                }
        write_json_atomic(run.export / "agreement.json", agreement)
        run.record["execution"].update(
            timings=timings,
            profiles=profiles,
            versions={
                "snnlab": version("snnlab"),
                "torch": torch.__version__,
                "snntorch": version("snntorch"),
            },
            device_name=torch.cuda.get_device_name(),
            cpu_threads=cpu_threads,
            scope="Single shared GPU, randomized timings, batch128; warmed Adam state, parameters restored outside each timed operation. Snapshot ablation discards continuation only; never a production change.",
        )
    return run.run_id


def analyse(identity):
    import json

    source = source_run(REPO / ".pingstore", identity, stage="compute", experiment=SLUG)
    execution = source.record["execution"]
    if execution["configuration"]["schema"] != "exp126.current-profile/v1":
        raise ValueError("wrong diagnostic schema")
    agreement = json.loads(source.file("agreement.json").read_text())
    medians = {}
    for state, variants in execution["timings"].items():
        medians[state] = {}
        for name, operations in variants.items():
            medians[state][name] = {}
            for operation, values in operations.items():
                if (
                    len(values) != 11
                    or not np.isfinite(values).all()
                    or min(values) <= 0
                ):
                    raise ValueError("invalid repetitions")
                medians[state][name][operation] = {
                    "median_ms": float(np.median(values) * 1000),
                    "min_ms": min(values) * 1000,
                    "max_ms": max(values) * 1000,
                }
    effects = {
        state: {
            name: {
                op: operations[op]["median_ms"] - variants[name][op]["median_ms"]
                for op in operations
            }
            for name in variants
        }
        for state, variants in medians.items()
        for operations in [variants["snnlab"]]
    }
    with stage_run(
        REPO,
        SLUG,
        "analyse",
        inputs={"profile": source},
        configuration={"schema": "exp126.current-profile-analysis/v1"},
    ) as run:
        write_json_atomic(
            run.export / "results.json",
            {
                "medians": medians,
                "saved_ms": effects,
                "agreement": agreement,
                "limitations": [
                    "One shared L4 GPU, one batch128, initial and snnTorch final weights shared between both implementations.",
                    "Snapshots removed only where caller discards them; zero-decay specialization checked for finite inputs only.",
                    "Warmed Adam moments evolve between repetitions; fixed weights restored outside timers.",
                    "No full training rerun; independent-host production timing cannot be attributed entirely from this probe.",
                ],
            },
        )
        run.record["execution"]["operator_profiles"] = execution["profiles"]
    print(run.run_id)
    return run.run_id


def present(identity):
    import csv
    import json

    import matplotlib.pyplot as plt
    from experiments.helpers.figsave import save_figure

    source = source_run(REPO / ".pingstore", identity, stage="analyse", experiment=SLUG)
    result = json.loads(source.file("results.json").read_text())
    if (
        source.record["execution"]["configuration"]["schema"]
        != "exp126.current-profile-analysis/v1"
    ):
        raise ValueError("wrong schema")
    with stage_run(
        REPO,
        SLUG,
        "present",
        inputs={"analysis": source},
        configuration={"schema": "exp126.current-profile-presentation/v1"},
    ) as run:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        names = ["snntorch", "snnlab", "zero_decay", "no_snapshots", "combined"]
        for axis, (state, variants) in zip(axes, result["medians"].items()):
            for i, op in enumerate(("forward", "forward_backward", "steady_step")):
                axis.bar(
                    np.arange(len(names)) + (i - 1) * 0.25,
                    [variants[n][op]["median_ms"] for n in names],
                    width=0.25,
                    label=op.replace("_", " "),
                )
            axis.set(
                xticks=np.arange(len(names)),
                xticklabels=[
                    "snnTorch",
                    "snnlab",
                    "Zero decay",
                    "No snapshots",
                    "Combined",
                ],
                ylabel="Median time (ms)",
                title=f"{state.capitalize()} weights: shared L4, batch 128",
            )
            axis.tick_params(axis="x", rotation=25)
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(fontsize=7, frameon=False)
        save_figure(fig, run.export / "remaining_timing")
        plt.close(fig)
        with (run.export / "timing_table.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["weights", "variant", "operation", "median_ms", "min_ms", "max_ms"]
            )
            for state, variants in result["medians"].items():
                for name, operations in variants.items():
                    for op, row in operations.items():
                        writer.writerow(
                            [
                                state,
                                name,
                                op,
                                row["median_ms"],
                                row["min_ms"],
                                row["max_ms"],
                            ]
                        )
        write_json_atomic(run.export / "numbers.json", result)
    print(run.run_id)
    return run.run_id
