"""Diagnostic-only GPU ablations of retained equation-matched computation."""

import ast
import contextlib
import cProfile
import functools
import hashlib
import inspect
import json
import random
import textwrap
import time
from pathlib import Path

import numpy as np
import torch
from experiments.exp126.compute import REPO
from experiments.exp126.compute_parts.comparison_compute import encode
from experiments.exp126.compute_parts.matched_models import make_model
from experiments.exp126.recipe import PROFILE_CONFIGURATION, SLUG
from pingstore.contracts import write_json_atomic
from pingstore.stages import source_run, stage_run
from snnlab.sim import execution as execution_module
from snnlab.sim import extensions as extension_module
from snnlab.sim.execution import GraphExecutor
from snntorch import surrogate
from torchvision.datasets import MNIST

VARIANTS = tuple(PROFILE_CONFIGURATION["variants"])


class DiagnosticTransform(ast.NodeTransformer):
    def __init__(self, finite=False, binary=False, instantaneous=False):
        self.finite, self.binary, self.instantaneous = finite, binary, instantaneous

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (
            self.finite
            and ast.unparse(node).startswith("torch.isfinite(")
            and ast.unparse(node).endswith(".all()")
        ):
            return ast.copy_location(ast.Constant(True), node)
        return node

    def visit_If(self, node):
        node = self.generic_visit(node)
        if (
            self.binary
            and "spike_values" in ast.unparse(node.test)
            and "spikes must be binary" in ast.unparse(node)
        ):
            node.test = ast.Constant(False)
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if (
            self.instantaneous
            and ast.unparse(node.value) == "conductance[key] * projection.decay + drive"
        ):
            node.value = ast.IfExp(
                test=ast.Compare(
                    ast.Attribute(
                        ast.Name("projection", ast.Load()), "decay", ast.Load()
                    ),
                    [ast.Eq()],
                    [ast.Constant(0)],
                ),
                body=ast.Name("drive", ast.Load()),
                orelse=node.value,
            )
        return node


@functools.lru_cache(maxsize=None)
def transform(function, **options):
    tree = DiagnosticTransform(**options).visit(
        ast.parse(textwrap.dedent(inspect.getsource(function)))
    )
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__)
    exec(compile(tree, function.__code__.co_filename, "exec"), namespace)
    return namespace[function.__name__]


@contextlib.contextmanager
def ablation(name):
    original_tensor, original_state, original_forward, original_spike = (
        extension_module.checked_tensor,
        extension_module.checked_state,
        GraphExecutor._forward,
        extension_module.spike,
    )
    try:
        if name not in ("snntorch", "snnlab"):
            extension_module.checked_tensor = transform(original_tensor, finite=True)
            extension_module.checked_state = transform(original_state, finite=True)
            # checked_state must resolve the diagnostic checked_tensor as well.
            extension_module.checked_state.__globals__["checked_tensor"] = (
                extension_module.checked_tensor
            )
        if name in ("no_finite_binary", "instantaneous", "cached_surrogate"):
            GraphExecutor._forward = transform(
                original_forward,
                binary=True,
                instantaneous=name in ("instantaneous", "cached_surrogate"),
            )
        if name == "cached_surrogate":
            spike = surrogate.fast_sigmoid(slope=25)
            extension_module.spike = lambda value, **kwargs: spike(value)
        yield
    finally:
        extension_module.checked_tensor, extension_module.checked_state = (
            original_tensor,
            original_state,
        )
        GraphExecutor._forward, extension_module.spike = (
            original_forward,
            original_spike,
        )


def cpu_profile(function):
    profiler = cProfile.Profile()
    profiler.runcall(function)
    rows = []
    for entry in profiler.getstats():
        if hasattr(entry.code, "co_name"):
            rows.append(
                dict(
                    function=entry.code.co_name,
                    file=entry.code.co_filename,
                    calls=entry.callcount,
                    self_seconds=entry.inlinetime,
                    total_seconds=entry.totaltime,
                )
            )
    return sorted(rows, key=lambda row: row["total_seconds"], reverse=True)[:35]


def compute(source_id, device_name, cpu_threads, run_id=None):
    from importlib.metadata import version

    if version("snnlab") != "0.9.0":
        raise ValueError(
            "The legacy ablation profiler requires snnlab 0.9.0; use the upgraded paired compute benchmark for current validation policies"
        )
    if device_name != "cuda" or not torch.cuda.is_available():
        raise ValueError("profiling requires CUDA")
    source = source_run(
        REPO / ".pingstore", source_id, stage="compute", experiment=SLUG
    )
    cfg = source.record["execution"]["configuration"]
    if cfg.get("schema") != "exp126.assembled/v1" or cfg["seeds"] != [42]:
        raise ValueError(
            "profile explicitly requires the completed seed-42 full comparison"
        )
    cfg = dict(cfg["protocol"], seed=42)
    torch.set_num_threads(cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    unit = source.unit("snntorch-seed-42")
    with np.load(unit / "initial_weights.npz") as values:
        initial = {key: torch.from_numpy(values[key].copy()) for key in values.files}
    with np.load(unit / "dataset.npz") as values:
        indices = values["validation_indices"][:128].copy()
    dataset = MNIST("/tmp/pinglab-mnist", train=True, download=True)
    images = dataset.data[indices].float().reshape(128, 784).cuda() / 255
    labels = dataset.targets[indices].cuda()
    events = encode(images, 642, cfg)
    lab_model = make_model("snnlab", cfg, initial, "cuda")
    models = {name: lab_model for name in VARIANTS}
    models["snntorch"] = make_model("snntorch", cfg, initial, "cuda")

    def operation(name, backward=False):
        model = models[name]
        model.zero_grad(set_to_none=True)
        with ablation(name):
            if backward:
                scores, _ = model(events)
                torch.nn.functional.cross_entropy(scores, labels).backward()
            else:
                with torch.inference_mode():
                    scores, _ = model(events)
        return scores

    configuration = dict(PROFILE_CONFIGURATION)
    with stage_run(
        REPO,
        SLUG,
        "compute",
        run_id=run_id,
        inputs={"comparison": source},
        configuration=configuration,
    ) as run:
        reference = operation("snntorch", True).detach().clone()
        reference_grad = {
            key: value.grad.detach().clone()
            for key, value in models["snntorch"].canonical().items()
        }
        agreement = {}
        for name in VARIANTS:
            scores = operation(name, True)
            gradients = models[name].canonical()
            agreement[name] = dict(
                score_max_abs=float((scores.detach() - reference).abs().max()),
                gradient_max_abs=max(
                    float((value.grad - reference_grad[key]).abs().max())
                    for key, value in gradients.items()
                ),
                gradients_close=all(
                    torch.allclose(
                        value.grad, reference_grad[key], atol=1e-5, rtol=1e-4
                    )
                    for key, value in gradients.items()
                ),
            )
            if (
                not torch.equal(scores, reference)
                or not agreement[name]["gradients_close"]
            ):
                raise ValueError(
                    f"diagnostic ablation changed initial results: {name}: {agreement[name]}"
                )
        rng = random.Random(20261014)
        timings = {name: {"forward": [], "forward_backward": []} for name in VARIANTS}
        for repeat in range(configuration["warmups"] + configuration["repeats"]):
            work = [(name, backward) for name in VARIANTS for backward in (False, True)]
            rng.shuffle(work)
            for name, backward in work:
                # AST transformation and patch setup are excluded from the measured interval.
                model = models[name]
                with ablation(name):
                    model.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    if backward:
                        scores, _ = model(events)
                        torch.nn.functional.cross_entropy(scores, labels).backward()
                    else:
                        with torch.inference_mode():
                            scores, _ = model(events)
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - started
                if repeat >= configuration["warmups"]:
                    timings[name]["forward_backward" if backward else "forward"].append(
                        seconds
                    )
        operator_profiles = {}
        for name in VARIANTS:
            operator_profiles[name] = {}
            for backward in (False, True):
                with ablation(name):
                    model = models[name]
                    model.zero_grad(set_to_none=True)
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as profiler:
                        if backward:
                            scores, _ = model(events)
                            torch.nn.functional.cross_entropy(scores, labels).backward()
                        else:
                            with torch.inference_mode():
                                scores, _ = model(events)
                        torch.cuda.synchronize()
                operator_profiles[name][
                    "forward_backward" if backward else "forward"
                ] = [
                    dict(
                        name=row.key,
                        calls=row.count,
                        self_cpu_us=row.self_cpu_time_total,
                        self_device_us=row.self_device_time_total,
                    )
                    for row in profiler.key_averages()
                ]
        cpu = {
            name: cpu_profile(lambda: operation(name))
            for name in ("snntorch", "snnlab", "no_finite_binary", "instantaneous")
        }
        write_json_atomic(
            run.export / "profile.json",
            dict(
                configuration=configuration,
                agreement=agreement,
                timings=timings,
                operator_profiles=operator_profiles,
                cpu_profiles=cpu,
            ),
        )
        import snntorch
        import torchvision

        run.record["execution"].update(
            device="cuda",
            device_name=torch.cuda.get_device_name(),
            cpu_threads=cpu_threads,
            versions=dict(
                torch=torch.__version__,
                snnlab=__import__("importlib.metadata", fromlist=["version"]).version(
                    "snnlab"
                ),
                snntorch=snntorch.__version__,
                torchvision=torchvision.__version__,
            ),
            library_source_sha256={
                str(Path(module.__file__).name): hashlib.sha256(
                    Path(module.__file__).read_bytes()
                ).hexdigest()
                for module in (execution_module, extension_module)
            },
            diagnostic_ablations="temporary monkeypatches only within diagnostic context; all originals restored; no production library files edited",
        )
    return run.run_id


def analyse(identity):
    source = source_run(REPO / ".pingstore", identity, stage="compute", experiment=SLUG)
    if source.record["execution"]["configuration"].get("schema") != "exp126.profile/v1":
        raise ValueError("not a diagnostic profile input")

    raw = json.loads(source.file("profile.json").read_text())
    medians = {}
    for name, operations in raw["timings"].items():
        medians[name] = {}
        for operation, values in operations.items():
            if (
                len(values) != raw["configuration"]["repeats"]
                or not np.isfinite(values).all()
                or min(values) <= 0
            ):
                raise ValueError("invalid timing repetitions")
            medians[name][operation] = dict(
                median_ms=float(np.median(values) * 1000),
                min_ms=min(values) * 1000,
                max_ms=max(values) * 1000,
            )
    effects = {}
    for operation in ("forward", "forward_backward"):
        reference = medians["snntorch"][operation]["median_ms"]
        original = medians["snnlab"][operation]["median_ms"]
        gap = original - reference
        effects[operation] = {
            name: dict(
                milliseconds_saved=original - row[operation]["median_ms"],
                original_gap_removed_fraction=(original - row[operation]["median_ms"])
                / gap,
                speedup_over_original=original / row[operation]["median_ms"],
            )
            for name, row in medians.items()
            if name not in ("snntorch", "snnlab")
        }
    synchronization = {
        name: {
            operation: {
                row["name"]: row["calls"]
                for row in rows
                if row["name"]
                in (
                    "aten::_local_scalar_dense",
                    "aten::isfinite",
                    "aten::any",
                    "cudaStreamSynchronize",
                    "cudaDeviceSynchronize",
                    "aten::item",
                )
            }
            for operation, rows in operations.items()
        }
        for name, operations in raw["operator_profiles"].items()
    }
    if any(
        row["score_max_abs"] != 0 or not row["gradients_close"]
        for row in raw["agreement"].values()
    ):
        raise ValueError("an ablation changed the checked computation")
    result = dict(
        schema="exp126.profile-analysis/v1",
        profile_configuration=raw["configuration"],
        medians=medians,
        effects=effects,
        synchronization=synchronization,
        agreement=raw["agreement"],
        limitations=[
            "one L4 allocation, one initial model and batch of 128",
            "diagnostic checks disabled only for causal ablation, not a safe production change",
            "ablations cumulative and interaction-dependent, not additive independent cost estimates",
            "timing repetitions unprofiled; operator and Python profiles measured separately",
            "no full training or test-inference rerun, no claimed optimization of the production library",
        ],
    )
    with stage_run(
        REPO,
        SLUG,
        "analyse",
        inputs={"profile": source},
        configuration={"schema": "exp126.profile-analysis/v1"},
    ) as run:
        write_json_atomic(run.export / "results.json", result)
    return run.run_id


def present(identity):
    source = source_run(REPO / ".pingstore", identity, stage="analyse", experiment=SLUG)
    if (
        source.record["execution"]["configuration"].get("schema")
        != "exp126.profile-analysis/v1"
    ):
        raise ValueError("not a profile analysis")
    from pingstore.contracts import load_json

    result = load_json(source.file("results.json"))
    profile_pin = source.record["inputs"]["profile"]
    profile_source = source_run(
        REPO / ".pingstore",
        profile_pin["run_id"],
        stage="compute",
        experiment=SLUG,
        reference=profile_pin,
    )
    profile_configuration = profile_source.record["execution"]["configuration"]
    result["profile_configuration"] = profile_configuration
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/pinglab-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import csv

    import matplotlib.pyplot as plt
    from experiments.helpers.figsave import save_figure

    labels = [
        "snnTorch",
        "snnlab",
        "No finite checks",
        "+ no binary checks",
        "+ skip zero decay",
        "+ cached surrogate",
    ]
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "svg.fonttype": "none"})
    with stage_run(
        REPO,
        SLUG,
        "present",
        inputs={"analyse": source},
        configuration={"schema": "exp126.profile-presentation/v1"},
    ) as run:
        fig, axes = plt.subplots(
            1, 2, figsize=(180 / 25.4, 100 / 25.4), constrained_layout=True
        )
        rows = []
        for axis, operation, title in zip(
            axes,
            ("forward", "forward_backward"),
            ("A  Forward inference", "B  Forward, loss and backward"),
            strict=True,
        ):
            values = [result["medians"][name][operation] for name in VARIANTS]
            medians = [row["median_ms"] for row in values]
            axis.barh(range(6), medians, color=["#202020"] + ["#a34d45"] * 5, alpha=0.8)
            axis.errorbar(
                medians,
                range(6),
                xerr=[
                    [row["median_ms"] - row["min_ms"] for row in values],
                    [row["max_ms"] - row["median_ms"] for row in values],
                ],
                fmt="none",
                ecolor="#333333",
                capsize=2,
            )
            axis.axvline(medians[0], color="#202020", linestyle="--", linewidth=0.8)
            axis.set(
                yticks=range(6),
                yticklabels=labels,
                xlabel="Synchronized wall time (ms)",
                title=title,
            )
            axis.invert_yaxis()
            axis.spines[["top", "right"]].set_visible(False)
            for name, row in zip(VARIANTS, values, strict=True):
                rows.append(dict(variant=name, operation=operation, **row))
        save_figure(fig, run.export / "profiling_ablations")
        plt.close(fig)
        with (run.export / "profiling_table.csv").open("w") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["variant", "operation", "median_ms", "min_ms", "max_ms"],
            )
            writer.writeheader()
            writer.writerows(rows)
        result["figure_description"] = (
            f"Same initial network and encoded batch of 128, 50 steps, one L4 GPU, four CPU threads; bars are medians of {profile_configuration['repeats']} unprofiled repetitions after {profile_configuration['warmups']} warmups; whiskers min-max, not confidence intervals; dashed line snnTorch median; ablations cumulative and diagnostic only."
        )
        write_json_atomic(run.export / "numbers.json", result)
    return run.run_id
