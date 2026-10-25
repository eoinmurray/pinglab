"""Integrate an explicit delayed closure from validated exp133 evidence.

One uncoupled control and one coupling-onset trajectory share initial history,
inputs and closure settings. Each invocation creates an immutable Pingstore run;
analysis and presentation are invoked separately.
"""

import argparse
import os
import shutil
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import numpy as np
from experiments.exp153 import recipe
from hpc.concurrent_compute import run_active_workers
from pingstore.contracts import file_sha256, write_json_atomic
from pingstore.stages import source_run, stage_run
from snnlab import lang
from snnlab import meanfield as mf
from snnlab.sim import NetworkSnapshot


def ancestry(repo, identity, *, reference=None):
    """Validate every explicitly pinned ancestor, including source parameters."""
    found, visiting = {}, set()

    def visit(name, pin):
        if name in visiting:
            raise ValueError("cyclic source lineage")
        if name in found:
            if pin is not None and found[name].reference != pin:
                raise ValueError("conflicting source pins")
            return
        item = source_run(repo / ".pingstore", name, reference=pin)
        visiting.add(name)
        for upstream in item.record["inputs"].values():
            visit(upstream["run_id"], upstream)
        visiting.remove(name)
        found[name] = item

    visit(identity, reference)
    return found


def imported_weights(source, bundle):
    """Use native effective matrices: mean incoming sums, never initializers."""
    snapshot = NetworkSnapshot.read(source.file(recipe.SNAPSHOT, "snapshot.json"))
    if (
        snapshot.graph_identity != bundle.manifest["graph_digest"]
        or snapshot.stage != "start"
    ):
        raise ValueError("native parameter snapshot differs from source graph or stage")
    matrices = snapshot.load_parameters()
    targets = {p["id"]: p["size"] for p in bundle.graph["populations"]}
    weights = {}
    for projection in bundle.graph["projections"]:
        name = projection["id"]
        matrix = matrices[name + ".weight"].numpy()
        count = targets[projection["target"].rsplit(".", 1)[0]]
        if (
            matrix.ndim != 2
            or matrix.shape[1] != count
            or not np.isfinite(matrix).all()
        ):
            raise ValueError("invalid native source-target matrix")
        weights[name] = float(matrix.sum(dtype=np.float64) / count)
    return weights


def integrate(model, cfg):
    initial = recipe.initial_state(model, cfg)
    solver = cfg["solver"]
    return mf.integrate(
        model,
        duration=cfg["duration_ms"],
        initial=initial,
        history=mf.ConstantHistory(initial),
        sample_interval=cfg["sample_interval_ms"],
        method=solver["method"],
        rtol=solver["rtol"],
        atol=solver["atol"],
        max_step=solver["max_step_ms"],
    )


def run_conditions(directory, item, worker_index, cfg):
    """Shared serial/parallel worker; each condition has disjoint exports."""
    import torch

    torch.set_num_threads(1)
    export = directory / "export"
    bundle = lang.load_bundle(export / "network.bundle")
    for condition in item["conditions"]:
        model = recipe.build_model(
            bundle,
            cfg,
            difference_hz=condition["difference_hz"],
            coupling_us=condition["coupling_us"],
        )
        label = condition["label"]
        model.write(export / f"{label}-model.json")
        integrate(model, cfg).write(export / f"{label}-trajectory.json")
        print(
            f"Completed {label}: drive difference {condition['difference_hz']:g} Hz, coupling {condition['coupling_us']:g} µS",
            flush=True,
        )
    return len(item["conditions"])


def condition_inventory(directory, item, worker_index):
    export = directory / "export"
    return {
        f"{condition['label']}-{kind}.json": file_sha256(
            export / f"{condition['label']}-{kind}.json"
        )
        for condition in item["conditions"]
        for kind in ("model", "trajectory")
    }


def compute(
    *,
    source,
    difference_hz=10.0,
    coupling_us=0.08,
    rate_scale_b=0.8,
    parallel=1,
    run_id=None,
):
    if not 1 <= parallel <= (os.cpu_count() or 1):
        raise ValueError("parallel workers must fit available logical CPUs")
    ancestors = ancestry(REPO, source)
    spiking = ancestors[source]
    if spiking.record["experiment"] != "exp133" or spiking.record["stage"] != "compute":
        raise ValueError("requires an explicit exp133 compute source")
    net = spiking.record["execution"]["configuration"]
    with tempfile.TemporaryDirectory(prefix="exp153-source-") as temporary:
        bundle = lang.load_bundle(
            spiking.restore_bundle(
                "paired-ping.bundle", Path(temporary) / "network.bundle"
            )
        )
        if bundle.manifest["graph_digest"] != net["bundle_digest"]:
            raise ValueError("source bundle does not match recorded configuration")
        cfg = recipe.configuration(
            net,
            imported_weights(spiking, bundle),
            difference_hz=difference_hz,
            coupling_us=coupling_us,
            rate_scale_b=rate_scale_b,
        )
        sources = dict(spiking=spiking)
        with stage_run(
            REPO,
            recipe.SLUG,
            "compute",
            inputs=sources,
            configuration=cfg,
            run_id=run_id,
        ) as run:
            bundle.write(run.export / "network.bundle", visualise=False)
            bundle.visualise(run.export / "circuit.svg", view="circuit")
            retained = run.export / "source-snapshot" / "start"
            retained.mkdir(parents=True)
            for filename in ("snapshot.json", "parameters.npz"):
                shutil.copyfile(
                    spiking.file(recipe.SNAPSHOT, filename), retained / filename
                )
            run.record["scientific_artifacts"] = dict(
                snapshot_stages={"source-snapshot": ["start"]}
            )
            run.record["execution"].update(
                backend="snnlab.meanfield method-of-steps DDE",
                snnlab_version=version("snnlab"),
            )
            conditions = [
                (cfg["example"]["difference_hz"], k)
                for k in (0.0, cfg["example"]["coupling_us"])
            ]
            index = [
                dict(label=f"condition-{n:03d}", difference_hz=d, coupling_us=k)
                for n, (d, k) in enumerate(conditions)
            ]
            write_json_atomic(run.export / "conditions.json", index)
            count = min(parallel, len(index))
            items = [
                dict(
                    id=f"cpu-{worker}",
                    device="cpu",
                    threads=1,
                    conditions=index[worker::count],
                )
                for worker in range(count)
            ]
            run.record["execution"]["parallel_workers"] = count
            if count == 1:
                run_conditions(run.directory, items[0], 0, cfg)
            else:
                # Spawned imports see bounded BLAS/OpenMP pools as well as the
                # per-worker Torch setting. The coordinator owns finalization.
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                ):
                    os.environ[name] = "1"
                run_active_workers(
                    run=run,
                    items=items,
                    worker=run_conditions,
                    arguments=(cfg,),
                    inventory=condition_inventory,
                )
            for item in ancestors.values():
                item.check_unchanged()
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--difference-hz", type=float, default=10.0)
    parser.add_argument("--coupling-us", type=float, default=0.08)
    parser.add_argument("--rate-scale-b", type=float, default=0.8)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--run-id")
    compute(**vars(parser.parse_args()))
