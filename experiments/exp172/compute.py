"""Execute the fixed spatial cue sweep, or import existing evidence without simulation."""

import argparse
import copy
import json
import shutil
import sys
import zipfile
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import numpy as np
import torch
from experiments.exp172 import recipe
from experiments.exp172.analyse import analyse, dump, stability_metrics, validate
from experiments.exp172.recipe import author_network, inputs
from pingstore.library import file_sha256, source_run, stage_run
from snnlab.sim import (
    DenseArrayBinding,
    NetworkSnapshot,
    Performance,
    SnapshotSpec,
    build,
    simulate,
)


def execute(cfg, parameters, item, data, diagram=False):
    tick = perf_counter()
    trial = data / item["id"]
    trial.mkdir()
    local = copy.deepcopy(cfg)
    local["cue_hz"] = item["cue_hz"]
    dump(
        trial / "configuration.json",
        dict(
            local,
            trial=item,
            input_seed_sequences={
                key: [item["seed"], 700 + i]
                for i, key in enumerate(["X_A", "X_B", "X_C", "U"])
            },
        ),
    )
    bundle, _ = author_network(local)
    bundle.write(trial / "network.bundle", visualise=False)
    if diagram:
        bundle.visualise(data / "circuit.svg", view="circuit")
    built = build(
        graph=bundle.graph,
        seed=item["seed"],
        device="cpu",
        performance=Performance(compile_mode=None),
    )
    with torch.no_grad():
        for k, p in built.model.parameter_map().items():
            p.copy_(parameters[k])
    checkpoint = trial / "checkpoint.pt"
    torch.save(built.model.state_dict(), checkpoint)
    drives, rates = inputs(
        local, item["seed"], item["cue_hz"], item["swap"], item["active"]
    )
    np.savez_compressed(trial / "input_rates.npz", **rates)
    sim_tick = perf_counter()
    result = simulate(
        graph=bundle.graph,
        seed=item["seed"],
        device="cpu",
        checkpoint=checkpoint,
        input_bindings=tuple(
            DenseArrayBinding(
                k, torch.from_numpy(drives["U" if k.startswith("U_") else k]).float()
            )
            for k in ("X_A", "X_B", "X_C", "U_A", "U_B")
        ),
        snapshot=SnapshotSpec(trial / "start", stage="start", parameters=True),
        performance=Performance(compile_mode=None),
        retain_parameters=False,
        retain_runtime_state=False,
        retain_final_state=False,
    )
    simulation_seconds = perf_counter() - sim_tick
    spikes = {}
    for k, v in result.diagnostics.items():
        raw = v.detach().cpu().numpy()[:, 0, :]
        if not ((raw == 0) | (raw == 1)).all():
            raise ValueError(
                f"Non-binary/non-finite native spikes in {item['id']}: {k}"
            )
        spikes[k.removesuffix("_spikes")] = raw.astype(np.uint8)
    np.savez_compressed(trial / "spikes.npz", **spikes)
    checkpoint.unlink()
    epochs, profiles = analyse(spikes, cfg, item["swap"])
    np.savez_compressed(trial / "profiles.npz", **profiles)
    record = dict(
        **item,
        epochs=epochs,
        simulation_seconds=simulation_seconds,
        total_seconds=perf_counter() - tick,
    )
    record["stability"] = stability_metrics(spikes, cfg, item["swap"])
    dump(trial / "metrics.json", record)
    print(
        json.dumps(
            dict(
                id=item["id"],
                simulation_s=round(simulation_seconds, 2),
                er=[e["er"] for e in epochs],
                C_fit_R2=[e["fits"]["C"]["r_squared"] for e in epochs],
            )
        ),
        flush=True,
    )
    return record


def parameter_reference():
    cfg = recipe.configuration()
    return source_run(
        REPO / ".pingstore",
        cfg["source_run"],
        experiment="exp162",
        stage="compute",
        reference=dict(run_id=cfg["source_run"], payload_digest=cfg["source_digest"]),
    )


def import_existing(directory, run_id=None):
    """Authenticate and retain original scratch evidence; import time is not simulation time."""
    started = perf_counter()
    directory = Path(directory).resolve()
    cfg, summary = validate(directory)
    reference = parameter_reference()
    provenance = json.loads((directory / "provenance.json").read_text())
    if (
        file_sha256(reference.file("execution/start/snapshot.json"))
        != provenance["snapshot_sha256"]
    ):
        raise ValueError("Original parameter source hash differs")
    original_files = {
        p.relative_to(directory).as_posix(): file_sha256(p)
        for p in directory.rglob("*")
        if p.is_file()
    }
    with stage_run(
        REPO,
        "exp172",
        "compute",
        inputs={"parameter_reference": reference},
        run_id=run_id,
        configuration=cfg,
        operation="import",
    ) as run:
        shutil.copytree(directory, run.export, dirs_exist_ok=True)
        run.record["scientific_artifacts"] = dict(
            snapshot_stages={r["id"]: ["start"] for r in summary["trials"]}
        )
        dump(
            run.export / "import.json",
            dict(
                original_directory=str(directory),
                original_file_sha256=original_files,
                original_provenance=provenance,
                note="Imported existing scratch simulations; no simulation performed during migration. Original per-trial settings, raw spikes, snapshots and timing records retained byte-for-byte.",
            ),
        )
        with zipfile.ZipFile(
            run.export / "original-source.zip", "w", zipfile.ZIP_DEFLATED
        ) as archive:
            for p in sorted(directory.parent.iterdir()):
                if p.is_file():
                    archive.write(p, p.name)
        run.record["execution"].update(
            import_seconds=perf_counter() - started,
            original_simulation_seconds=sum(
                r["simulation_seconds"] for r in summary["trials"]
            ),
        )
    print(f"Import {perf_counter() - started:.2f} s; no simulation")
    return run.run_id


def compute(run_id=None, tuning_sources=None, **overrides):
    started = perf_counter()
    torch.set_num_threads(1)
    cfg = recipe.configuration(**overrides)
    reference = parameter_reference()
    snapshot = reference.file("execution/start/snapshot.json")
    if file_sha256(snapshot) != cfg["source_snapshot_sha256"]:
        raise ValueError("Snapshot hash mismatch")
    parameters = recipe.topographic(
        NetworkSnapshot.read(snapshot).load_parameters(), cfg
    )
    items = [
        dict(
            id=f"seed{seed}-swap{int(swap)}-cue{cue:g}",
            seed=seed,
            swap=swap,
            cue_hz=cue,
            active="AB",
        )
        for seed in cfg["seeds"]
        for swap in cfg["spatial_assignments"]
        for cue in cfg["cue_strengths_hz"]
    ]
    items += [
        dict(
            id=f"single{c}-swap{int(swap)}",
            seed=cfg["seeds"][0],
            swap=swap,
            cue_hz=0.0,
            active=c,
        )
        for swap in cfg["spatial_assignments"]
        for c in "AB"
    ]
    tuning = {
        f"search_{i}": source_run(
            REPO / ".pingstore", identity, experiment="exp172", stage="analyse"
        )
        for i, identity in enumerate(tuning_sources or [], 1)
    }
    with stage_run(
        REPO,
        "exp172",
        "compute",
        inputs={"parameter_reference": reference, **tuning},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        dump(run.export / "configuration.json", cfg)
        if tuning:
            conditions = [
                row
                for source in tuning.values()
                for row in json.loads(
                    source.file("stability-results.json").read_text()
                )["conditions"]
            ]
            feasible = [
                r
                for r in conditions
                if min(r["copy_R2"]) >= 0.85
                and min(r["ER"]) >= 0.9
                and r["switch_latency_ms"] is not None
                and r["switch_latency_ms"] <= 100
            ]
            chosen = min(
                feasible,
                key=lambda r: (
                    r["worst_window_wrong_fraction"],
                    r["mean_wrong_fraction"],
                    -min(r["copy_R2"]),
                ),
            )
            if (
                chosen["recurrence_scale"] != cfg["c_ee_scale"]
                or chosen["inhibition_scale"] != cfg["c_ie_scale"]
            ):
                raise ValueError(
                    "Requested condition differs from the recorded feasible stability winner"
                )
            baseline = next(
                r
                for r in conditions
                if r["recurrence_scale"] == 0 and r["inhibition_scale"] == 1.25
            )
            dump(
                run.export / "stability-selection.json",
                dict(
                    baseline=baseline,
                    selected=chosen,
                    conditions=conditions,
                    criterion="Minimise worst-window then mean competing-pattern fraction, subject to both-window copy R2 >= .85, ER >= .90 and sustained-switch latency <= 100 ms; ties prefer weaker-window fidelity. Single-seed post-selection.",
                ),
            )
        records = [
            execute(cfg, parameters, item, run.export, diagram=i == 0)
            for i, item in enumerate(items)
        ]
        dump(
            run.export / "summary.json",
            dict(trials=records, compute_seconds=perf_counter() - started),
        )
        dump(
            run.export / "timings.json",
            dict(simulation_seconds=sum(r["simulation_seconds"] for r in records)),
        )
        run.record["scientific_artifacts"] = dict(
            snapshot_stages={r["id"]: ["start"] for r in records}
        )
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
    return run.run_id


def stability_sweep(
    run_id=None, recurrence_scales=None, inhibition_scales=None, seeds=None
):
    started = perf_counter()
    torch.set_num_threads(1)
    cfg = recipe.configuration()
    settings = cfg["stability_search"]
    for key, value in [
        ("recurrence_scales", recurrence_scales),
        ("inhibition_scales", inhibition_scales),
        ("seeds", seeds),
    ]:
        if value is not None:
            settings[key] = value
    if any(v < 0 or not np.isfinite(v) for v in settings["recurrence_scales"]):
        raise ValueError("Invalid recurrence grid")
    if any(v <= 0 or not np.isfinite(v) for v in settings["inhibition_scales"]):
        raise ValueError("Invalid inhibition grid")
    reference = parameter_reference()
    snapshot = reference.file("execution/start/snapshot.json")
    if file_sha256(snapshot) != cfg["source_snapshot_sha256"]:
        raise ValueError("Snapshot hash mismatch")
    original = NetworkSnapshot.read(snapshot).load_parameters()
    with stage_run(
        REPO,
        "exp172",
        "compute",
        inputs={"parameter_reference": reference},
        run_id=run_id,
        configuration=cfg,
        phase="stability-search",
    ) as run:
        print(f"Running {run.run_id}", flush=True)
        dump(run.export / "configuration.json", cfg)
        records = []
        for seed in settings["seeds"]:
            for ee in settings["recurrence_scales"]:
                for ie in settings["inhibition_scales"]:
                    local = recipe.configuration(
                        c_ee_scale=ee,
                        c_ie_scale=ie,
                        c_ee_kernel_sigma=settings["local_kernel_sigma"],
                        stability_search=settings,
                    )
                    item = dict(
                        id=f"seed{seed}-ee{ee:g}-ie{ie:g}",
                        seed=seed,
                        swap=False,
                        cue_hz=settings["cue_hz"],
                        active="AB",
                    )
                    record = execute(
                        local,
                        recipe.topographic(original, local),
                        item,
                        run.export,
                        diagram=not records,
                    )
                    records.append(record)
                    print(
                        json.dumps(
                            dict(
                                id=item["id"],
                                wrong=record["stability"][
                                    "worst_window_wrong_fraction"
                                ],
                                latency_ms=record["stability"]["switch_latency_ms"],
                            )
                        ),
                        flush=True,
                    )
                    dump(
                        run.export / "summary.json",
                        dict(trials=records, compute_seconds=perf_counter() - started),
                    )
        run.record["scientific_artifacts"] = dict(
            snapshot_stages={r["id"]: ["start"] for r in records}
        )
        run.record["execution"].update(
            elapsed_seconds=perf_counter() - started,
            simulation_seconds=sum(r["simulation_seconds"] for r in records),
        )
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--import-from", type=Path, help="Import completed evidence without simulation"
    )
    parser.add_argument("--stability-sweep", action="store_true")
    parser.add_argument("--recurrence-scales", nargs="+", type=float)
    parser.add_argument("--inhibition-scales", nargs="+", type=float)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--tuning-sources", nargs="+")
    parser.add_argument("--c-ee-scale", type=float)
    parser.add_argument("--c-ie-scale", type=float)
    parser.add_argument("--c-ee-kernel-sigma", type=float)
    args = parser.parse_args()
    if not args.stability_sweep and (
        args.recurrence_scales is not None or args.inhibition_scales is not None
    ):
        parser.error("Grid axes require --stability-sweep")
    if args.import_from:
        import_existing(args.import_from, args.run_id)
    elif args.stability_sweep:
        stability_sweep(
            args.run_id, args.recurrence_scales, args.inhibition_scales, args.seeds
        )
    else:
        overrides = {
            k: v
            for k, v in dict(
                c_ee_scale=args.c_ee_scale,
                c_ie_scale=args.c_ie_scale,
                c_ee_kernel_sigma=args.c_ee_kernel_sigma,
            ).items()
            if v is not None
        }
        if args.seeds:
            overrides["seeds"] = args.seeds
        compute(args.run_id, tuning_sources=args.tuning_sources, **overrides)
