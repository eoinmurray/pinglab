"""Execute the fixed cue sweep and controls used by the report."""

import argparse
import copy
import json
import sys
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import numpy as np
import torch
from experiments.exp172 import recipe
from experiments.exp172.analyse import analyse, dump, stability_metrics
from experiments.exp172.recipe import author_network, inputs
from hpc.concurrent_compute import run_active_workers
from pingstore.library import file_sha256, source_run, stage_run
from snnlab.sim import (
    DenseArrayBinding,
    NetworkSnapshot,
    Performance,
    RecordingSpec,
    SignalRecording,
    SnapshotSpec,
    build,
    simulate,
)


def execute(
    cfg, parameters, item, data, diagram=False, replay=None, record_inhibition=False
):
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
    bundle, _ = author_network(
        local, replay_sender=cfg["mechanism"]["sender"] if replay is not None else None
    )
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
    binding_names = ["X_A", "X_B", "X_C", "U_A", "U_B"]
    if replay is not None:
        name = "replay_" + cfg["mechanism"]["sender"]
        drives[name] = replay[:, None, :]
        binding_names.append(name)
        np.savez_compressed(trial / "replayed_spikes.npz", spikes=replay)
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
            for k in binding_names
        ),
        snapshot=SnapshotSpec(trial / "start", stage="start", parameters=True),
        recording=RecordingSpec(
            (SignalRecording("C_I_to_E.conductance"),),
            block_steps=round(cfg["duration_ms"] / cfg["dt_ms"]),
        )
        if record_inhibition
        else None,
        performance=Performance(compile_mode=None),
        retain_parameters=False,
        retain_runtime_state=False,
        retain_final_state=False,
    )
    simulation_seconds = perf_counter() - sim_tick
    if record_inhibition:
        result.write_recording(trial / "recording")
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
    epochs, profiles = analyse(spikes, local, item["swap"])
    np.savez_compressed(trial / "profiles.npz", **profiles)
    record = dict(
        **item,
        epochs=epochs,
        simulation_seconds=simulation_seconds,
        total_seconds=perf_counter() - tick,
    )
    record["stability"] = stability_metrics(spikes, local, item["swap"])
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
        experiment="exp172",
        stage="compute",
        reference=dict(run_id=cfg["source_run"], payload_digest=cfg["source_digest"]),
    )


def mechanism_pair(cfg, parameters, original_id, data):
    settings = cfg["mechanism"]
    with np.load(data / original_id / "spikes.npz") as z:
        original = dict(z)
    packet = original[settings["sender"] + "_E"]
    item = dict(
        seed=settings["source_seed"], cue_hz=settings["cue_hz"], swap=False, active="AB"
    )
    control = execute(
        cfg,
        parameters,
        dict(item, id="timing-control"),
        data,
        diagram=True,
        replay=packet,
        record_inhibition=True,
    )
    delayed = execute(
        cfg,
        parameters,
        dict(item, id="timing-rescue"),
        data,
        replay=recipe.shifted_packet(packet, cfg["dt_ms"], settings),
        record_inhibition=True,
    )
    descriptor = dict(
        settings=settings,
        original=original_id,
        control=control["id"],
        delayed=delayed["id"],
    )
    dump(data / "mechanism.json", descriptor)
    return [control, delayed]


def compute_mechanism(source_id, run_id=None):
    """Fresh native illustration, pinned to the report's original compute evidence."""
    started = perf_counter()
    torch.set_num_threads(1)
    source = source_run(
        REPO / ".pingstore", source_id, experiment="exp172", stage="compute"
    )
    cfg = dict(
        source.record["execution"]["configuration"],
        mechanism=recipe.mechanism_configuration(),
        stage_mode="mechanism",
    )
    settings = cfg["mechanism"]
    item = next(
        i
        for i in recipe.trial_plan(cfg)
        if i["seed"] == settings["source_seed"]
        and i["cue_hz"] == settings["cue_hz"]
        and i["active"] == "AB"
        and not i["swap"]
    )
    parameters = NetworkSnapshot.read(
        source.file(item["id"], "start/snapshot.json")
    ).load_parameters()
    with stage_run(
        REPO,
        "exp172",
        "compute",
        inputs={"reference_compute": source},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        dump(run.export / "configuration.json", cfg)
        original = execute(
            cfg, parameters, item, run.export, diagram=True, record_inhibition=True
        )
        with (
            np.load(source.file(item["id"], "spikes.npz")) as pinned,
            np.load(run.export / item["id"] / "spikes.npz") as fresh,
        ):
            for key in pinned.files:
                np.testing.assert_array_equal(fresh[key], pinned[key])
        records = [original, *mechanism_pair(cfg, parameters, item["id"], run.export)]
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
    print(run.run_id, flush=True)
    return run.run_id


def candidate_trial(cfg, parameters, original_id, item, data):
    local = copy.deepcopy(cfg)
    settings = local.get("candidate", recipe.candidate_configuration())
    local["background_hz"] = settings["background_hz"]
    candidate_item = {key: item[key] for key in ["seed", "cue_hz", "swap", "active"]}
    candidate_item["id"] = "candidate-background32"
    record = execute(local, parameters, candidate_item, data)
    dump(
        data / "candidate.json",
        dict(original=original_id, trial_id=record["id"], settings=settings),
    )
    return record


def compute_candidate(source_id, run_id=None):
    started = perf_counter()
    torch.set_num_threads(1)
    source = source_run(
        REPO / ".pingstore", source_id, experiment="exp172", stage="compute"
    )
    cfg = dict(
        source.record["execution"]["configuration"],
        candidate=recipe.candidate_configuration(),
        stage_mode="candidate",
    )
    settings = cfg["candidate"]
    item = next(
        i
        for i in recipe.trial_plan(cfg)
        if i["seed"] == settings["source_seed"]
        and i["cue_hz"] == settings["cue_hz"]
        and i["active"] == "AB"
        and not i["swap"]
    )
    parameters = NetworkSnapshot.read(
        source.file(item["id"], "start/snapshot.json")
    ).load_parameters()
    with stage_run(
        REPO,
        "exp172",
        "compute",
        inputs={"reference_compute": source},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        dump(run.export / "configuration.json", cfg)
        record = candidate_trial(cfg, parameters, item["id"], item, run.export)
        dump(
            run.export / "summary.json",
            dict(trials=[record], compute_seconds=perf_counter() - started),
        )
        dump(
            run.export / "timings.json",
            dict(simulation_seconds=record["simulation_seconds"]),
        )
        run.record["scientific_artifacts"] = dict(
            snapshot_stages={record["id"]: ["start"]}
        )
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
    return run.run_id


def background_worker(
    directory, allocation, index, cfg, parameter_path, parameter_digest
):
    torch.set_num_threads(1)
    assert file_sha256(parameter_path) == parameter_digest
    parameters = torch.load(parameter_path, map_location="cpu", weights_only=True)
    records = []
    for position, item in enumerate(allocation["trials"]):
        local = copy.deepcopy(cfg)
        local["background_hz"] = item["background_hz"]
        record = execute(
            local,
            parameters,
            item,
            directory / "export",
            diagram=index == 0 and position == 0,
        )
        records.append(record)
        print(
            f"Background worker {index}: {position + 1}/{len(allocation['trials'])}",
            flush=True,
        )
    assert file_sha256(parameter_path) == parameter_digest
    return records


def background_inventory(directory, allocation, index):
    paths = [
        p
        for item in allocation["trials"]
        for p in (directory / "export" / item["id"]).rglob("*")
        if p.is_file()
    ]
    if index == 0:
        paths.append(directory / "export/circuit.svg")
    return {str(p.relative_to(directory)): file_sha256(p) for p in sorted(paths)}


def compute_background_sweep(source_id, run_id=None, parallel=False):
    started = perf_counter()
    torch.set_num_threads(1)
    source = source_run(
        REPO / ".pingstore", source_id, experiment="exp172", stage="compute"
    )
    settings = recipe.background_sweep_configuration()
    cfg = dict(
        source.record["execution"]["configuration"],
        background_sweep=settings,
        stage_mode="background_sweep",
    )
    original_id = next(
        i["id"]
        for i in recipe.trial_plan(cfg)
        if i["seed"] == 185
        and i["cue_hz"] == settings["cue_hz"]
        and i["active"] == "AB"
        and not i["swap"]
    )
    parameters = NetworkSnapshot.read(
        source.file(original_id, "start/snapshot.json")
    ).load_parameters()
    plan = recipe.background_sweep_plan(settings)
    with stage_run(
        REPO,
        "exp172",
        "compute",
        inputs={"reference_compute": source},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        dump(run.export / "configuration.json", cfg)
        dump(
            run.export / "background-plan.json",
            dict(settings=settings, trials=plan, baseline_trial=original_id),
        )
        prepared = run.scratch / "background-parameters.pt"
        torch.save(parameters, prepared)
        sha = file_sha256(prepared)
        allocations = (
            [
                dict(
                    id=f"background-lane-{i}",
                    device="cpu",
                    threads=1,
                    trials=plan[i::2],
                )
                for i in range(2)
            ]
            if parallel
            else [dict(id="background-lane-0", device="cpu", threads=1, trials=plan)]
        )
        if parallel:
            workers = run_active_workers(
                run=run,
                items=allocations,
                worker=background_worker,
                arguments=(cfg, prepared, sha),
                inventory=background_inventory,
            )
            records = [record for worker in workers for record in worker["result"]]
        else:
            records = background_worker(
                run.directory, allocations[0], 0, cfg, prepared, sha
            )
        assert file_sha256(prepared) == sha
        by_id = {record["id"]: record for record in records}
        assert set(by_id) == {i["id"] for i in plan}
        records = [by_id[i["id"]] for i in plan]
        dump(
            run.export / "summary.json",
            dict(trials=records, compute_seconds=perf_counter() - started),
        )
        dump(
            run.export / "timings.json",
            dict(
                simulation_seconds=sum(r["simulation_seconds"] for r in records),
                wall_seconds=perf_counter() - started,
            ),
        )
        run.record["scientific_artifacts"] = dict(
            snapshot_stages={r["id"]: ["start"] for r in records}
        )
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
    return run.run_id


def compute(
    run_id=None,
    mechanism_source=None,
    candidate_source=None,
    background_source=None,
    parallel=False,
):
    if background_source is not None:
        return compute_background_sweep(background_source, run_id, parallel)
    if parallel:
        raise ValueError("--parallel is currently scoped to the background sweep")
    if candidate_source is not None:
        return compute_candidate(candidate_source, run_id)
    if mechanism_source is not None:
        return compute_mechanism(mechanism_source, run_id)

    started = perf_counter()
    torch.set_num_threads(1)
    cfg = recipe.configuration()
    reference = parameter_reference()
    snapshot = reference.file(cfg["source_snapshot"])
    if file_sha256(snapshot) != cfg["source_snapshot_sha256"]:
        raise ValueError("Snapshot hash mismatch")
    original = NetworkSnapshot.read(snapshot).load_parameters()
    baseline = json.loads(reference.file("configuration.json").read_text())
    items = recipe.trial_plan(cfg)
    with stage_run(
        REPO,
        "exp172",
        "compute",
        inputs={"parameter_reference": reference},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        dump(run.export / "configuration.json", cfg)
        records = [
            execute(
                cfg,
                recipe.selected_parameters(original, cfg, baseline),
                item,
                run.export,
                diagram=i == 0,
                record_inhibition=(
                    item["seed"] == cfg["mechanism"]["source_seed"]
                    and item["cue_hz"] == cfg["mechanism"]["cue_hz"]
                    and item["active"] == "AB"
                    and not item["swap"]
                ),
            )
            for i, item in enumerate(items)
        ]
        selected = next(
            r
            for r in records
            if r["seed"] == cfg["mechanism"]["source_seed"]
            and r["cue_hz"] == cfg["mechanism"]["cue_hz"]
            and r["active"] == "AB"
            and not r["swap"]
        )
        mechanism_records = mechanism_pair(
            cfg,
            recipe.selected_parameters(original, cfg, baseline),
            selected["id"],
            run.export,
        )
        candidate_record = candidate_trial(
            cfg,
            recipe.selected_parameters(original, cfg, baseline),
            selected["id"],
            selected,
            run.export,
        )
        dump(
            run.export / "summary.json",
            dict(trials=records, compute_seconds=perf_counter() - started),
        )
        dump(
            run.export / "timings.json",
            dict(
                simulation_seconds=sum(
                    r["simulation_seconds"]
                    for r in records + mechanism_records + [candidate_record]
                )
            ),
        )
        run.record["scientific_artifacts"] = dict(
            snapshot_stages={
                r["id"]: ["start"]
                for r in records + mechanism_records + [candidate_record]
            }
        )
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--mechanism-source")
    modes.add_argument("--candidate-source")
    modes.add_argument("--background-source")
    parser.add_argument("--parallel", action="store_true")
    compute(**vars(parser.parse_args()))
