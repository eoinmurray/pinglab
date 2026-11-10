"""Execute the fixed cue sweep and controls used by the report."""

import argparse
import copy
import json
import shutil
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
    save_runtime_state,
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
        if "contrast_ratio" in item:
            local["contrast_ratio"] = item["contrast_ratio"]
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


def compute_cue_sweep(source_id, run_id=None, parallel=False):
    """Fresh cue trials from the pinned five-seed background-screen winner."""
    started = perf_counter()
    torch.set_num_threads(1)
    analysed = source_run(
        REPO / ".pingstore", source_id, experiment="exp172", stage="analyse"
    )
    ref = analysed.record["inputs"]["compute"]
    source = source_run(
        REPO / ".pingstore",
        ref["run_id"],
        experiment="exp172",
        stage="compute",
        reference=ref,
    )
    measured = json.loads(analysed.file("results.json").read_text())["background_sweep"]
    settings = recipe.cue_sweep_configuration()
    settings["background_hz"] = recipe.best_background(measured["summary"])
    assert settings["seeds"] == measured["settings"]["seeds"]
    original_id = next(
        r["trial_id"]
        for r in measured["seed_rows"]
        if r["seed"] == settings["seeds"][0]
        and r["background_hz"] == settings["background_hz"]
    )
    cfg = json.loads(source.file(original_id, "configuration.json").read_text())
    for key in ("trial", "input_seed_sequences"):
        cfg.pop(key, None)
    cfg.update(cue_sweep=settings, stage_mode="cue_sweep")
    parameters = NetworkSnapshot.read(
        source.file(original_id, "start/snapshot.json")
    ).load_parameters()
    plan = recipe.cue_sweep_plan(settings)
    with stage_run(
        REPO,
        "exp172",
        "compute",
        inputs={"reference_compute": source, "background_analyse": analysed},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        dump(run.export / "configuration.json", cfg)
        dump(
            run.export / "cue-plan.json",
            dict(settings=settings, trials=plan, baseline_trial=original_id),
        )
        prepared = run.scratch / "cue-parameters.pt"
        torch.save(parameters, prepared)
        sha = file_sha256(prepared)
        allocations = (
            [
                dict(id=f"cue-lane-{i}", device="cpu", threads=1, trials=plan[i::2])
                for i in range(2)
            ]
            if parallel
            else [dict(id="cue-lane-0", device="cpu", threads=1, trials=plan)]
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
        by_id = {r["id"]: r for r in records}
        assert set(by_id) == {i["id"] for i in plan}
        records = [by_id[i["id"]] for i in plan]
        dump(
            run.export / "summary.json",
            dict(trials=records, compute_seconds=perf_counter() - started),
        )
        run.record["scientific_artifacts"] = dict(
            snapshot_stages={r["id"]: ["start"] for r in records}
        )
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
    return run.run_id


def compute_contrast_sweep(source_id, run_id=None, parallel=False, extend_source=None):
    """Contrast is the cue; the separate uniform cue remains disabled."""
    started = perf_counter()
    torch.set_num_threads(1)
    analysed = source_run(
        REPO / ".pingstore", source_id, experiment="exp172", stage="analyse"
    )
    ref = analysed.record["inputs"]["compute"]
    source = source_run(
        REPO / ".pingstore",
        ref["run_id"],
        experiment="exp172",
        stage="compute",
        reference=ref,
    )
    measured = json.loads(analysed.file("results.json").read_text())["cue_sweep"]
    settings = recipe.contrast_sweep_configuration()
    settings["background_hz"] = measured["settings"]["background_hz"]
    assert settings["seeds"] == measured["settings"]["seeds"]
    original_id = next(
        r["trial_id"]
        for r in measured["seed_rows"]
        if r["seed"] == settings["seeds"][0] and r["cue_hz"] == 0
    )
    cfg = json.loads(source.file(original_id, "configuration.json").read_text())
    for key in ("trial", "input_seed_sequences"):
        cfg.pop(key, None)
    cfg.update(contrast_sweep=settings, stage_mode="contrast_sweep", contrast_ratio=1.0)
    parameters = NetworkSnapshot.read(
        source.file(original_id, "start/snapshot.json")
    ).load_parameters()
    plan = recipe.contrast_sweep_plan(settings)
    stage_inputs = {"reference_compute": source, "cue_analyse": analysed}
    previous_compute = None
    if extend_source is not None:
        previous = source_run(
            REPO / ".pingstore", extend_source, experiment="exp172", stage="analyse"
        )
        old = json.loads(previous.file("results.json").read_text())["contrast_sweep"]
        assert {k: v for k, v in old["settings"].items() if k != "rates_hz"} == {
            k: v for k, v in settings.items() if k != "rates_hz"
        }
        assert set(old["settings"]["rates_hz"]).issubset(settings["rates_hz"])
        ref_previous = previous.record["inputs"]["compute"]
        previous_compute = source_run(
            REPO / ".pingstore",
            ref_previous["run_id"],
            experiment="exp172",
            stage="compute",
            reference=ref_previous,
        )
        assert previous_compute.record["inputs"]["cue_analyse"] == {
            "run_id": analysed.record["run_id"],
            "payload_digest": analysed.record["payload_digest"],
        }
        plan = [
            item
            for item in plan
            if item["contrast_ratio"] not in old["settings"]["rates_hz"]
        ]
        if not plan:
            raise ValueError("Contrast extension has no new settings")
        stage_inputs["previous_analyse"] = previous
    with stage_run(
        REPO,
        "exp172",
        "compute",
        inputs=stage_inputs,
        run_id=run_id,
        configuration=cfg,
    ) as run:
        dump(run.export / "configuration.json", cfg)
        dump(
            run.export / "contrast-plan.json",
            dict(settings=settings, trials=plan, baseline_trial=original_id),
        )
        if previous_compute is not None:
            # Retain only the small rate profiles needed by the unchanged illustrations.
            for example in old["examples"]:
                target = run.export / example["trial_id"]
                target.mkdir()
                shutil.copyfile(
                    previous_compute.file(example["trial_id"], "input_rates.npz"),
                    target / "input_rates.npz",
                )
        prepared = run.scratch / "contrast-parameters.pt"
        torch.save(parameters, prepared)
        sha = file_sha256(prepared)
        allocations = (
            [
                dict(
                    id=f"contrast-lane-{i}", device="cpu", threads=1, trials=plan[i::2]
                )
                for i in range(2)
            ]
            if parallel
            else [dict(id="contrast-lane-0", device="cpu", threads=1, trials=plan)]
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
        by_id = {r["id"]: r for r in records}
        assert set(by_id) == {i["id"] for i in plan}
        records = [by_id[i["id"]] for i in plan]
        dump(
            run.export / "summary.json",
            dict(trials=records, compute_seconds=perf_counter() - started),
        )
        run.record["scientific_artifacts"] = dict(
            snapshot_stages={r["id"]: ["start"] for r in records}
        )
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
    return run.run_id


def compute_arrivals(source_id, candidate_id, run_id=None):
    started = perf_counter()
    torch.set_num_threads(1)
    if candidate_id is None:
        raise ValueError("--arrival-candidate-source is required")
    baseline = source_run(
        REPO / ".pingstore", source_id, experiment="exp172", stage="compute"
    )
    candidate = source_run(
        REPO / ".pingstore", candidate_id, experiment="exp172", stage="compute"
    )
    settings = recipe.arrival_distribution_configuration()
    base_cfg = baseline.record["execution"]["configuration"]
    base_item = next(
        i
        for i in recipe.trial_plan(base_cfg)
        if i["seed"] == settings["seed"]
        and i["cue_hz"] == settings["cue_hz"]
        and i["active"] == "AB"
        and not i["swap"]
    )
    candidate_trial_id = json.loads(candidate.file("candidate.json").read_text())[
        "trial_id"
    ]
    cfg = dict(
        base_cfg, stage_mode="arrival_distribution", arrival_distribution=settings
    )
    with stage_run(
        REPO,
        "exp172",
        "compute",
        inputs={"figure1_compute": baseline, "figure6_compute": candidate},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        dump(run.export / "configuration.json", cfg)
        cases = []
        records = []
        for index, (label, source, original_id) in enumerate(
            [
                ("figure1", baseline, base_item["id"]),
                ("figure6", candidate, candidate_trial_id),
            ]
        ):
            local = json.loads(
                source.file(original_id, "configuration.json").read_text()
            )
            item = {
                key: local["trial"][key] for key in ["seed", "cue_hz", "swap", "active"]
            }
            item["id"] = "arrivals-" + label
            parameters = NetworkSnapshot.read(
                source.file(original_id, "start/snapshot.json")
            ).load_parameters()
            record = execute(
                local,
                parameters,
                item,
                run.export,
                diagram=index == 0,
                record_inhibition=True,
            )
            with (
                np.load(source.file(original_id, "spikes.npz")) as original,
                np.load(run.export / item["id"] / "spikes.npz") as fresh,
            ):
                for key in original.files:
                    np.testing.assert_array_equal(original[key], fresh[key])
            records.append(record)
            cases.append(
                dict(
                    label=label,
                    trial_id=item["id"],
                    reference_trial=original_id,
                    reference_role=label + "_compute",
                    background_hz=local["background_hz"],
                )
            )
        dump(run.export / "arrival-plan.json", dict(settings=settings, cases=cases))
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


def compute_phase_inhibition(source_id, run_id=None):
    """Record inhibition for exactly the saved Figure 2/3 background trial."""
    started = perf_counter()
    torch.set_num_threads(1)
    source = source_run(
        REPO / ".pingstore", source_id, experiment="exp172", stage="compute"
    )
    plan = json.loads(source.file("background-plan.json").read_text())
    settings = recipe.phase_plot_configuration()
    original_id = next(
        item["id"]
        for item in plan["trials"]
        if item["seed"] == settings["seed"]
        and item["background_hz"]
        == source.record["execution"]["configuration"]["background_hz"]
    )
    cfg = json.loads(source.file(original_id, "configuration.json").read_text())
    ref = source.record["inputs"]["reference_compute"]
    baseline = source_run(
        REPO / ".pingstore",
        ref["run_id"],
        experiment="exp172",
        stage="compute",
        reference=ref,
    )
    parameters = NetworkSnapshot.read(
        baseline.file(plan["baseline_trial"], "start/snapshot.json")
    ).load_parameters()
    item = dict(cfg["trial"], id="phase-inhibition")
    with stage_run(
        REPO,
        "exp172",
        "compute",
        run_id=run_id,
        inputs={"reference_compute": source, "parameter_compute": baseline},
        configuration=dict(cfg, stage_mode="phase_inhibition", phase_plot=settings),
    ) as run:
        record = execute(
            cfg, parameters, item, run.export, diagram=True, record_inhibition=True
        )
        with (
            np.load(source.file(original_id, "spikes.npz")) as original,
            np.load(run.export / item["id"] / "spikes.npz") as fresh,
        ):
            assert set(original.files) == set(fresh.files)
            for key in original.files:
                np.testing.assert_array_equal(original[key], fresh[key])
        dump(
            run.export / "phase-inhibition-plan.json",
            dict(
                trial_id=item["id"],
                reference_trial=original_id,
                settings=settings,
                exact_replay=True,
            ),
        )
        dump(run.export / "summary.json", dict(trials=[record]))
        run.record["scientific_artifacts"] = dict(
            snapshot_stages={item["id"]: ["start"]}
        )
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
    return run.run_id


def compute_probe_window(source_id, run_id=None):
    """Paired phase probes branched from exact native states of the displayed trial."""
    from experiments.exp172.analyse import cycle_spatial_fits

    started = perf_counter()
    torch.set_num_threads(1)
    source = source_run(
        REPO / ".pingstore", source_id, experiment="exp172", stage="compute"
    )
    descriptor = json.loads(source.file("phase-inhibition-plan.json").read_text())
    original_trial = descriptor["trial_id"]
    cfg = json.loads(source.file(original_trial, "configuration.json").read_text())
    settings = recipe.probe_configuration()
    dt = cfg["dt_ms"]
    parameters = NetworkSnapshot.read(
        source.file(original_trial, "start/snapshot.json")
    ).load_parameters()
    with np.load(source.file(original_trial, "spikes.npz")) as z:
        original = dict(z)
    cycles = cycle_spatial_fits(original, cfg)
    peaks = np.array([c["peak_ms"] for c in cycles])
    counts = np.arange(len(peaks))
    selected = [int(np.argmin(abs(peaks - anchor))) for anchor in settings["anchor_ms"]]
    assert len(set(selected)) == len(selected)
    plan = []
    for index in selected:
        cycle = cycles[index]
        assert any(
            cycle["start_ms"] >= lo and cycle["end_ms"] <= hi
            for lo, hi in cfg["stability_search"]["windows_ms"]
        )
        phases = np.array(settings["phases_cycles"])
        emission = np.interp(index - phases, counts, peaks, left=np.nan, right=np.nan)
        assert np.isfinite(emission).all()
        steps = np.rint(emission / dt).astype(int)
        start = int(steps.min() - 1)
        end = int(
            steps.max() + round((settings["delay_ms"] + settings["response_ms"]) / dt)
        )
        actual = index - np.interp(steps * dt, peaks, counts)
        plan.append(
            dict(
                peak_ms=cycle["peak_ms"],
                attended="A" if cycle["peak_ms"] < cfg["switch_ms"] else "B",
                start_step=start,
                end_step=end,
                probe_steps=steps.tolist(),
                phases_cycles=actual.tolist(),
            )
        )
    plan.sort(key=lambda row: row["start_step"])
    drives, _ = inputs(
        cfg,
        cfg["trial"]["seed"],
        cfg["cue_hz"],
        cfg["trial"]["swap"],
        cfg["trial"]["active"],
    )
    native_inputs = {
        k: torch.from_numpy(drives["U" if k.startswith("U_") else k]).float()
        for k in ["X_A", "X_B", "X_C", "U_A", "U_B"]
    }
    bundle, _ = author_network(cfg, probe=settings)
    built = build(
        graph=bundle.graph,
        seed=cfg["trial"]["seed"],
        device="cpu",
        performance=Performance(compile_mode=None),
    )
    with torch.no_grad():
        for key, value in built.model.parameter_map().items():
            if key in parameters:
                value.copy_(parameters[key])
            else:
                assert key == "probe_to_C_E.weight"
                np.testing.assert_allclose(
                    value.detach().numpy(), settings["conductance_us"]
                )
    with stage_run(
        REPO,
        "exp172",
        "compute",
        run_id=run_id,
        inputs={"reference_compute": source},
        configuration=dict(cfg, stage_mode="probe_window", probe=settings),
    ) as run:
        bundle.write(run.export / "network.bundle", visualise=False)
        bundle.visualise(run.export / "probe-circuit.svg", view="circuit")
        checkpoint = run.scratch / "probe-weights.pt"
        torch.save(built.model.state_dict(), checkpoint)
        snapshots = {}
        arrays = {}
        timings = []

        def segment(name, lo, hi, state, probe_step=None, retain_state=False):
            pulse = torch.zeros((hi - lo, 1, 1))
            if probe_step is not None:
                pulse[probe_step - lo, 0, 0] = 1
            tick = perf_counter()
            result = simulate(
                graph=bundle.graph,
                seed=cfg["trial"]["seed"],
                device="cpu",
                checkpoint=checkpoint,
                runtime_state=state.detached() if state is not None else None,
                input_bindings=tuple(
                    DenseArrayBinding(k, value[lo:hi])
                    for k, value in native_inputs.items()
                )
                + (DenseArrayBinding("probe", pulse),),
                snapshot=SnapshotSpec(run.export / name / "start", parameters=True),
                performance=Performance(compile_mode=None),
                retain_parameters=False,
                retain_final_state=False,
                retain_runtime_state=retain_state,
            )
            timings.append(perf_counter() - tick)
            snapshots[name] = ["start"]
            spikes = {
                k.removesuffix("_spikes"): v.detach()
                .cpu()
                .numpy()[:, 0, :]
                .astype(np.uint8)
                for k, v in result.diagnostics.items()
            }
            return result.runtime_state, spikes

        state, completed = None, 0
        rows = []
        for anchor, item in enumerate(plan):
            lo, hi = item["start_step"], item["end_step"]
            state, prefix = segment(
                f"prefix{anchor:02d}", completed, lo, state, retain_state=True
            )
            for key in original:
                np.testing.assert_array_equal(prefix[key], original[key][completed:lo])
            completed = lo
            save_runtime_state(run.export / f"state{anchor:02d}", state)
            _, control = segment(f"control{anchor:02d}", lo, hi, state)
            for key in original:
                np.testing.assert_array_equal(control[key], original[key][lo:hi])
            arrays[f"control{anchor:02d}_C_E"] = control["C_E"]
            for phase_index, step in enumerate(item["probe_steps"]):
                name = f"probe{anchor:02d}-{phase_index:02d}"
                _, spikes = segment(name, lo, hi, state, step)
                for key in original:
                    np.testing.assert_array_equal(
                        spikes[key][: step - lo], control[key][: step - lo]
                    )
                for key in ["A_E", "A_I", "B_E", "B_I"]:
                    np.testing.assert_array_equal(spikes[key], control[key])
                arrays[name + "_C_E"] = spikes["C_E"]
                arrays[name + "_C_I"] = spikes["C_I"]
                rows.append(
                    dict(
                        id=name,
                        anchor=anchor,
                        phase_index=phase_index,
                        phase_cycles=item["phases_cycles"][phase_index],
                        probe_step=step,
                        start_step=lo,
                        end_step=hi,
                    )
                )
            print(f"Probe anchor {anchor + 1}/{len(plan)} complete", flush=True)
        state, tail = segment("tail", completed, len(original["C_E"]), state)
        for key in original:
            np.testing.assert_array_equal(tail[key], original[key][completed:])
        np.savez_compressed(run.export / "probe-spikes.npz", **arrays)
        dump(
            run.export / "probe-plan.json",
            dict(
                settings=settings,
                anchors=plan,
                trials=rows,
                exact_control=True,
                reference_trial=descriptor["reference_trial"],
                reference_seed=cfg["trial"]["seed"],
                reference_background=source.record["inputs"]["reference_compute"],
                response_definition="Net extra C E spikes versus paired control in [arrival, arrival+5ms)",
            ),
        )
        dump(
            run.export / "timings.json",
            dict(simulation_seconds=sum(timings), calls=len(timings)),
        )
        run.record["scientific_artifacts"] = dict(snapshot_stages=snapshots)
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
    return run.run_id


def compute_used_window(source_id, run_id=None):
    """Delete each naturally occurring sender packet on its native path to C."""

    from experiments.exp172.analyse import natural_volley_plan

    started = perf_counter()
    torch.set_num_threads(1)
    source = source_run(
        REPO / ".pingstore", source_id, experiment="exp172", stage="compute"
    )
    descriptor = json.loads(source.file("phase-inhibition-plan.json").read_text())
    trial = descriptor["trial_id"]
    cfg = json.loads(source.file(trial, "configuration.json").read_text())
    settings = recipe.used_window_configuration()
    with np.load(source.file(trial, "spikes.npz")) as z:
        original = dict(z)
    events = natural_volley_plan(original, cfg)
    groups = []
    for peak in sorted({event["reference_peak_ms"] for event in events}):
        group = [dict(event) for event in events if event["reference_peak_ms"] == peak]
        lo, hi = (
            min(row["start_step"] for row in group),
            max(row["end_step"] for row in group),
        )
        for row in group:
            row.update(start_step=lo, end_step=hi)
        groups.append(group)
    groups.sort(key=lambda group: group[0]["start_step"])
    parameters = NetworkSnapshot.read(
        source.file(trial, "start/snapshot.json")
    ).load_parameters()
    bundle, _ = author_network(cfg, replay_both=True)
    built = build(
        graph=bundle.graph,
        seed=cfg["trial"]["seed"],
        device="cpu",
        performance=Performance(compile_mode=None),
    )
    assert set(built.model.parameter_map()) == set(parameters)
    with torch.no_grad():
        for key, value in built.model.parameter_map().items():
            value.copy_(parameters[key])
    drives, _ = inputs(
        cfg,
        cfg["trial"]["seed"],
        cfg["cue_hz"],
        cfg["trial"]["swap"],
        cfg["trial"]["active"],
    )
    native_inputs = {
        key: torch.from_numpy(drives["U" if key.startswith("U_") else key]).float()
        for key in ["X_A", "X_B", "X_C", "U_A", "U_B"]
    }
    native_inputs.update(
        {
            "replay_" + c: torch.from_numpy(original[c + "_E"][:, None, :]).float()
            for c in "AB"
        }
    )
    with stage_run(
        REPO,
        "exp172",
        "compute",
        run_id=run_id,
        inputs={"reference_compute": source},
        configuration=dict(cfg, stage_mode="used_window", used_window=settings),
    ) as run:
        bundle.write(run.export / "network.bundle", visualise=False)
        bundle.visualise(run.export / "used-window-circuit.svg", view="circuit")
        checkpoint = run.scratch / "weights.pt"
        torch.save(built.model.state_dict(), checkpoint)
        snapshots, arrays, timings, records = {}, {}, [], []

        def segment(name, lo, hi, state, deletion=None, retain_state=False):
            bindings = []
            for key, values in native_inputs.items():
                window = values[lo:hi]
                if deletion is not None and key == "replay_" + deletion["sender"]:
                    window = window.clone()
                    window[
                        deletion["packet_start"] - lo : deletion["packet_end"] - lo
                    ] = 0
                bindings.append(DenseArrayBinding(key, window))
            tick = perf_counter()
            result = simulate(
                graph=bundle.graph,
                seed=cfg["trial"]["seed"],
                device="cpu",
                checkpoint=checkpoint,
                runtime_state=state.detached() if state is not None else None,
                input_bindings=tuple(bindings),
                snapshot=SnapshotSpec(run.export / name / "start", parameters=True),
                performance=Performance(compile_mode=None),
                retain_parameters=False,
                retain_final_state=False,
                retain_runtime_state=retain_state,
            )
            timings.append(perf_counter() - tick)
            snapshots[name] = ["start"]
            spikes = {
                key.removesuffix("_spikes"): value.detach()
                .cpu()
                .numpy()[:, 0, :]
                .astype(np.uint8)
                for key, value in result.diagnostics.items()
            }
            return result.runtime_state, spikes

        state, completed = None, 0
        for index, group in enumerate(groups):
            lo, hi = group[0]["start_step"], group[0]["end_step"]
            assert lo >= completed
            if lo > completed:
                state, prefix = segment(
                    f"prefix{index:03d}", completed, lo, state, retain_state=True
                )
                for key in original:
                    np.testing.assert_array_equal(
                        prefix[key], original[key][completed:lo]
                    )
                completed = lo
            control_id = f"control{index:03d}"
            _, control = segment(control_id, lo, hi, state)
            for key in original:
                np.testing.assert_array_equal(control[key], original[key][lo:hi])
            arrays[control_id] = control["C_E"]
            save_runtime_state(run.export / f"state{index:03d}", state)
            for event in group:
                _, deleted = segment(event["id"], lo, hi, state, event)
                unaffected = (
                    event["packet_start"]
                    + round(cfg["network"]["cross"]["delay_ms"] / cfg["dt_ms"])
                    - lo
                )
                for key in ["A_E", "A_I", "B_E", "B_I"]:
                    np.testing.assert_array_equal(deleted[key], control[key])
                np.testing.assert_array_equal(
                    deleted["C_E"][:unaffected], control["C_E"][:unaffected]
                )
                arrays[event["id"]] = deleted["C_E"]
                records.append(dict(event, control_id=control_id))
            if (index + 1) % 10 == 0 or index + 1 == len(groups):
                print(
                    f"Natural-volley states {index + 1}/{len(groups)} complete",
                    flush=True,
                )
        _, tail = segment("tail", completed, len(original["C_E"]), state)
        for key in original:
            np.testing.assert_array_equal(tail[key], original[key][completed:])
        np.savez_compressed(run.export / "used-window-spikes.npz", **arrays)
        dump(
            run.export / "used-window-plan.json",
            dict(
                settings=settings,
                events=records,
                states=len(groups),
                exact_control=True,
                reference_trial=descriptor["reference_trial"],
                reference_seed=cfg["trial"]["seed"],
                reference_background=source.record["inputs"]["reference_compute"],
            ),
        )
        dump(
            run.export / "timings.json",
            dict(simulation_seconds=sum(timings), calls=len(timings)),
        )
        run.record["scientific_artifacts"] = dict(snapshot_stages=snapshots)
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
    return run.run_id


def compute(
    run_id=None,
    mechanism_source=None,
    candidate_source=None,
    background_source=None,
    cue_source=None,
    contrast_source=None,
    contrast_extend_source=None,
    parallel=False,
    arrival_source=None,
    arrival_candidate_source=None,
    phase_inhibition_source=None,
    probe_source=None,
    used_window_source=None,
):
    if used_window_source is not None:
        return compute_used_window(used_window_source, run_id)
    if probe_source is not None:
        return compute_probe_window(probe_source, run_id)
    if phase_inhibition_source is not None:
        return compute_phase_inhibition(phase_inhibition_source, run_id)
    if arrival_source is not None:
        return compute_arrivals(arrival_source, arrival_candidate_source, run_id)
    if arrival_candidate_source is not None:
        raise ValueError("--arrival-candidate-source requires --arrival-source")
    if contrast_source is not None:
        return compute_contrast_sweep(
            contrast_source, run_id, parallel, contrast_extend_source
        )
    if cue_source is not None:
        return compute_cue_sweep(cue_source, run_id, parallel)
    if background_source is not None:
        return compute_background_sweep(background_source, run_id, parallel)
    if parallel:
        raise ValueError("--parallel requires --background-source or --cue-source")
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
    modes.add_argument("--cue-source")
    modes.add_argument("--contrast-source")
    modes.add_argument("--arrival-source")
    modes.add_argument("--phase-inhibition-source")
    modes.add_argument("--probe-source")
    modes.add_argument("--used-window-source")
    parser.add_argument("--arrival-candidate-source")
    parser.add_argument("--contrast-extend-source")
    parser.add_argument("--parallel", action="store_true")
    compute(**vars(parser.parse_args()))
