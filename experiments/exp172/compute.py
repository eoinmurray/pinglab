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


def compute(run_id=None):
    started = perf_counter()
    torch.set_num_threads(1)
    cfg = recipe.configuration()
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
    selection = source_run(
        REPO / ".pingstore",
        cfg["selection_reference"]["run_id"],
        experiment="exp172",
        stage="compute",
        reference=cfg["selection_reference"],
    )
    with stage_run(
        REPO,
        "exp172",
        "compute",
        inputs={"parameter_reference": reference, "selection_reference": selection},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        dump(run.export / "configuration.json", cfg)
        selected = json.loads(selection.file("stability-selection.json").read_text())
        if (
            selected["selected"]["recurrence_scale"] != cfg["c_ee_scale"]
            or selected["selected"]["inhibition_scale"] != cfg["c_ie_scale"]
        ):
            raise ValueError("Recipe differs from the recorded report selection")
        dump(run.export / "stability-selection.json", selected)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    compute(**vars(parser.parse_args()))
