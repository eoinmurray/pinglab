"""Strict v4 inputs and complete pinned lineage, without historical fallbacks.

The owning experiment calls these local definitions from its explicit stages.
Inputs and outputs retain that stage’s validated scientific contract; importing
this module does not start computation or select a scientific run.
"""

from contextlib import contextmanager

from experiments.exp049 import recipe
from pingstore.contracts import PingstoreError, load_json
from pingstore.stages import source_run, stage_run


def lineage(repo, identity, reference=None):
    found, visiting = {}, set()

    def visit(name, pin):
        if name in visiting:
            raise PingstoreError("exp049 input lineage contains a cycle")
        if name in found:
            if pin is not None and found[name].reference != pin:
                raise PingstoreError("conflicting exp049 input pins")
            return
        run = source_run(repo / ".pingstore", name, reference=pin)
        visiting.add(name)
        for upstream in run.record["inputs"].values():
            visit(upstream["run_id"], upstream)
        visiting.remove(name)
        found[name] = run

    visit(identity, reference)
    return found


def source(repo, identity, stage, *, experiment=recipe.SLUG, reference=None):
    run = lineage(repo, identity, reference)[identity]
    if run.record["stage"] != stage or run.record["experiment"] != experiment:
        raise PingstoreError(f"{identity} is not a {experiment} {stage} run")
    return run


@contextmanager
def execution(repo, stage, *, sources, run_id=None, configuration=None):
    ancestors = {}
    for run in sources.values():
        ancestors.update(lineage(repo, run.record["run_id"], run.reference))
    with stage_run(
        repo,
        recipe.SLUG,
        stage,
        inputs=sources,
        run_id=run_id,
        configuration=configuration,
    ) as run:
        yield run
        if stage == "compute":
            from experiments.exp049.custom.network_artifacts import (
                record_snapshot_policy,
            )

            record_snapshot_policy(run)
        for ancestor in ancestors.values():
            ancestor.check_unchanged()


def configuration(run):
    cfg = run.record["execution"].get("configuration")
    if (
        not isinstance(cfg, dict)
        or cfg.get("schema")
        not in ("exp049.recipe/v1", "exp049.recipe/v2", "exp049.recipe/v3")
        or cfg.get("profile") not in ("smoke", "production")
        or cfg
        != recipe.configuration(
            smoke=cfg["profile"] == "smoke",
            version=int(cfg["schema"].rsplit("v", 1)[1]),
        )
        or set(run.record["inputs"]) != {"bank"}
    ):
        raise PingstoreError("inconsistent exp049 compute recipe or bank input")
    return cfg


def compute_evidence(repo, run):
    from experiments.exp049.custom import evidence

    cfg = configuration(run)
    pin = run.record["inputs"]["bank"]
    bank = source(repo, pin["run_id"], "compute", experiment="exp022", reference=pin)
    contract = evidence.training_contract(bank.export)
    expected = {
        "schema": "exp049.compute/v1",
        "recipe": cfg,
        "training_contract": contract,
        "jobs": recipe.jobs(cfg),
    }
    if load_json(run.export / "evidence.json") != expected:
        raise PingstoreError("compute evidence differs from the pinned bank/recipe")
    return cfg, bank, contract


def paired_evidence(repo, run, compute, bank):
    """Require a complete matched-input diagnostic pinned to these endpoints."""
    import numpy as np
    from experiments.exp049.custom import evidence

    cfg = recipe.raster_configuration()
    if run.record["execution"].get("configuration") != cfg or run.record["inputs"] != {
        "endpoints": compute.reference,
        "bank": bank.reference,
    }:
        raise PingstoreError("paired raster recipe or endpoint ancestry differs")
    saved = load_json(run.export / "paired-rasters.json")
    if saved.get("config") != cfg:
        raise PingstoreError("paired raster configuration differs")
    expected = [(cond, stage) for cond in recipe.COND_ORDER for stage in cfg["stages"]]
    entries = saved.get("entries", [])
    if [(r.get("condition"), r.get("stage")) for r in entries] != expected:
        raise PingstoreError("paired raster grid incomplete")
    _, _, contract = compute_evidence(repo, compute)
    checkpoints = {r["training_cell"]: r for r in contract["checkpoints"]}
    with np.load(run.export / "input.npz", allow_pickle=False) as raw:
        # Use the same content digest as the producer without importing compute.
        import hashlib

        array = np.ascontiguousarray(raw["spikes"])
        digest = hashlib.sha256()
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
        encoded_digest = digest.hexdigest()
        label = int(raw["label"])
    for entry in entries:
        name = recipe.cell_name(entry["condition"], cfg["seed"])
        path = run.file(entry["stage"], name, "recording.npz")
        relative = path.relative_to(run.export).as_posix()
        if (
            entry.get("file") != relative
            or entry.get("encoded_spikes_sha256") != encoded_digest
            or entry.get("checkpoint")
            != (checkpoints[name] if entry["stage"] == "after" else None)
        ):
            raise PingstoreError("paired raster input or checkpoint differs")
        evidence.recordings(path, contract["configs"][name], {"kind": "snapshot"})
        with np.load(path, allow_pickle=False) as raw:
            if int(raw["label"]) != label:
                raise PingstoreError("paired raster image labels differ")
    return saved
