"""Strict v4 inputs and complete pinned lineage, without historical fallbacks.

The owning experiment calls these local definitions from its explicit stages.
Inputs and outputs retain that stage’s validated scientific contract; importing
this module does not start computation or select a scientific run.
"""

from contextlib import contextmanager

from experiments.exp046 import recipe
from pingstore.contracts import PingstoreError, load_json
from pingstore.stages import source_run, stage_run


def lineage(repo, identity, reference=None):
    found, visiting = {}, set()

    def visit(name, pin):
        if name in visiting:
            raise PingstoreError("exp046 input lineage contains a cycle")
        if name in found:
            if pin is not None and found[name].reference != pin:
                raise PingstoreError("conflicting exp046 input pins")
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
            from experiments.exp046.custom.network_artifacts import (
                record_snapshot_policy,
            )

            record_snapshot_policy(run)
        for ancestor in ancestors.values():
            ancestor.check_unchanged()


def configuration(run):
    cfg = run.record["execution"].get("configuration")
    if (
        not isinstance(cfg, dict)
        or cfg.get("schema") not in ("exp046.recipe/v1", "exp046.recipe/v2")
        or cfg.get("profile") not in ("smoke", "production")
        or cfg
        != recipe.configuration(
            smoke=cfg["profile"] == "smoke",
            version=int(cfg["schema"].rsplit("v", 1)[1]),
        )
        or set(run.record["inputs"]) != {"bank"}
    ):
        raise PingstoreError("inconsistent exp046 compute recipe or bank input")
    return cfg


def compute_evidence(repo, compute):
    from experiments.exp046.custom import evidence

    cfg = configuration(compute)
    ref = compute.record["inputs"]["bank"]
    bank = source(repo, ref["run_id"], "compute", experiment="exp022", reference=ref)
    contract = evidence.training_contract(bank.export)
    checkpoints = evidence.checkpoints(bank.export, contract)
    retained = load_json(compute.export / "evidence.json")
    from experiments.exp046.custom.source_exp041_custom_evidence import (
        retained_contract,
    )

    if retained != {
        "schema": "exp046.compute/v1",
        "config": cfg,
        "training_contract": retained_contract(
            retained.get("training_contract"), contract, bank.export
        ),
        "checkpoint_provenance": checkpoints,
    }:
        raise PingstoreError("exp046 compute evidence differs from its pinned bank")
    return cfg, bank, contract, checkpoints


def frequency_evidence(repo, frequencies, bank, cfg, checkpoints):
    from experiments.exp046.custom import source_exp041_custom_inputs as upstream_inputs
    from experiments.exp046.custom.evidence import finite
    from experiments.exp046.custom.source_exp041_analyse import MEASUREMENT

    refs = frequencies.record["inputs"]
    if set(refs) != {"compute", "bank"} or refs["bank"] != bank.reference:
        raise PingstoreError(
            "exp041 frequencies and exp046 require the same pinned training bank"
        )
    upstream = source(
        repo,
        refs["compute"]["run_id"],
        "compute",
        experiment="exp041",
        reference=refs["compute"],
    )
    upstream_cfg = upstream_inputs.configuration(upstream)
    if (
        upstream.record["inputs"]["bank"] != bank.reference
        or upstream_cfg["profile"] != cfg["profile"]
        or upstream_cfg["evaluation_samples"] != cfg["evaluation_samples"]
    ):
        raise PingstoreError(
            "exp041 and exp046 evaluation profiles or bank pins differ"
        )
    result = load_json(frequencies.export / "results.json")
    if (
        result.get("schema") != "exp041.analysis/v1"
        or result.get("recipe") != upstream_cfg
        or result.get("measurement") != MEASUREMENT
        or frequencies.record["execution"].get("configuration") != MEASUREMENT
        or result.get("checkpoint_provenance") != checkpoints
    ):
        raise PingstoreError("inconsistent exp041 frequency or checkpoint evidence")
    found = {}
    expected = {
        (tau, seed) for tau in recipe.TAU_GABA_SWEEP_MS for seed in recipe.SEEDS
    }
    for row in result["results"]:
        key = (row["tau_gaba_ms"], row["seed"])
        if (
            key not in expected
            or key in found
            or row.get("n_total") != cfg["evaluation_samples"]
        ):
            raise PingstoreError("incomplete or duplicate exp041 frequency grid")
        value = finite(row.get("f_gamma_hz"), "upstream gamma frequency")
        if value <= 0:
            raise PingstoreError("upstream gamma frequency must be positive")
        found[key] = value
    if set(found) != expected:
        raise PingstoreError("incomplete exp041 frequency grid")
    return found
