"""Strict v4 inputs and complete pinned lineage, without historical fallbacks."""

from contextlib import contextmanager

from pingstore.contracts import PingstoreError, load_json
from pingstore.stages import source_run, stage_run

from . import recipe


def lineage(repo, identity, reference=None):
    found, visiting = {}, set()

    def visit(name, pin):
        if name in visiting:
            raise PingstoreError("exp025 input lineage contains a cycle")
        if name in found:
            if pin is not None and found[name].reference != pin:
                raise PingstoreError("conflicting exp025 input pins")
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
        for ancestor in ancestors.values():
            ancestor.check_unchanged()


def configuration(run):
    cfg = run.record["execution"].get("configuration")
    if (
        not isinstance(cfg, dict)
        or cfg.get("schema") != "exp025.recipe/v1"
        or cfg.get("profile") not in ("smoke", "production")
        or cfg != recipe.configuration(smoke=cfg["profile"] == "smoke")
        or set(run.record["inputs"]) != {"bank"}
    ):
        raise PingstoreError("inconsistent exp025 compute recipe or bank input")
    return cfg


def compute_evidence(repo, run):
    from . import evidence

    cfg = configuration(run)
    pin = run.record["inputs"]["bank"]
    bank = source(repo, pin["run_id"], "compute", experiment="exp022", reference=pin)
    contract = evidence.training_contract(bank.export)
    expected = {
        "schema": "exp025.compute/v1",
        "recipe": cfg,
        "training_contract": contract,
        "jobs": recipe.jobs(cfg),
    }
    if load_json(run.export / "evidence.json") != expected:
        raise PingstoreError("compute evidence differs from the pinned bank/recipe")
    return cfg, bank, contract
