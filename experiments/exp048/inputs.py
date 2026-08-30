"""Explicit v4 stage inputs, with full ancestry validation before and after work."""

from contextlib import contextmanager

from pingstore.contracts import PingstoreError, load_json
from pingstore.stages import source_run, stage_run

from . import recipe


def lineage(repo, identity, reference=None):
    found, visiting = {}, set()

    def visit(name, pin):
        if name in visiting:
            raise PingstoreError("exp048 input lineage contains a cycle")
        if name in found:
            if pin is not None and found[name].reference != pin:
                raise PingstoreError("conflicting exp048 input pins")
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
def execution(
    repo, stage, *, sources, run_id=None, configuration=None, operation="execute"
):
    ancestors = {}
    for run in sources.values():
        for name, ancestor in lineage(
            repo, run.record["run_id"], run.reference
        ).items():
            if name in ancestors and ancestor.reference != ancestors[name].reference:
                raise PingstoreError("conflicting exp048 ancestry")
            ancestors[name] = ancestor
    with stage_run(
        repo,
        recipe.SLUG,
        stage,
        inputs=sources,
        run_id=run_id,
        configuration=configuration,
        operation=operation,
    ) as run:
        yield run
        for ancestor in ancestors.values():
            ancestor.check_unchanged()


def compute_evidence(repo, run):
    from .evidence import training_contract

    if run.record["execution"].get("configuration") != recipe.configuration() or set(
        run.record["inputs"]
    ) != {"bank"}:
        raise PingstoreError("exp048 compute recipe or bank role differs")
    pin = run.record["inputs"]["bank"]
    bank = source(repo, pin["run_id"], "compute", experiment="exp022", reference=pin)
    contract = training_contract(bank.export)
    expected = {
        "schema": "exp048.compute/v1",
        "recipe": recipe.configuration(),
        "training_contract": contract,
        "jobs": recipe.jobs(),
    }
    if load_json(run.export / "evidence.json") != expected:
        raise PingstoreError("exp048 compute evidence differs from recipe or bank")
    return bank, contract
