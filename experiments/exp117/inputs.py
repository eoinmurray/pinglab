"""Explicit v4 sources and ancestry checks for exp117 stages."""

from __future__ import annotations

from contextlib import contextmanager

from pingstore.contracts import PingstoreError
from pingstore.stages import source_run, stage_run

from . import recipe


def lineage(repo, identity, reference=None):
    found, visiting = {}, set()

    def visit(name, pin):
        if name in visiting:
            raise PingstoreError("exp117 input lineage contains a cycle")
        if name in found:
            if pin is not None and found[name].reference != pin:
                raise PingstoreError("conflicting exp117 input pins")
            return
        run = source_run(repo / ".pingstore", name, reference=pin)
        visiting.add(name)
        for upstream in run.record["inputs"].values():
            visit(upstream["run_id"], upstream)
        visiting.remove(name)
        found[name] = run

    visit(identity, reference)
    return found


def source(repo, identity, stage, *, reference=None):
    run = lineage(repo, identity, reference)[identity]
    if run.record["stage"] != stage or run.record["experiment"] != recipe.SLUG:
        raise PingstoreError(f"{identity} is not an exp117 {stage} run")
    return run


@contextmanager
def execution(repo, stage, *, sources, run_id=None, configuration=None):
    ancestors = {}
    for source_run_record in sources.values():
        for name, ancestor in lineage(
            repo, source_run_record.record["run_id"], source_run_record.reference
        ).items():
            if name in ancestors and ancestors[name].reference != ancestor.reference:
                raise PingstoreError("conflicting exp117 input ancestry")
            ancestors[name] = ancestor
    cfg = recipe.validate(configuration or recipe.configuration())
    with stage_run(
        repo,
        recipe.SLUG,
        stage,
        inputs=sources,
        run_id=run_id,
        configuration=cfg,
    ) as run:
        yield run
        for ancestor in ancestors.values():
            ancestor.check_unchanged()


def configuration(run):
    return recipe.validate(run.record["execution"]["configuration"])
