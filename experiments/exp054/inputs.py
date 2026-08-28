"""Explicit v3 sources and complete ancestry, rechecked before completion."""

import os
from contextlib import contextmanager

from pingstore.contracts import PingstoreError
from pingstore.stages import source_run, stage_run

from . import recipe


def lineage(repo, identity, reference=None):
    found, visiting = {}, set()

    def visit(name, pin):
        if name in visiting:
            raise PingstoreError("exp054 input lineage contains a cycle")
        if name in found:
            if pin is not None and found[name].reference != pin:
                raise PingstoreError("conflicting exp054 input pins")
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
    if os.environ.get("SLURM_JOB_ID") and run_id is None:
        raise PingstoreError("exp054 HPC identities must be reserved before submission")
    ancestors = {}
    for run in sources.values():
        for name, ancestor in lineage(
            repo, run.record["run_id"], run.reference
        ).items():
            if name in ancestors and ancestors[name].reference != ancestor.reference:
                raise PingstoreError("conflicting exp054 input ancestry")
            ancestors[name] = ancestor
    with stage_run(
        repo,
        recipe.SLUG,
        stage,
        inputs=sources,
        run_id=run_id,
        configuration=configuration or recipe.configuration(),
        operation=operation,
    ) as run:
        yield run
        for ancestor in ancestors.values():
            ancestor.check_unchanged()


def configuration(run):
    cfg = run.record["execution"].get("configuration")
    return recipe.validate(cfg)
