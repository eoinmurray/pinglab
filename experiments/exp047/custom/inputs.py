"""Strict v4 sources and complete pinned ancestry, rechecked before completion.

The owning experiment calls these local definitions from its explicit stages.
Inputs and outputs retain that stage’s validated scientific contract; importing
this module does not start computation or select a scientific run.
"""

from contextlib import contextmanager

from pingstore.contracts import PingstoreError
from pingstore.stages import source_run, stage_run

from experiments.exp047 import recipe


def lineage(repo, identity, reference=None):
    found, visiting = {}, set()

    def visit(name, pin):
        if name in visiting:
            raise PingstoreError("exp047 input lineage contains a cycle")
        if name in found:
            if pin is not None and found[name].reference != pin:
                raise PingstoreError("conflicting exp047 input pins")
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
        raise PingstoreError(f"{identity} is not an exp047 {stage} run")
    return run


@contextmanager
def execution(repo, stage, *, sources, run_id=None, configuration=None):
    ancestors = {}
    for source_ in sources.values():
        for identity, ancestor in lineage(
            repo, source_.record["run_id"], source_.reference
        ).items():
            if (
                identity in ancestors
                and ancestors[identity].reference != ancestor.reference
            ):
                raise PingstoreError("conflicting exp047 input pins")
            ancestors[identity] = ancestor
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
            from experiments.exp047.custom.network_artifacts import (
                record_snapshot_policy,
            )

            record_snapshot_policy(run)
        for ancestor in ancestors.values():
            ancestor.check_unchanged()
