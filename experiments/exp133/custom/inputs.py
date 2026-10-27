"""Resolve explicit Pingstore sources and preserve their recorded ancestry.

The three stage CLIs use these local helpers to reject conflicting payload pins
and check that source runs remain unchanged while a dependent stage executes."""

from contextlib import contextmanager

from experiments.exp133 import recipe
from pingstore.library import PingstoreError, source_run, stage_run


def lineage(repo, identity, reference=None):
    found, visiting = {}, set()

    def visit(name, pin):
        if name in visiting:
            raise PingstoreError("input lineage contains a cycle")
        if name in found:
            if pin is not None and found[name].reference != pin:
                raise PingstoreError("conflicting input payload pins")
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
        raise PingstoreError(f"expected a {recipe.SLUG} {stage} run")
    if run.record.get("scope", "full") != "full":
        raise PingstoreError("a full compute result is required")
    return run


@contextmanager
def execution(repo, stage, *, sources=None, run_id=None, configuration=None):
    ancestors = {}
    for selected in (sources or {}).values():
        ancestors.update(lineage(repo, selected.record["run_id"], selected.reference))
    with stage_run(
        repo,
        recipe.SLUG,
        stage,
        inputs=sources or {},
        run_id=run_id,
        configuration=configuration or recipe.configuration(),
    ) as run:
        yield run
        for ancestor in ancestors.values():
            ancestor.check_unchanged()
