"""Validated v4 runs and complete export ancestry for exp085."""

from contextlib import contextmanager

from pingstore.contracts import PingstoreError
from pingstore.stages import source_run, stage_run

from . import recipe


def lineage(repo, identity, reference=None):
    found, visiting = {}, set()

    def visit(name, pin):
        if name in visiting:
            raise PingstoreError("exp085 input lineage contains a cycle")
        if name in found:
            if pin is not None and found[name].reference != pin:
                raise PingstoreError("conflicting exp085 input pins")
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
        raise PingstoreError(f"{identity} is not an exp085 {stage} run")
    if run.record["execution"].get("configuration") != recipe.configuration():
        raise PingstoreError("inconsistent exp085 recipe")
    return run


@contextmanager
def execution(repo, stage, *, sources=None, run_id=None):
    sources = sources or {}
    ancestors = {}
    for source in sources.values():
        for name, ancestor in lineage(
            repo, source.record["run_id"], source.reference
        ).items():
            if name in ancestors and ancestors[name].reference != ancestor.reference:
                raise PingstoreError("conflicting exp085 ancestry")
            ancestors[name] = ancestor
    with stage_run(
        repo,
        recipe.SLUG,
        stage,
        inputs=sources,
        run_id=run_id,
        configuration=recipe.configuration(),
    ) as run:
        yield run
        for ancestor in ancestors.values():
            ancestor.check_unchanged()
