"""Validated v3 evidence with exact stage roles and complete input ancestry."""

from contextlib import contextmanager

from pingstore.contracts import PingstoreError
from pingstore.stages import source_run, stage_run

from . import recipe


def lineage(repo, identity, reference=None):
    found, visiting = {}, set()

    def visit(name, pin):
        if name in visiting:
            raise PingstoreError("exp083 input lineage contains a cycle")
        if name in found:
            if pin is not None and found[name].reference != pin:
                raise PingstoreError("conflicting exp083 input pins")
            return found[name]
        run = source_run(
            repo / ".pingstore", name, experiment=recipe.SLUG, reference=pin
        )
        if run.record["execution"].get("configuration") != recipe.configuration():
            raise PingstoreError(
                "exp083 requires the complete retained scientific recipe"
            )
        roles = {
            "compute": {},
            "analyse": {"compute": "compute"},
            "present": {"analysis": "analyse"},
        }[run.record["stage"]]
        if set(run.record["inputs"]) != set(roles):
            raise PingstoreError("exp083 stage has unexpected input roles")
        visiting.add(name)
        for role, expected in roles.items():
            pin = run.record["inputs"][role]
            upstream = visit(pin["run_id"], pin)
            if upstream.record["stage"] != expected:
                raise PingstoreError("exp083 input stage disagrees")
        visiting.remove(name)
        found[name] = run
        return run

    visit(identity, reference)
    return found


def source(repo, identity, stage):
    run = lineage(repo, identity)[identity]
    if run.record["stage"] != stage:
        raise PingstoreError(f"{identity} is not an exp083 {stage} run")
    return run


@contextmanager
def execution(repo, stage, *, sources, run_id=None):
    ancestors = {}
    for source in sources.values():
        ancestors.update(lineage(repo, source.record["run_id"], source.reference))
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
