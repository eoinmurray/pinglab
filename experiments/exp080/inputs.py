"""Validated v3 inputs and complete ancestry, without implicit source selection."""

from contextlib import contextmanager

from pingstore.contracts import PingstoreError
from pingstore.stages import source_run, stage_run

from . import recipe


def configuration(run):
    cfg = run.record["execution"].get("configuration")
    if (
        not isinstance(cfg, dict)
        or cfg.get("profile") not in ("smoke", "production")
        or cfg != recipe.configuration(smoke=cfg["profile"] == "smoke")
    ):
        raise PingstoreError("exp080 requires its complete retained scientific recipe")
    return cfg


def lineage(repo, identity, reference=None):
    found, visiting = {}, set()

    def visit(name, pin):
        if name in visiting:
            raise PingstoreError("exp080 input lineage contains a cycle")
        if name in found:
            if pin is not None and found[name].reference != pin:
                raise PingstoreError("conflicting exp080 input pins")
            return found[name]
        run = source_run(
            repo / ".pingstore", name, experiment=recipe.SLUG, reference=pin
        )
        cfg = configuration(run)
        stage = run.record["stage"]
        roles = {
            "compute": {},
            "analyse": {"compute": "compute"},
            "present": {"analysis": "analyse", "compute": "compute"},
        }[stage]
        if set(run.record["inputs"]) != set(roles):
            raise PingstoreError("exp080 stage has unexpected input roles")
        visiting.add(name)
        ancestors = {}
        for role, expected in roles.items():
            upstream = run.record["inputs"][role]
            ancestor = visit(upstream["run_id"], upstream)
            if ancestor.record["stage"] != expected or configuration(ancestor) != cfg:
                raise PingstoreError("exp080 input stage or recipe disagrees")
            ancestors[role] = ancestor
        if stage == "present" and (
            ancestors["analysis"].record["inputs"]["compute"]
            != ancestors["compute"].reference
        ):
            raise PingstoreError("exp080 presentation has conflicting compute ancestry")
        visiting.remove(name)
        found[name] = run
        return run

    visit(identity, reference)
    return found


def source(repo, identity, stage, *, reference=None):
    result = lineage(repo, identity, reference)[identity]
    if result.record["stage"] != stage:
        raise PingstoreError(f"{identity} is not an exp080 {stage} run")
    return result


@contextmanager
def execution(repo, stage, *, sources, run_id=None, configuration=None):
    ancestors = {}
    for source in sources.values():
        ancestors.update(lineage(repo, source.record["run_id"], source.reference))
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
