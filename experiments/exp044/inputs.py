"""Validate exp044 lineage up to its explicitly selected v3 training bank."""

from contextlib import contextmanager
from pathlib import Path

from pingstore.contracts import RUN_SCHEMA, PingstoreError, load_json, run_root
from pingstore.stages import source_run, stage_run

from . import recipe

SOURCE_POLICY = "selected-v3-training-bank"


def lineage(repo: Path, identity: str, *, reference: dict | None = None) -> dict:
    found = {}
    visiting = set()

    def visit(run_id, pin):
        if run_id in visiting:
            raise PingstoreError("exp044 input lineage contains a cycle")
        if run_id in found:
            if pin is not None and found[run_id].reference != pin:
                raise PingstoreError(f"conflicting input pins for {run_id}")
            return
        directory = run_root(repo / ".pingstore", run_id)
        if any(
            p.is_symlink()
            for p in (directory / "run.json", directory, *directory.parents)
        ):
            raise PingstoreError("exp044 input paths must not use symlinks")
        if not (directory / "run.json").is_file():
            raise PingstoreError(
                f"exp044 requires complete v3 input lineage: missing {run_id}"
            )
        if load_json(directory / "run.json").get("schema") != RUN_SCHEMA:
            raise PingstoreError(f"exp044 requires v3 evidence: {run_id}")
        run = source_run(repo / ".pingstore", run_id, reference=pin)
        visiting.add(run_id)
        # The user selected the self-contained bank as exp044's new source.
        # Validate its entire payload and manifest, but do not turn its older
        # import history into additional execution dependencies or rewrite it.
        is_bank = (
            run.record["experiment"] == "exp022" and run.record["stage"] == "compute"
        )
        if not is_bank:
            for upstream in run.record["inputs"].values():
                visit(upstream["run_id"], upstream)
        visiting.remove(run_id)
        found[run_id] = run

    visit(identity, reference)
    return found


def source(
    repo: Path,
    identity: str,
    stage: str,
    *,
    experiment: str = recipe.SLUG,
    reference: dict | None = None,
):
    run = lineage(repo, identity, reference=reference)[identity]
    if run.record["stage"] != stage or run.record["experiment"] != experiment:
        raise PingstoreError(f"{identity} is not a {experiment} {stage} run")
    return run


@contextmanager
def execution(
    repo: Path, stage: str, *, sources: dict, run_id=None, configuration=None
):
    ancestors = {}
    for run in sources.values():
        ancestors.update(lineage(repo, run.record["run_id"], reference=run.reference))
    with stage_run(
        repo,
        recipe.SLUG,
        stage,
        inputs=sources,
        run_id=run_id,
        configuration=configuration,
    ) as run:
        run.record["source_boundary"] = {
            "policy": SOURCE_POLICY,
            "scope": "user-selected bank is starting evidence for exp044; not a repository-wide legacy exception",
            "banks": {
                identity: {
                    "reference": ancestor.reference,
                    "historical_inputs_not_traversed": ancestor.record["inputs"],
                }
                for identity, ancestor in ancestors.items()
                if ancestor.record["experiment"] == "exp022"
                and ancestor.record["stage"] == "compute"
            },
        }
        yield run
        # Recheck every dependency up to and including the selected source bank.
        for ancestor in ancestors.values():
            ancestor.check_unchanged()


def configuration(compute) -> dict:
    cfg = compute.record["execution"].get("configuration")
    if (
        not isinstance(cfg, dict)
        or cfg.get("schema") != "exp044.recipe/v1"
        or cfg.get("profile") not in ("smoke", "production")
        or cfg != recipe.configuration(smoke=cfg["profile"] == "smoke")
    ):
        raise PingstoreError("unsupported or inconsistent retained exp044 recipe")
    if set(compute.record["inputs"]) != {"bank"}:
        raise PingstoreError("exp044 compute must pin exactly its training bank")
    return cfg
