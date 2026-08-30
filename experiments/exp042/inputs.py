"""Strict v4 inputs and complete pinned lineage, without historical fallbacks."""

from contextlib import contextmanager

from experiments.helpers.checkpoints import public_provenance, resolve_checkpoint
from pingstore.contracts import PingstoreError, load_json
from pingstore.stages import source_run, stage_run

from . import recipe


def lineage(repo, identity, reference=None):
    found, visiting = {}, set()

    def visit(name, pin):
        if name in visiting:
            raise PingstoreError("exp042 input lineage contains a cycle")
        if name in found:
            if pin is not None and found[name].reference != pin:
                raise PingstoreError("conflicting exp042 input pins")
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
        or cfg.get("schema") != "exp042.recipe/v1"
        or cfg.get("profile") not in ("smoke", "production")
        or cfg != recipe.configuration(smoke=cfg["profile"] == "smoke")
        or set(run.record["inputs"]) != {"bank"}
    ):
        raise PingstoreError("inconsistent exp042 compute recipe or bank input")
    return cfg


def bank_evidence(bank):
    """Validate the three explicit TR-02 baseline cells before reserving work."""
    configs, checkpoints = {}, []
    for seed in recipe.SEEDS:
        name = recipe.cell_name(seed)
        directory = bank.export / name
        cfg = load_json(directory / "config.json")
        for key, expected in (
            ("training_run_id", recipe.TRAINING_RUN),
            ("training_cell_name", name),
            ("seed", seed),
            ("dataset", "mnist"),
            ("ei_strength", 1.0),
            ("fr_reg_upper_strength", 0.0),
        ):
            if cfg.get(key) != expected:
                raise PingstoreError(f"{name}: expected {key}={expected!r}")
        for key in ("dt", "t_ms", "n_hidden", "n_inh"):
            if not isinstance(cfg.get(key), (int, float)) or cfg[key] <= 0:
                raise PingstoreError(f"{name}: invalid {key}")
        try:
            checkpoint = resolve_checkpoint(directory, recipe.CHECKPOINT_ROLE)
        except (RuntimeError, ValueError, TypeError) as exc:
            raise PingstoreError(str(exc)) from exc
        configs[name] = cfg
        checkpoints.append(public_provenance(checkpoint))
    return {"configurations": configs, "checkpoints": checkpoints}
