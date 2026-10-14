"""Strict v4 inputs and complete pinned lineage, without historical fallbacks."""

from contextlib import contextmanager

from pingstore.contracts import PingstoreError, load_json
from pingstore.stages import source_run, stage_run

from . import recipe


def lineage(repo, identity, reference=None):
    found, visiting = {}, set()

    def visit(name, pin):
        if name in visiting:
            raise PingstoreError("exp025 input lineage contains a cycle")
        if name in found:
            if pin is not None and found[name].reference != pin:
                raise PingstoreError("conflicting exp025 input pins")
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
        or cfg.get("schema") not in ("exp025.recipe/v1", "exp025.recipe/v2")
        or cfg.get("profile") not in ("smoke", "production")
        or cfg
        != recipe.configuration(
            smoke=cfg["profile"] == "smoke",
            version=int(cfg["schema"].rsplit("v", 1)[1]),
        )
        or set(run.record["inputs"]) != {"bank"}
    ):
        raise PingstoreError("inconsistent exp025 compute recipe or bank input")
    return cfg


def compute_evidence(repo, run):
    from . import evidence

    cfg = configuration(run)
    pin = run.record["inputs"]["bank"]
    bank = source(repo, pin["run_id"], "compute", experiment="exp022", reference=pin)
    contract = evidence.training_contract(bank.export)
    # Historical v4 imports retained this metadata envelope in the payload.
    # Validate it when present, but future compute runs keep execution metadata
    # solely in run.json and export only scientific outputs.
    legacy = run.export / "evidence.json"
    if legacy.is_file():
        expected = {
            "schema": "exp025.compute/v1",
            "recipe": cfg,
            "training_contract": contract,
            "jobs": recipe.jobs(cfg),
        }
        if load_json(legacy) != expected:
            raise PingstoreError("compute evidence differs from the pinned bank/recipe")
    return cfg, bank, contract


def analysis_evidence(repo, run):
    """Validate an endpoint analysis or an explicit full-MNIST panel revision."""
    from .analyse import MEASUREMENT, MEASUREMENT_V1

    result = load_json(run.export / "results.json")
    refs = run.record["inputs"]
    if result.get("schema") == "exp025.analysis/v3":
        from .full_mnist_baseline import MEASUREMENT as BASELINE_MEASUREMENT
        from .full_mnist_baseline import canonical_evidence

        if set(refs) != {"frontier_analysis", "baseline_bank"}:
            raise PingstoreError(
                "baseline analysis requires explicit frontier and bank"
            )
        previous = source(
            repo,
            refs["frontier_analysis"]["run_id"],
            "analyse",
            reference=refs["frontier_analysis"],
        )
        retained = analysis_evidence(repo, previous)
        bank = source(
            repo,
            refs["baseline_bank"]["run_id"],
            "compute",
            experiment="exp022",
            reference=refs["baseline_bank"],
        )
        if run.record["execution"].get("configuration") != BASELINE_MEASUREMENT:
            raise PingstoreError("baseline measurement contract differs")
        if result.get("baseline") != canonical_evidence(bank)[0]:
            raise PingstoreError("canonical baseline disagrees with the bank")
        for key, value in retained.items():
            if key in {"schema", "baseline", "plot_data"}:
                continue
            if result.get(key) != value:
                raise PingstoreError(f"retained analysis changed: {key}")
        for key, value in retained["plot_data"].items():
            if key != "baseline" and result["plot_data"].get(key) != value:
                raise PingstoreError(f"retained plot data changed: {key}")
        if result["plot_data"]["baseline"] != result["baseline"]["curves"]:
            raise PingstoreError("baseline plotting curves differ")
        return result
    if set(refs) != {"compute", "bank"}:
        raise PingstoreError("analysis must pin compute and bank")
    compute = source(
        repo, refs["compute"]["run_id"], "compute", reference=refs["compute"]
    )
    cfg, bank, _ = compute_evidence(repo, compute)
    schema = result.get("schema")
    measurement = MEASUREMENT_V1 if schema == "exp025.analysis/v1" else MEASUREMENT
    if (
        refs["bank"] != bank.reference
        or schema not in ("exp025.analysis/v1", "exp025.analysis/v2")
        or result.get("recipe") != cfg
        or run.record["execution"].get("configuration") != measurement
        or (schema == "exp025.analysis/v1" and result.get("measurement") != measurement)
    ):
        raise PingstoreError("analysis evidence or bank pin differs")
    return result
