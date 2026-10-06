"""Strict v4 inputs and complete pinned lineage, without historical fallbacks."""

import math
from contextlib import contextmanager

from experiments.helpers.checkpoints import pinned_checkpoint
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
        or cfg.get("schema") != "exp042.recipe/v7"
        or cfg.get("profile") not in ("smoke", "production")
        or cfg != recipe.configuration(smoke=cfg["profile"] == "smoke")
        or set(run.record["inputs"]) != {"bank"}
    ):
        raise PingstoreError("inconsistent exp042 compute recipe or bank input")
    return cfg


def bank_configuration(directory):
    """Read the supported bank's scientific fields, never a simulator config."""
    cfg = load_json(directory / "config.json")
    name = directory.name
    seed = next((seed for seed in recipe.SEEDS if recipe.cell_name(seed) == name), None)
    if seed is None:
        raise PingstoreError(f"unregistered training cell: {name}")
    expected = {**recipe.BANK_REQUIREMENTS, "training_cell_name": name, "seed": seed}
    for key, value in expected.items():
        if type(cfg.get(key)) is not type(value) or cfg[key] != value:
            raise PingstoreError(f"{name}: expected {key}={value!r}")
    for feature in recipe.UNSUPPORTED_BANK_DYNAMICS:
        if cfg.get(feature) is not False:
            raise PingstoreError(f"{name}: unsupported bank dynamics: {feature}")
    for key in ("n_in", "n_hidden", "n_inh", "n_out", "epochs"):
        if type(cfg.get(key)) is not int or cfg[key] <= 0:
            raise PingstoreError(f"{name}: invalid {key}")
    if (
        cfg["n_in"] != 784
        or cfg["n_out"] != 10
        or cfg.get("hidden_sizes") != [cfg["n_hidden"]]
    ):
        raise PingstoreError(
            f"{name}: requires one hidden population and MNIST dimensions"
        )
    for key in (
        "dt",
        "t_ms",
        "tau_ampa_ms",
        "tau_gaba_ms",
        "input_rate",
        "surrogate_slope",
        "v_grad_dampen",
    ):
        value = cfg.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise PingstoreError(f"{name}: invalid {key}")
    for key, value in recipe.refractory_configuration().items():
        # These read-only bank inputs can predate explicit refractory metadata.
        # Missing metadata is allowed only at the original 0.1-ms timestep.
        if key not in cfg and cfg["dt"] == 0.1:
            continue
        if cfg.get(key) != value:
            raise PingstoreError(f"{name}: inconsistent {key}")
    return {
        key: cfg[key]
        for key in (
            *expected,
            "n_in",
            "n_hidden",
            "n_inh",
            "n_out",
            "epochs",
            "dt",
            "t_ms",
            "tau_ampa_ms",
            "tau_gaba_ms",
            "input_rate",
            "surrogate_slope",
            "v_grad_dampen",
        )
    }


def final_checkpoint(directory, training):
    """Authenticate the registered final epoch; no selection or fallback."""
    metrics = load_json(directory / "metrics.json")
    row = metrics.get("checkpoints", {}).get(recipe.CHECKPOINT_ROLE)
    if (
        not isinstance(row, dict)
        or row.get("filename") != recipe.CHECKPOINT_FILENAME
        or type(row.get("epoch")) is not int
        or row["epoch"] != training["epochs"]
        or metrics.get("config", {}).get("epochs") != training["epochs"]
        or metrics.get("training_cell_name", directory.name) != directory.name
    ):
        raise PingstoreError(f"{directory.name}: invalid final checkpoint registration")
    digest = row.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise PingstoreError("invalid checkpoint SHA-256")
    path = pinned_checkpoint(
        directory,
        role=recipe.CHECKPOINT_ROLE,
        filename=row["filename"],
        sha256=digest,
    )
    return path, {
        "training_cell": directory.name,
        "role": recipe.CHECKPOINT_ROLE,
        "filename": path.name,
        "epoch": row["epoch"],
        "sha256": digest,
    }


def checkpoint_tensors(path, training):
    """Read only the six bank tensors; bind four and verify absent recurrence."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    shapes = recipe.checkpoint_shapes(training)
    if not isinstance(state, dict) or set(state) != set(shapes):
        raise PingstoreError("exp042 checkpoint must contain exactly six bank matrices")
    for key, shape in shapes.items():
        value = state[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or tuple(value.shape) != shape
            or not torch.isfinite(value).all()
        ):
            raise PingstoreError(f"invalid checkpoint tensor: {key}")
        if key in ("W_ee.1", "W_ii.1") and torch.count_nonzero(value):
            raise PingstoreError(f"unsupported same-population recurrence: {key}")
        if key in ("W_ei.1", "W_ie.1") and (value < 0).any():
            raise PingstoreError(f"negative recurrent weight: {key}")
    return state


def bank_evidence(bank):
    """Validate only the three explicit TR-02 cells needed by this experiment."""
    configs, checkpoints = {}, []
    for seed in recipe.SEEDS:
        name = recipe.cell_name(seed)
        directory = bank.unit(name)
        cfg = bank_configuration(directory)
        _, checkpoint = final_checkpoint(directory, cfg)
        configs[name] = cfg
        checkpoints.append(checkpoint)
    common = [
        {k: v for k, v in cfg.items() if k not in ("seed", "training_cell_name")}
        for cfg in configs.values()
    ]
    if any(cfg != common[0] for cfg in common[1:]):
        raise PingstoreError("TR-02 baseline cells disagree on scientific settings")
    return {"configurations": configs, "checkpoints": checkpoints}
