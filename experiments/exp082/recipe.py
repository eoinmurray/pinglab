"""Frozen streaming-inference recipe; no execution or storage on import."""

from pathlib import Path
from typing import Any

from experiments.exp022 import training_run_cell, training_run_values
from experiments.helpers.checkpoints import checkpoint_policy

SLUG = "exp082"
SHARDS = 6
ANALYSIS_PURPOSE = "deployment_performance"
CHECKPOINT_POLICY = checkpoint_policy(ANALYSIS_PURPOSE)
CHECKPOINT_ROLE = CHECKPOINT_POLICY["role"]
SEEDS = tuple(training_run_values("TR-06", "seed"))
TRAINING_RATES_HZ = tuple(training_run_values("TR-06", "input_rates_hz")[0])
PSYCHOMETRIC_RATES_HZ = TRAINING_RATES_HZ
DURATIONS_MS = (25.0, 50.0, 100.0, 200.0)
MATCHED_DURATION_MS, MATCHED_RATE_HZ = 200.0, 5.0
N_CLASSES, N_INPUT = 10, 784
DT_MS = 0.1
STREAMS_PER_CELL, DIGITS_PER_STREAM, STREAM_BATCH_SIZE = 40, 5, 5
EVALUATION_PROFILE = "production"
VARIABLE_STREAM = ((200.0, 0.5), (50.0, 25.0), (100.0, 2.0), (25.0, 10.0), (200.0, 5.0))
SINGLE_TRIAL_TRANSITION_WINDOW_MS = (91.5, 94.5)
CLASS_PROBABILITY_TICKS = (0.0, 0.25, 0.5, 0.75, 1.0)
FIGURES = (
    "single_trial.png",
    "single_trial_transition.png",
    "matched_stream.png",
    "variable_stream.png",
    "psychometric_200ms.svg",
    "duration_rate_summary.png",
    "shared_design_schematic.svg",
)
def training_cell_name(seed):
    return training_run_cell("TR-06", seed=seed)["name"]


def training_dir(seed):
    """Name-only compatibility for registry callers, never an operational input."""
    return Path(training_cell_name(seed))


def configuration(*, smoke=False, streams=None, digits=None, batch=None):
    pilot = any(v is not None for v in (streams, digits))
    cfg: dict[str, Any] = {
        "schema": "exp082.recipe/v1",
        "profile": "smoke" if smoke else "pilot" if pilot else "production",
        "checkpoint_policy": CHECKPOINT_POLICY,
        "seeds": list(SEEDS),
        "training_rates_hz": list(TRAINING_RATES_HZ),
        "psychometric_rates_hz": [0.5, 5.0, 25.0] if smoke else list(TRAINING_RATES_HZ),
        "durations_ms": [50.0, 200.0] if smoke else list(DURATIONS_MS),
        "matched_duration_ms": MATCHED_DURATION_MS,
        "matched_rate_hz": MATCHED_RATE_HZ,
        "streams_per_cell": streams if streams is not None else 1 if smoke else 40,
        "digits_per_stream": digits if digits is not None else 3 if smoke else 5,
        "stream_batch_size": batch if batch is not None else 1 if smoke else 5,
        "dt_ms": DT_MS,
    }
    for k in ("streams_per_cell", "digits_per_stream", "stream_batch_size"):
        if type(cfg[k]) is not int or cfg[k] < 1:
            raise ValueError(f"{k} must be a positive integer")
    cfg["digits_per_seed_cell"] = cfg["streams_per_cell"] * cfg["digits_per_stream"]
    return cfg


def environment_configuration():
    import os

    def value(name):
        raw = os.environ.get("PINGLAB_EXP082_" + name)
        return int(raw) if raw is not None else None

    return configuration(
        smoke=os.environ.get("PINGLAB_SMOKE") == "1",
        streams=value("STREAMS_PER_CELL"),
        digits=value("DIGITS_PER_STREAM"),
        batch=value("STREAM_BATCH_SIZE"),
    )


def validate_configuration(cfg):
    if not isinstance(cfg, dict) or cfg.get("profile") not in (
        "smoke",
        "pilot",
        "production",
    ):
        raise ValueError("invalid exp082 recipe profile")
    expected = configuration(
        smoke=cfg["profile"] == "smoke",
        streams=cfg["streams_per_cell"] if cfg["profile"] != "production" else None,
        digits=cfg["digits_per_stream"] if cfg["profile"] != "production" else None,
        batch=cfg["stream_batch_size"],
    )
    if cfg != expected:
        raise ValueError("exp082 recipe differs from frozen contract")
    return cfg


def _number_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _number_from_tag(value: str) -> float:
    return float(value.replace("p", "."))


def condition_job_id(seed: int, duration_ms: float, rate_hz: float) -> str:
    return f"seed{seed}__d{_number_tag(duration_ms)}__r{_number_tag(rate_hz)}"


def parse_condition_job_id(job_id: str) -> tuple[int, float, float]:
    parts = job_id.split("__")
    if (
        len(parts) != 3
        or not parts[0].startswith("seed")
        or not parts[1].startswith("d")
        or not parts[2].startswith("r")
    ):
        raise ValueError(f"invalid exp082 condition job: {job_id}")
    return (
        int(parts[0].removeprefix("seed")),
        _number_from_tag(parts[1].removeprefix("d")),
        _number_from_tag(parts[2].removeprefix("r")),
    )


def jobs(cfg):
    return [
        {
            "id": condition_job_id(s, d, r),
            "path": "jobs/" + condition_job_id(s, d, r),
            "seed": s,
            "duration_ms": d,
            "rate_hz": r,
            "cell_name": training_cell_name(s),
        }
        for d in cfg["durations_ms"]
        for r in cfg["psychometric_rates_hz"]
        for s in cfg["seeds"]
    ]


def infer_jobs():
    return [j["id"] for j in jobs(environment_configuration())]
