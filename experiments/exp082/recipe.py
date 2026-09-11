"""Frozen streaming-inference recipe; no execution or storage on import."""

from pathlib import Path
from typing import Any

from experiments.exp022 import training_run_cell, training_run_values
from experiments.exp022.checkpoints import checkpoint_policy
from snnsim.timing import refractory_metadata

SLUG = "exp082"
REFRACTORY_E_MS = 1.2
REFRACTORY_I_MS = 0.6
REFRACTORY_POLICY = "exact"
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
IMAGE_SAMPLING_SEED = 82_000
ENCODING_SEED_BASE = 830_000
IMAGE_STREAM_POLICY = "shared-across-training-seeds-durations-rates/v1"
ENCODING_SEED_POLICY = "cantor-paired-training-duration-rate-stream-indices/v1"
VARIABLE_STREAM = ((200.0, 0.5), (50.0, 25.0), (100.0, 2.0), (25.0, 10.0), (200.0, 5.0))
# Both duration and rate vary within each candidate. The protocol and candidate
# order are fixed before inference; retain the first 5/5 and first 3/5 streams.
SHOWCASE_CONDITIONS = (
    (100.0, 5.0),
    (200.0, 7.5),
    (50.0, 25.0),
    (100.0, 15.0),
    (200.0, 10.0),
)
FIXED_DURATION_SHOWCASE_CONDITIONS = tuple(
    (200.0, rate) for rate in (5.0, 7.5, 10.0, 15.0, 25.0)
)
SHOWCASE_DIGIT_SEED_BASE = 820_000
SHOWCASE_ENCODING_SEED_BASE = 830_000
SHOWCASE_CANDIDATE_LIMIT = 100
SHOWCASE_TARGETS = {"hero": 5, "alternative": 3}
SINGLE_TRIAL_TRANSITION_WINDOW_MS = (91.5, 94.5)
CLASS_PROBABILITY_TICKS = (0.0, 0.25, 0.5, 0.75, 1.0)
FIGURES = (
    "hero_stream.png",
    "alternative_stream.png",
    "single_trial.png",
    "single_trial_transition.png",
    "matched_stream.png",
    "variable_stream.png",
    "psychometric_200ms.svg",
    "duration_rate_summary.png",
    "continuous_stream_compound.png",
    "continuous_stream_compound.pdf",
)


def refractory_configuration() -> dict:
    return {
        "refractory_e_ms": REFRACTORY_E_MS,
        "refractory_i_ms": REFRACTORY_I_MS,
        "refractory_policy": REFRACTORY_POLICY,
    }


def refractory_args() -> list[str]:
    return [
        "--refractory-e-ms",
        str(REFRACTORY_E_MS),
        "--refractory-i-ms",
        str(REFRACTORY_I_MS),
        "--refractory-policy",
        REFRACTORY_POLICY,
    ]


def refractory_execution_configuration(dt_ms: float) -> dict:
    return refractory_metadata(
        REFRACTORY_E_MS,
        REFRACTORY_I_MS,
        dt_ms,
        policy=REFRACTORY_POLICY,
    )


def training_cell_name(seed):
    return training_run_cell("TR-06", seed=seed)["name"]


def training_dir(seed):
    """Name-only compatibility for registry callers, never an operational input."""
    return Path(training_cell_name(seed))


def configuration(*, smoke=False, streams=None, digits=None, batch=None, version=3):
    if version not in (1, 2, 3):
        raise ValueError("unsupported exp082 recipe version")
    pilot = any(v is not None for v in (streams, digits))
    cfg: dict[str, Any] = {
        "schema": f"exp082.recipe/v{version}",
        **(refractory_configuration() if version >= 2 else {}),
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
        **(
            {
                "image_stream_policy": IMAGE_STREAM_POLICY,
                "image_sampling_seed": IMAGE_SAMPLING_SEED,
                "encoding_seed_policy": ENCODING_SEED_POLICY,
                "encoding_seed_base": ENCODING_SEED_BASE,
            }
            if version >= 3
            else {}
        ),
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
    if cfg.get("schema") not in (
        "exp082.recipe/v1",
        "exp082.recipe/v2",
        "exp082.recipe/v3",
    ):
        raise ValueError("invalid exp082 recipe schema")
    expected = configuration(
        version=int(cfg["schema"].rsplit("v", 1)[1]),
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


def _cantor_pair(left: int, right: int) -> int:
    total = left + right
    return total * (total + 1) // 2 + right


def encoding_seed(job: dict[str, Any], stream_index: int) -> int:
    """Injectively map one condition and stream to its encoding RNG seed."""
    if type(stream_index) is not int or stream_index < 0:
        raise ValueError("stream index must be a non-negative integer")
    try:
        indices = (
            SEEDS.index(job["seed"]),
            DURATIONS_MS.index(job["duration_ms"]),
            PSYCHOMETRIC_RATES_HZ.index(job["rate_hz"]),
            stream_index,
        )
    except (KeyError, ValueError) as exc:
        raise ValueError("encoding seed requires a frozen exp082 condition") from exc
    value = indices[0]
    for index in indices[1:]:
        value = _cantor_pair(value, index)
    return ENCODING_SEED_BASE + value


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
