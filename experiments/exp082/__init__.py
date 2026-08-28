"""Pure exp082 definitions; invoke compute, analyse or present explicitly."""

from .measurements import (
    first_correct_trial_from_stream,
    grid_output_preflight,
    output_activity_summary,
    single_trial_from_stream,
    spike_count_logits,
)
from .recipe import (
    ANALYSIS_PURPOSE,
    CHECKPOINT_POLICY,
    CHECKPOINT_ROLE,
    DIGITS_PER_STREAM,
    DURATIONS_MS,
    EVALUATION_PROFILE,
    MATCHED_DURATION_MS,
    PSYCHOMETRIC_RATES_HZ,
    SEEDS,
    STREAM_BATCH_SIZE,
    STREAMS_PER_CELL,
    TRAINING_RATES_HZ,
    infer_jobs,
    parse_condition_job_id,
    training_dir,
)

__all__ = [
    "first_correct_trial_from_stream",
    "grid_output_preflight",
    "output_activity_summary",
    "single_trial_from_stream",
    "spike_count_logits",
    "ANALYSIS_PURPOSE",
    "CHECKPOINT_POLICY",
    "CHECKPOINT_ROLE",
    "SEEDS",
    "TRAINING_RATES_HZ",
    "PSYCHOMETRIC_RATES_HZ",
    "DURATIONS_MS",
    "MATCHED_DURATION_MS",
    "STREAMS_PER_CELL",
    "DIGITS_PER_STREAM",
    "STREAM_BATCH_SIZE",
    "EVALUATION_PROFILE",
    "training_dir",
    "infer_jobs",
    "parse_condition_job_id",
]
