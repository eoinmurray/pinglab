"""The retained timestep audit recipe; training remains owned by exp022."""

from experiments.exp022.recipe import training_run_cell, training_run_values
from experiments.helpers.checkpoints import checkpoint_policy
from experiments.helpers.datasets import MNIST_REDUCED_EVAL_SAMPLES

SLUG = "exp044"
TRAINING_RUN = "TR-04"
ANALYSIS_PURPOSE = "endpoint_dynamics"
CHECKPOINT_POLICY = checkpoint_policy(ANALYSIS_PURPOSE)
CHECKPOINT_ROLE = CHECKPOINT_POLICY["role"]
DT_SWEEP_MS = training_run_values(TRAINING_RUN, "dt_ms")
SEEDS = training_run_values(TRAINING_RUN, "seed")
T_MS = 200.0
MAX_SAMPLES = 7000
EPOCHS = 50
EVAL_MAX_SAMPLES = MNIST_REDUCED_EVAL_SAMPLES
SMOKE_MAX_SAMPLES = 100
RASTER_SAMPLE_IDX = 0
RASTER_N_E_PLOT = 200
RASTER_N_I_PLOT = 64
RASTER_T_WINDOW_MS = 100.0
FIGURES = (
    "dt_sweep.svg",
    "dt_sweep.pdf",
    "raster_strip.png",
    "raster_strip.pdf",
    "training_curves.svg",
    "training_curves.pdf",
)

TRAINING_COMMON_FIELDS = (
    "model",
    "dataset",
    "max_samples",
    "epochs",
    "t_ms",
    "tau_ampa_ms",
    "tau_gaba_ms",
    "input_rate",
    "input_rate_sampling",
    "hidden_sizes",
    "n_in",
    "n_hidden",
    "n_inh",
    "n_out",
    "ei_strength",
    "w_in",
    "w_in_initial_zero_fraction",
    "readout_mode",
    "readout_w_init_mean",
    "readout_w_init_std",
    "surrogate_slope",
    "lr",
    "batch_size",
    "weight_decay",
    "grad_clip",
    "v_grad_dampen",
    "dales_law",
    "trainable_w_ei",
    "trainable_w_ie",
    "dataset_split",
    "validation_encoder_draws",
    "fr_reg_upper_strength",
    "fr_reg_upper_target_hz",
    "recurrent_initial_zero_fraction",
    "adaptive_threshold",
    "train_leak",
    "signed_readout",
    "readout_bias",
    "trainable_w_ee",
    "trainable_w_ii",
    "state_clamp",
    "ei_ratio",
    "w_ee",
    "readout_reduction",
    "readout_reference",
    "readout_units",
    "readout_w_out_scale",
    "tau_m_e_bounds_ms",
    "tau_m_i_bounds_ms",
    "readout_tau_bounds_ms",
    "adapt_tau_bounds_ms",
    "adapt_strength_init_mv",
    "adapt_strength_max_mv",
)


def dt_label(dt_ms: float) -> str:
    return "dt" + f"{dt_ms:g}".replace(".", "p")


def cell_name(dt_ms: float, seed: int) -> str:
    return training_run_cell(TRAINING_RUN, dt_ms=dt_ms, seed=seed)["name"]


def configuration(*, smoke: bool = False) -> dict:
    return {
        "schema": "exp044.recipe/v1",
        "profile": "smoke" if smoke else "production",
        "dt_sweep_ms": list(DT_SWEEP_MS),
        "seeds": list(SEEDS),
        "evaluation_samples": SMOKE_MAX_SAMPLES if smoke else EVAL_MAX_SAMPLES,
        "checkpoint_policy": CHECKPOINT_POLICY,
        "raster": {
            "seed": SEEDS[0],
            "sample_index": RASTER_SAMPLE_IDX,
            "n_e_plot": RASTER_N_E_PLOT,
            "n_i_plot": RASTER_N_I_PLOT,
            "selection_seed": 0,
            "window_ms": RASTER_T_WINDOW_MS,
        },
    }


def inference_args(
    train_dir, checkpoint, destination, *, samples: int, sample_index: int | None = None
) -> list[str]:
    args = [
        "sim",
        "--infer",
        "--device",
        "auto",
        "--load-config",
        str(train_dir / "config.json"),
        "--load-weights",
        str(checkpoint),
        "--out-dir",
        str(destination),
    ]
    if sample_index is None:
        args += ["--max-samples", str(samples)]
    else:
        args += ["--sample-index", str(sample_index)]
    return args
