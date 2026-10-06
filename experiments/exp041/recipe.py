"""The retained inhibitory-decay sweep recipe; training remains owned by exp022."""

from experiments.exp022.checkpoints import checkpoint_policy
from experiments.exp022.recipe import training_run_cell, training_run_values
from experiments.helpers.datasets import MNIST_REDUCED_EVAL_SAMPLES

SLUG = "exp041"
TRAINING_RUN = "TR-03"
ANALYSIS_PURPOSE = "endpoint_dynamics"
CHECKPOINT_POLICY = checkpoint_policy(ANALYSIS_PURPOSE)
CHECKPOINT_ROLE = CHECKPOINT_POLICY["role"]
TAU_GABA_SWEEP = training_run_values(TRAINING_RUN, "tau_gaba")
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
DT_TRAIN = 0.1
REFRACTORY_E_MS = 1.2
REFRACTORY_I_MS = 0.6
REFRACTORY_POLICY = "exact"
TAU_GABA_REFERENCE_MS = 6.0
F_GAMMA_BAND_HZ = (5.0, 150.0)
FIGURES = tuple(
    name + "." + ext
    for name, extensions in (
        ("rate_vs_fgamma", ("svg", "pdf")),
        ("training_curves", ("svg", "pdf")),
        ("psds", ("svg", "pdf")),
        ("per_trial_peaks", ("svg", "pdf")),
        ("raster_strip", ("png", "pdf")),
    )
    for ext in extensions
)


def refractory_configuration() -> dict:
    return {
        "refractory_e_ms": REFRACTORY_E_MS,
        "refractory_i_ms": REFRACTORY_I_MS,
        "refractory_policy": REFRACTORY_POLICY,
    }




TRAINING_COMMON_FIELDS = (
    'model','dataset','max_samples','epochs','t_ms','tau_ampa_ms','dt','input_rate',
    'input_rate_sampling','hidden_sizes','n_in','n_hidden','n_inh','n_out','ei_strength','ei_ratio',
    'w_in','w_in_initial_zero_fraction','recurrent_initial_zero_fraction','readout_mode',
    'readout_w_init_mean','readout_w_init_std','surrogate_slope','lr','batch_size','weight_decay',
    'grad_clip','v_grad_dampen','dales_law','trainable_w_ei','trainable_w_ie',
    'dataset_split','validation_encoder_draws','fr_reg_upper_strength','fr_reg_upper_target_hz',
    'adaptive_threshold','train_leak','signed_readout','readout_bias','trainable_w_ee','trainable_w_ii','state_clamp',
)


def cell_name(tau_ms: float, seed: int) -> str:
    return training_run_cell(TRAINING_RUN, tau_gaba=tau_ms, seed=seed)["name"]


def configuration(*, smoke: bool = False, version=2) -> dict:
    if version not in (1, 2):
        raise ValueError("unsupported exp041 recipe version")
    return {
        "schema": f"exp041.recipe/v{version}",
        **(refractory_configuration() if version >= 2 else {}),
        "profile": "smoke" if smoke else "production",
        "tau_gaba_sweep_ms": list(TAU_GABA_SWEEP),
        "seeds": list(SEEDS),
        "evaluation_samples": SMOKE_MAX_SAMPLES if smoke else EVAL_MAX_SAMPLES,
        "checkpoint_policy": CHECKPOINT_POLICY,
        "f_gamma_band_hz": list(F_GAMMA_BAND_HZ),
        "raster": {
            "seed": SEEDS[0],
            "sample_index": RASTER_SAMPLE_IDX,
            "n_e_plot": RASTER_N_E_PLOT,
            "n_i_plot": RASTER_N_I_PLOT,
            "selection_seed": 0,
            "window_ms": RASTER_T_WINDOW_MS,
        },
    }




BIOPHYSICS = {
    "capacitance_e_nf": 1.0, "capacitance_i_nf": 0.5,
    "leak_e_us": 0.05, "leak_i_us": 0.10,
    "resting_mv": -65.0, "threshold_mv": -50.0, "reset_mv": -65.0,
    "readout_tau_ms": 2.0, "readout_threshold": 1.0,
}


def author_network(training, *, observables=()):
    from experiments.helpers.checkpoint_graph import author_network as author

    return author(
        SLUG, training, BIOPHYSICS, refractory_configuration(), observables=observables
    )


def inference_request(training, cfg, *, snapshot=False):
    request = {
        "checkpoint_role": CHECKPOINT_ROLE,
        "input": "snapshot" if snapshot else "dataset",
        "t_ms": training["t_ms"],
        "input_rate_hz": training["input_rate"],
        "samples": cfg["evaluation_samples"],
        "batch_size": 1 if snapshot else 64,
        "subset_seed": 42,
        "encoder_seed": 20260415,
        "sample_index": cfg["raster"]["sample_index"],
        "observables": ["spikes"] if snapshot else [],
        "products": [] if snapshot else ["population"],
    }
    return request
