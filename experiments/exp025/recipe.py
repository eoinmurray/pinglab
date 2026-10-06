"""The retained exp025 recipe and explicit inference jobs; no storage access."""

from experiments.exp022 import FR_STRENGTH_UPPER as FR_STRENGTH_UPPER
from experiments.exp022 import training_run_cell, training_run_values
from experiments.helpers.datasets import MNIST_REDUCED_EVAL_SAMPLES

SLUG = "exp025"
REFRACTORY_E_MS = 1.2
REFRACTORY_I_MS = 0.6
REFRACTORY_POLICY = "exact"
ANALYSIS_PURPOSE = "endpoint_dynamics"
CHECKPOINT_ROLE = "final_epoch"
CHECKPOINT_POLICY = {"purpose": ANALYSIS_PURPOSE, "role": CHECKPOINT_ROLE}
MODELS = list(training_run_values("TR-02", "model"))
SEEDS = list(training_run_values("TR-02", "seed"))
RATE_TARGET_GRID_HZ = list(training_run_values("TR-02", "rate_target_hz"))
LOW_W_IN_VALUES = training_run_values("TR-07", "w_in")
EVAL_MAX_SAMPLES = MNIST_REDUCED_EVAL_SAMPLES
F_GAMMA_BAND_HZ = (5.0, 150.0)
W_IN_SCALE_VALUES = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.80,
    0.90,
    1.00,
    1.15,
    1.30,
    1.50,
    1.75,
    2.00,
    2.50,
    3.00,
]
FIGURES = tuple(
    f"{name}.{ext}"
    for name in (
        "theta_p_fgamma",
        "low_w_in_sweep",
        "w_in_scale_sweep",
        "w_in_scale_sweep_vs_rate",
    )
    for ext in ("svg", "pdf")
) + tuple(f"results_compound.{ext}" for ext in ("png", "pdf"))


def refractory_configuration() -> dict:
    return {
        "refractory_e_ms": REFRACTORY_E_MS,
        "refractory_i_ms": REFRACTORY_I_MS,
        "refractory_policy": REFRACTORY_POLICY,
    }




def cell_name(model, rate_target_hz, seed):
    return training_run_cell(
        "TR-02", model=model, rate_target_hz=rate_target_hz, seed=seed
    )["name"]


def low_w_in_cell_name(w_in, seed):
    return training_run_cell("TR-07", w_in=w_in, seed=seed)["name"]


def rate_target_display(value):
    return "off" if value is None else f"{value:g}"


def bank_cells():
    return [
        {
            "cell_name": cell_name(m, t, s),
            "group": "shared_tr02",
            "model": m,
            "rate_target_hz": t,
            "seed": s,
            "w_in": 0.9,
        }
        for m in MODELS
        for t in RATE_TARGET_GRID_HZ
        for s in SEEDS
    ] + [
        {
            "cell_name": low_w_in_cell_name(w, s),
            "group": "low_w_in_controls",
            "model": "ping",
            "rate_target_hz": 1.0,
            "seed": s,
            "w_in": w,
        }
        for w in LOW_W_IN_VALUES
        for s in SEEDS
    ]


def configuration(*, smoke=False, version=2):
    if version not in (1, 2):
        raise ValueError("unsupported exp025 recipe version")
    return {
        "schema": f"exp025.recipe/v{version}",
        **(refractory_configuration() if version >= 2 else {}),
        "profile": "smoke" if smoke else "production",
        "evaluation_samples": 100 if smoke else EVAL_MAX_SAMPLES,
        "pfg_samples": 100 if smoke else min(256 * 5, EVAL_MAX_SAMPLES),
        "low_w_in_seeds": [42] if smoke else list(SEEDS),
        "scales": [0.5, 1.0, 3.0] if smoke else list(W_IN_SCALE_VALUES),
        "checkpoint_policy": CHECKPOINT_POLICY,
    }


def jobs(cfg):
    rows = []
    for m in MODELS:
        for t in RATE_TARGET_GRID_HZ:
            for s in SEEDS:
                name = cell_name(m, t, s)
                rows.append(
                    {
                        "kind": "frontier",
                        "cell_name": name,
                        "path": f"frontier/{name}",
                        "samples": cfg["evaluation_samples"],
                    }
                )
            name = cell_name(m, t, 42)
            rows.append(
                {
                    "kind": "pfg",
                    "cell_name": name,
                    "path": f"pfg/{name}",
                    "samples": cfg["pfg_samples"],
                    "is_ping": m == "ping",
                }
            )
        name = cell_name(m, None, 42)
        rows.append(
            {"kind": "snapshot", "cell_name": name, "path": f"snapshot/{m}", "model": m}
        )
    for m in ("ping", "coba"):
        name = cell_name(m, 1.0, 42)
        for scale in cfg["scales"]:
            rows.append(
                {
                    "kind": "scale",
                    "cell_name": name,
                    "path": f"win_scale/s{scale:g}/{name}",
                    "samples": cfg["evaluation_samples"],
                    "scale": scale,
                    "label": f"{m}@rt1hz",
                }
            )
    return rows




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


def inference_request(training, job):
    snapshot = job["kind"] == "snapshot"
    request = {
        "checkpoint_role": CHECKPOINT_ROLE,
        "input": "snapshot" if snapshot else "dataset",
        "t_ms": 400.0 if snapshot else training["t_ms"],
        "input_rate_hz": training["input_rate"],
        "samples": job.get("samples"),
        "batch_size": 1 if snapshot else 64,
        "subset_seed": 42,
        "encoder_seed": 20260415,
        "digit": 0,
        "sample": 0,
        "observables": ["spikes"] if snapshot else [],
        "products": [],
        "scale": job.get("scale", 1),
    }
    if job["kind"] == "pfg" and job["is_ping"]:
        request["products"] = ["population", "rasters"]
    if job["kind"] == "scale":
        request["products"] = ["rates"]
    return request
