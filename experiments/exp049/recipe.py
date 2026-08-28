"""Retained TR-05 endpoint recipe; all training belongs to exp022."""

from experiments.exp022 import training_run_cell, training_run_values
from experiments.helpers.checkpoints import checkpoint_policy
from experiments.helpers.datasets import MNIST_REDUCED_EVAL_SAMPLES

SLUG = "exp049"

ANALYSIS_PURPOSE = "endpoint_dynamics"

CHECKPOINT_POLICY = checkpoint_policy(ANALYSIS_PURPOSE)

CHECKPOINT_ROLE = CHECKPOINT_POLICY["role"]

MAX_SAMPLES = 7000

EPOCHS = 50

T_MS = 200.0

DT_TRAIN = 0.1
N_E, N_I = 1024, 256

SEEDS: list[int] = list(training_run_values("TR-05", "seed"))

CONDITIONS: dict[str, dict] = {
    "frozen_ping": {
        "label": "Frozen PING (control)",
    },
    "trainable_ping_init": {
        "label": "Trainable, PING init",
    },
    "trainable_zero_init": {
        "label": "Trainable, zero init",
    },
    "trainable_small_init": {
        "label": "Trainable, small seed init",
    },
}

COND_ORDER = list(training_run_values("TR-05", "tag"))
if set(COND_ORDER) != set(CONDITIONS):
    raise ValueError("TR-05 condition contract drift")

COMMON_RECIPE: dict[str, str] = {
    "--v-grad-dampen": "1000",
    "--w-in": "0.9",
    "--w-in-initial-zero-fraction": "0.95",
    "--readout": "mem-mean",
    "--surrogate-slope": "1",
    "--readout-w-init-mean": "1.12060546875",
    "--readout-w-init-std": "0.8349609375",
    "--lr": "0.0004",
    "--batch-size": "256",
}

F_GAMMA_BAND_HZ: tuple[float, float] = (5.0, 150.0)
EVAL_MAX_SAMPLES = MNIST_REDUCED_EVAL_SAMPLES
WEIGHT_ARRAYS = tuple(
    f"W_{direction}_1_{state}"
    for direction in ("ei", "ie")
    for state in ("init", "trained")
)
SNAPSHOT_ARRAYS = ("dt", "n_e", "n_i", "label", "spk_e", "spk_i")
PAYLOADS = {
    "infer": ("metrics.json", "pop_traces.npz"),
    "weights_dump": ("weights_dump.npz",),
    "snapshot": ("snapshot.npz",),
}
ARRAYS = {
    "pop_traces.npz": ("dt", "pop_e"),
    "weights_dump.npz": WEIGHT_ARRAYS,
    "snapshot.npz": SNAPSHOT_ARRAYS,
}
FIGURES = tuple(
    name + "." + ext
    for name, exts in [("card__" + c, ("png", "pdf")) for c in COND_ORDER]
    + [("weights__" + c, ("svg", "pdf")) for c in COND_ORDER]
    + [
        (n, ("svg", "pdf"))
        for n in (
            "attractor_ei",
            "training_curves",
            "phase_portrait",
            "acc_rate_trajectory",
        )
    ]
    for ext in exts
)


def cell_name(cond, seed):
    return training_run_cell("TR-05", tag=cond, seed=seed)["name"]


def bank_cells():
    return [
        {"cell_name": cell_name(c, s), "condition": c, "seed": s}
        for c in COND_ORDER
        for s in SEEDS
    ]


def configuration(*, smoke=False):
    return {
        "schema": "exp049.recipe/v1",
        "profile": "smoke" if smoke else "production",
        "checkpoint_policy": CHECKPOINT_POLICY,
        "evaluation_samples": 100 if smoke else EVAL_MAX_SAMPLES,
        "seeds": SEEDS,
        "conditions": COND_ORDER,
        "snapshot_seed": SEEDS[0],
        "sample_index": 0,
    }


def jobs(cfg):
    return [
        {
            **cell,
            "kind": kind,
            "path": f"{kind}/{cell['cell_name']}",
            **(
                {"samples": cfg["evaluation_samples"]}
                if kind == "infer"
                else {"sample_index": cfg["sample_index"]}
                if kind == "snapshot"
                else {}
            ),
        }
        for cell in bank_cells()
        for kind in ("infer", "weights_dump", "snapshot")
        if kind != "snapshot" or cell["seed"] == cfg["snapshot_seed"]
    ]


def inference_args(train, checkpoint, output, job):
    args = [
        "dump-weights" if job["kind"] == "weights_dump" else "sim",
        "--load-config",
        str(train / "config.json"),
        "--load-weights",
        str(checkpoint),
        "--out-dir",
        str(output),
    ]
    if job["kind"] != "weights_dump":
        args += ["--infer"]
        if job["kind"] == "snapshot":
            args += ["--sample-index", str(job["sample_index"])]
        else:
            args += ["--outputs", "pop_traces", "--max-samples", str(job["samples"])]
    fields = [key for name in PAYLOADS[job["kind"]] for key in ARRAYS.get(name, ())]
    args += ["--output-fields", *dict.fromkeys(fields)]
    if job["kind"] != "weights_dump":
        args += ["--recording-mode", "spikes"]
    return args
