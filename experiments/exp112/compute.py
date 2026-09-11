"""Train and test one of the four committed exp112 conditions."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp112 import recipe
from experiments.helpers.run_cli import run_cli
from pingstore.contracts import PingstoreError, load_json
from pingstore.stages import stage_run


def _array_digest(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return "sha256:" + digest.hexdigest()


def dataset_identity() -> dict:
    """Fingerprint the exact source arrays and deterministic exp112 split."""
    from sklearn.model_selection import train_test_split
    from torchvision import datasets

    train = datasets.MNIST(root="/tmp/mnist", train=True, download=False)
    test = datasets.MNIST(root="/tmp/mnist", train=False, download=False)
    train_images = train.data.numpy()
    train_labels = train.targets.numpy().astype(np.int64)
    test_images = test.data.numpy()
    test_labels = test.targets.numpy().astype(np.int64)
    if len(train_labels) != recipe.MNIST_TRAIN_SAMPLES:
        raise PingstoreError("official MNIST training partition has unexpected size")
    if len(test_labels) != recipe.MNIST_TEST_SAMPLES:
        raise PingstoreError("official MNIST test partition has unexpected size")

    raster_matches = np.flatnonzero(test_labels == recipe.RASTER_DIGIT)
    if len(raster_matches) <= recipe.RASTER_SAMPLE_WITHIN_CLASS:
        raise PingstoreError("requested illustrative MNIST test example is absent")
    raster_index = int(raster_matches[recipe.RASTER_SAMPLE_WITHIN_CLASS])

    selected = np.random.RandomState(recipe.MNIST_SUBSET_SEED).choice(
        len(train_labels), recipe.TRAINING_POOL_SAMPLES, replace=False
    )
    optimizer, validation = train_test_split(
        selected,
        test_size=recipe.VALIDATION_SAMPLES,
        random_state=recipe.MNIST_SPLIT_SEED,
        stratify=train_labels[selected],
    )
    if len(optimizer) != recipe.OPTIMIZER_TRAIN_SAMPLES:
        raise PingstoreError("MNIST optimizer split has unexpected size")
    return {
        "schema": "exp112.dataset-identity/v1",
        "official_train_images": _array_digest(train_images),
        "official_train_labels": _array_digest(train_labels),
        "selected_pool_indices_in_order": _array_digest(selected.astype(np.int64)),
        "optimizer_indices_in_order": _array_digest(optimizer.astype(np.int64)),
        "validation_indices_in_order": _array_digest(validation.astype(np.int64)),
        "official_test_images": _array_digest(test_images),
        "official_test_labels": _array_digest(test_labels),
        "illustrative_raster_digit": recipe.RASTER_DIGIT,
        "illustrative_raster_sample_within_class": recipe.RASTER_SAMPLE_WITHIN_CLASS,
        "illustrative_raster_raw_test_index": raster_index,
        "illustrative_raster_image": _array_digest(test_images[raster_index]),
    }


def _validate_training(config: dict, metrics: dict, case: dict) -> None:
    expected = {
        "model": "ping",
        "dataset": "mnist",
        "max_samples": recipe.TRAINING_POOL_SAMPLES,
        "epochs": recipe.EPOCHS,
        "batch_size": recipe.BATCH_SIZE,
        "n_hidden": recipe.N_EXCITATORY,
        "n_in": recipe.N_INPUT,
        "n_inh": recipe.N_INHIBITORY,
        "n_out": recipe.N_OUTPUT,
        "dt": recipe.DT_MS,
        "t_ms": recipe.PRESENTATION_MS,
        "refractory_e_ms": recipe.REFRACTORY_E_MS,
        "refractory_i_ms": recipe.REFRACTORY_I_MS,
        "refractory_policy": recipe.REFRACTORY_POLICY,
        "tau_ampa_ms": recipe.TAU_AMPA_MS,
        "tau_gaba_ms": recipe.TAU_GABA_MS,
        "input_rate": recipe.INPUT_RATE_HZ,
        "ei_strength": case["ei_strength"],
        "ei_ratio": recipe.EI_RATIO,
        "w_in": [
            recipe.W_IN_SUMMED_PARENT_MEAN,
            recipe.W_IN_SUMMED_PARENT_MEAN * 0.1,
        ],
        "w_in_initial_zero_fraction": recipe.W_IN_INITIAL_ZERO_FRACTION,
        "recurrent_initial_zero_fraction": recipe.RECURRENT_INITIAL_ZERO_FRACTION,
        "v_grad_dampen": case["v_grad_dampen"],
        "surrogate_slope": recipe.SURROGATE_SLOPE,
        "grad_clip": recipe.GRADIENT_CLIP_NORM,
        "lr": recipe.LEARNING_RATE,
        "weight_decay": recipe.WEIGHT_DECAY,
        "dales_law": recipe.DALES_LAW,
        "readout_mode": recipe.READOUT,
        "readout_w_init_mean": recipe.READOUT_W_INIT_MEAN,
        "readout_w_init_std": recipe.READOUT_W_INIT_STD,
        "signed_readout": False,
        "readout_bias": False,
        "trainable_w_ee": False,
        "trainable_w_ei": False,
        "trainable_w_ie": False,
        "trainable_w_ii": False,
        "state_clamp": False,
        "train_leak": False,
        "adaptive_threshold": False,
        "fr_reg_upper_target_hz": recipe.RATE_PENALTY_TARGET_HZ,
        "fr_reg_upper_strength": recipe.RATE_PENALTY_STRENGTH,
        "seed": recipe.SEED,
    }
    observed = config
    wrong = {
        key: (observed.get(key), value)
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if wrong:
        raise PingstoreError(f"resolved training configuration differs: {wrong}")
    split = observed.get("dataset_split", {})
    split_expected = {
        "optimizer_train_samples": recipe.OPTIMIZER_TRAIN_SAMPLES,
        "validation_samples": recipe.VALIDATION_SAMPLES,
        "official_test_samples": recipe.MNIST_TEST_SAMPLES,
        "official_test_used_during_training": False,
        "split_seed": recipe.MNIST_SPLIT_SEED,
    }
    if any(split.get(key) != value for key, value in split_expected.items()):
        raise PingstoreError("resolved MNIST split differs from the exp112 contract")
    if len(metrics.get("epochs", [])) != recipe.EPOCHS:
        raise PingstoreError("training history did not complete all committed epochs")
    if config.get("max_samples") != recipe.TRAINING_POOL_SAMPLES:
        raise PingstoreError(
            "tool config does not identify the committed training pool"
        )


def _validate_test(metrics: dict) -> None:
    config = metrics.get("config", {})
    correct = metrics.get("n_correct")
    if (
        config.get("dataset") != "mnist"
        or config.get("evaluation_partition") != "official_mnist_test"
        or config.get("evaluation_samples") != recipe.MNIST_TEST_SAMPLES
        or config.get("max_samples") != recipe.MNIST_TEST_SAMPLES
        or config.get("seed") != recipe.SEED
        or Path(str(config.get("load_weights", ""))).name != "weights_final.pth"
        or metrics.get("n_total") != recipe.MNIST_TEST_SAMPLES
        or not isinstance(correct, int)
        or not 0 <= correct <= recipe.MNIST_TEST_SAMPLES
    ):
        raise PingstoreError(
            "inference did not cover the complete official MNIST test set"
        )


def _validate_raster(path: Path) -> None:
    if not path.is_file():
        raise PingstoreError("illustrative inference did not produce a spike recording")
    with np.load(path, allow_pickle=False) as arrays:
        required = {"spk_e", "spk_i", "dt", "n_e", "n_i", "label"}
        if not required.issubset(arrays.files):
            raise PingstoreError("illustrative spike recording lacks required arrays")
        e = arrays["spk_e"]
        i = arrays["spk_i"]
        if e.ndim == 3:
            e = e[:, 0, :]
        if i.ndim == 3:
            i = i[:, 0, :]
        expected_steps = round(recipe.PRESENTATION_MS / recipe.DT_MS)
        if (
            e.shape != (expected_steps, recipe.N_EXCITATORY)
            or i.shape != (expected_steps, recipe.N_INHIBITORY)
            or int(arrays["n_e"]) != recipe.N_EXCITATORY
            or int(arrays["n_i"]) != recipe.N_INHIBITORY
            or not np.isclose(float(arrays["dt"]), recipe.DT_MS)
            or int(arrays["label"]) != recipe.RASTER_DIGIT
        ):
            raise PingstoreError("illustrative spike recording differs from exp112")


def compute(index: int, *, run_id: str | None = None) -> str:
    case = recipe.case_for_shard(index)
    cfg = recipe.configuration(case)
    with stage_run(
        REPO, recipe.SLUG, "compute", run_id=run_id, configuration=cfg
    ) as run:
        run.record["execution"]["dataset_identity"] = dataset_identity()
        training = run.scratch / "training"
        testing = run.scratch / "official-test"
        raster = run.scratch / "illustrative-raster"
        train_args = recipe.training_args(case, training)
        test_args = recipe.test_args(training, testing)
        raster_args = recipe.raster_args(training, raster)
        run.record["execution"]["simulator"] = "tools/snnsim/tool.py legacy executor"
        run.record["execution"]["training_arguments"] = train_args
        run_cli(train_args, no_sync=True)

        training_config = load_json(training / "config.json")
        training_metrics = load_json(training / "metrics.json")
        _validate_training(training_config, training_metrics, case)
        if not (training / "weights_final.pth").is_file():
            raise PingstoreError("training did not produce the final checkpoint")

        run.record["execution"]["test_arguments"] = test_args
        run_cli(test_args, no_sync=True)
        test_metrics = load_json(testing / "metrics.json")
        _validate_test(test_metrics)
        run.record["execution"]["raster_arguments"] = raster_args
        run_cli(raster_args, no_sync=True)
        _validate_raster(raster / "recording.npz")
        run.record["execution"]["resolved_training_config"] = training_config
        shutil.copyfile(training / "metrics.json", run.export / "training_metrics.json")
        shutil.copyfile(testing / "metrics.json", run.export / "test_metrics.json")
        shutil.copyfile(
            training / "weights_final.pth", run.export / "weights_final.pth"
        )
        shutil.copyfile(raster / "recording.npz", run.export / "spikes.npz")
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-index", required=True, type=int, choices=range(len(recipe.CASES))
    )
    parser.add_argument(
        "--run-id", help="fresh v4 compute identity reserved before scheduler dispatch"
    )
    args = parser.parse_args()
    try:
        compute(args.shard_index, run_id=args.run_id)
    except (OSError, KeyError, ValueError, PingstoreError) as exc:
        parser.exit(1, f"exp112 compute: {exc}\n")


if __name__ == "__main__":
    main()
