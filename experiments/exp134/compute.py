"""Generate paired recordings for frozen-network membrane interventions.

Run compute.py --source TRAINING_RUN --image-source IMAGE_RUN, optionally
setting --images-per-digit, --batch-images and --run-id. Inputs are validated
Pingstore runs; the recipe resolves all scientific settings before execution.

A custom batched loop reuses encodings and frozen weights across conditions.
The run retains recordings, one circuit diagram, bundles and native start
snapshots. Analysis and presentation are independently invoked stages."""

import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import numpy as np
import torch
from experiments.exp134 import recipe
from pingstore.contracts import file_sha256, load_json
from pingstore.stages import source_run, stage_run
from snnlab.sim import Performance, build, encode_images_poisson
from experiments.exp134.custom.snapshots import snapshot_start


def training_input(source, cfg):
    unit = source.unit(cfg["checkpoint_unit"])
    record = (
        load_json(unit / "metrics.json")
        .get("checkpoints", {})
        .get(cfg["checkpoint_role"])
    )
    filename = cfg["checkpoint_file"]
    if not isinstance(record, dict) or record.get("filename") != filename:
        raise ValueError(
            f"Checkpoint role {cfg['checkpoint_role']!r} does not name {filename!r}"
        )
    if Path(filename).name != filename:
        raise ValueError("Checkpoint filename must be a single path component")
    checkpoint = unit / filename
    if (
        record.get("sha256") != cfg["checkpoint_sha256"]
        or file_sha256(checkpoint) != cfg["checkpoint_sha256"]
    ):
        raise ValueError(f"Checkpoint identity mismatch: {checkpoint}")
    config = json.loads((unit / "config.json").read_text())
    for key, value in cfg["checkpoint_contract"].items():
        if config.get(key) != value:
            raise ValueError(f"unsupported checkpoint setting: {key}")
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if weights["W_ee.1"].count_nonzero() or weights["W_ii.1"].count_nonzero():
        raise ValueError("same-population recurrence must be absent")
    return weights, checkpoint


def image_bank(source):
    definition = source.record["execution"]["configuration"]
    if definition.get("schema") != "exp123.compute/v3":
        raise ValueError("expected exp123's balanced shared sweep")
    cfg = definition["sweep"]
    if cfg["partition"] != "official_mnist_test" or cfg["image_selection_seed"] != 123:
        raise ValueError("incompatible image source")
    images = {}
    for index in cfg["image_indices"]:
        with np.load(
            source.unit(f"image{index:05d}-rate25") / "input.npz", allow_pickle=False
        ) as data:
            pixels, label = data["pixels"].copy(), int(data["label"])
        if (
            pixels.shape != (28, 28)
            or not np.isfinite(pixels).all()
            or pixels.min() < 0
            or pixels.max() > 1
            or not 0 <= label < 10
        ):
            raise ValueError("invalid normalized image")
        images[index] = (pixels, label)
    if len(images) != 500 or not np.array_equal(
        np.bincount([v[1] for v in images.values()], minlength=10), [50] * 10
    ):
        raise ValueError("expected 500 distinct images, 50 per digit")
    return images


def encode(pixels, index, cfg):
    pixels = torch.from_numpy(pixels).reshape(1, cfg["n_input"]).float()
    return torch.cat(
        [
            encode_images_poisson(
                pixels,
                round(cfg["duration_ms"] / cfg["dt_ms"]),
                cfg["dt_ms"],
                cfg["input_rate_hz"],
                generator=torch.Generator().manual_seed(1000 * index + seed),
            )
            for seed in cfg["seeds"]
        ],
        dim=1,
    )


def frozen_model(condition, weights, cfg):
    bundle = recipe.author_network(condition, cfg)
    built = build(
        bundle=bundle,
        device=cfg["device"],
        seed=cfg["model_seed"],
        performance=Performance(**cfg["performance"]),
    )
    model = built.model
    with torch.inference_mode():
        for name, target in model.parameter_map().items():
            value = weights[cfg["parameter_mapping"][name]]
            if (
                value.shape != target.shape
                or not torch.isfinite(value).all()
                or (value < 0).any()
            ):
                raise ValueError("checkpoint shape or polarity mismatch")
            target.copy_(value)
    return bundle, model, built.metrics["performance"]


def check_weights(model, weights, cfg):
    if any(
        not torch.equal(value, weights[cfg["parameter_mapping"][name]])
        for name, value in model.parameter_map().items()
    ):
        raise ValueError("frozen weights changed")


def compute(
    identity, image_identity, *, run_id=None, images_per_digit=None, batch_images=None
):
    cfg = recipe.configuration(
        images_per_digit=images_per_digit, batch_images=batch_images
    )
    bank = source_run(
        REPO / ".pingstore", identity, stage="compute", experiment="exp022"
    )
    image_source = source_run(
        REPO / ".pingstore", image_identity, stage="compute", experiment="exp123"
    )
    weights, checkpoint = training_input(bank, cfg)
    images = image_bank(image_source)
    cfg = recipe.configuration(
        images, images_per_digit=images_per_digit, batch_images=batch_images
    )
    torch.set_num_threads(cfg["torch_threads"])
    with stage_run(
        REPO,
        recipe.SLUG,
        "compute",
        inputs={"trained_network": bank, "images": image_source},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        run.record["execution"]["software"] = dict(
            snnlab=version("snnlab"), torch=torch.__version__
        )
        encoded = {}
        hashes = {}
        for index in cfg["image_indices"]:
            pixels, _ = images[index]
            spikes = encode(pixels, index, cfg)
            encoded[index] = spikes.numpy().astype(bool)
            hashes[str(index)] = {
                str(seed): hashlib.sha256(spikes[:, b].numpy().tobytes()).hexdigest()
                for b, seed in enumerate(cfg["seeds"])
            }
            # Encodings are generated once, then reused for every intervention.
            unit = run.export / f"image{index:05d}"
            unit.mkdir()
        run.record["scientific_artifacts"] = {
            "snapshot_stages": {
                f"{condition['condition_id']}-execution": ["start"]
                for condition in cfg["conditions"]
            }
        }
        performance = {}
        for condition_number, condition in enumerate(cfg["conditions"], start=1):
            name = condition["condition_id"]
            bundle, model, performance[name] = frozen_model(condition, weights, cfg)
            bundle.write(run.export / f"{name}.bundle", visualise=False)
            # Parameter sweeps share one circuit topology.
            if name == "baseline":
                bundle.visualise(run.export / "network.svg", view="circuit")
            snapshot_start(model, bundle, run.export / f"{name}-execution", cfg)
            with torch.inference_mode():
                for start in range(0, len(cfg["image_indices"]), cfg["batch_images"]):
                    indices = cfg["image_indices"][start : start + cfg["batch_images"]]
                    batch = torch.from_numpy(
                        np.concatenate([encoded[i] for i in indices], axis=1)
                    ).float()
                    # A fresh call resets all dynamic state; batch lanes are independent.
                    result = model({"image": batch}, diagnostics=True)
                    signals = {
                        key: result.diagnostics[key].cpu().numpy().astype(bool)
                        for key in cfg["recording_signals"]
                    }
                    for j, index in enumerate(indices):
                        selection = slice(
                            j * len(cfg["seeds"]), (j + 1) * len(cfg["seeds"])
                        )
                        np.savez_compressed(
                            run.export / f"image{index:05d}" / f"{name}--spikes.npz",
                            **{
                                key: value[:, selection]
                                for key, value in signals.items()
                            },
                            dt_ms=cfg["dt_ms"],
                            image_index=index,
                            seeds=cfg["seeds"],
                        )
                    del result, signals, batch
                check_weights(model, weights, cfg)
            print(
                f"{name}: complete ({condition_number}/{len(cfg['conditions'])} conditions)",
                flush=True,
            )
            del model, bundle
        if file_sha256(checkpoint) != cfg["checkpoint_sha256"]:
            raise ValueError("checkpoint changed")
        run.record["execution"].update(
            encoding_sha256=hashes,
            performance=performance,
            observations=dict(
                presentations=cfg["presentations"],
                weights_frozen=True,
                paired_across_both_sweeps=True,
            ),
        )
        result = run.run_id
    print(result, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--image-source", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--images-per-digit", type=int)
    parser.add_argument("--batch-images", type=int)
    args = parser.parse_args()
    compute(
        args.source,
        args.image_source,
        run_id=args.run_id,
        images_per_digit=args.images_per_digit,
        batch_images=args.batch_images,
    )
