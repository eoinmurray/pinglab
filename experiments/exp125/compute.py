"""Train a fully connected LIF classifier locally; complete one compute run."""
import argparse
import hashlib
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import numpy as np
import sklearn
import snntorch as snn
import torch
from experiments.exp125.recipe import CONFIG, SLUG
from pingstore.contracts import write_json_atomic
from pingstore.stages import stage_run
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from snntorch import surrogate
from torch import nn


class Classifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input = nn.Linear(cfg["inputs"], cfg["hidden"])
        self.output = nn.Linear(cfg["hidden"], cfg["outputs"])
        options = dict(beta=cfg["beta"], threshold=cfg["threshold"],
                       spike_grad=surrogate.fast_sigmoid(slope=cfg["surrogate_slope"]),
                       reset_mechanism="subtract", reset_delay=True)
        self.hidden_lif = snn.Leaky(**options)
        self.output_lif = snn.Leaky(**options)
        nn.init.constant_(self.input.bias, 0.1)
        nn.init.constant_(self.output.bias, 0.1)

    def forward(self, images, generator):
        shape = (len(images),)
        hidden = images.new_zeros((*shape, self.cfg["hidden"]))
        output = images.new_zeros((*shape, self.cfg["outputs"]))
        counts = torch.zeros_like(output)
        probability = images * (self.cfg["peak_input_rate_hz"] * self.cfg["dt_ms"] / 1000)
        for _ in range(self.cfg["steps"]):
            events = (torch.rand(images.shape, generator=generator) < probability).float()
            spikes, hidden = self.hidden_lif(self.input(events), hidden)
            out_spikes, output = self.output_lif(self.output(spikes), output)
            counts = counts + out_spikes
        return counts


def weights(model):
    return {name: value.detach().numpy().copy() for name, value in model.state_dict().items()}


@torch.no_grad()
def evaluate(model, images, labels, indices, seed, cfg):
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    losses, predictions = [], []
    for part in torch.tensor_split(torch.as_tensor(indices),
                                   max(1, int(np.ceil(len(indices) / cfg["evaluation_batch_size"])))):
        scores = model(images[part], generator)
        losses.extend(nn.functional.cross_entropy(scores, labels[part], reduction="none").tolist())
        predictions.extend(scores.argmax(1).tolist())
    predictions = np.asarray(predictions)
    return {"loss": float(np.mean(losses)),
            "accuracy": float(np.mean(predictions == labels[indices].numpy())),
            "samples": len(indices)}, predictions


def compute(cpu_threads):
    cfg = dict(CONFIG)
    torch.set_num_threads(cpu_threads)
    torch.manual_seed(cfg["seed"])
    dataset = load_digits()
    raw = np.asarray(dataset.data, dtype=np.uint8)
    target = np.asarray(dataset.target, dtype=np.int64)
    indices = np.arange(len(target))
    remaining, test = train_test_split(indices, test_size=cfg["test_fraction"],
                                      stratify=target, random_state=cfg["seed"])
    train, validation = train_test_split(remaining, test_size=cfg["validation_fraction_of_remainder"],
                                        stratify=target[remaining], random_state=cfg["seed"] + 1)
    images = torch.from_numpy(raw.astype(np.float32) / 16)
    labels = torch.from_numpy(target)
    model = Classifier(cfg)
    initial = weights(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    order_generator = torch.Generator().manual_seed(cfg["seed"] + 2)
    encoding_generator = torch.Generator().manual_seed(cfg["seed"] + 3)
    with stage_run(REPO, SLUG, "compute", configuration=cfg) as run:
        np.savez_compressed(run.export / "dataset.npz", pixels=raw, labels=target,
                            train_indices=train, validation_indices=validation, test_indices=test)
        np.savez_compressed(run.export / "initial_weights.npz", **initial)
        history, best_loss, selected_epoch = [], float("inf"), 0
        started = time.perf_counter()
        for epoch in range(cfg["epochs"] + 1):
            gradient_norms = []
            if epoch:
                model.train()
                permutation = torch.randperm(len(train), generator=order_generator)
                for positions in permutation.split(cfg["batch_size"]):
                    part = train[positions.numpy()]
                    optimizer.zero_grad(set_to_none=True)
                    scores = model(images[part], encoding_generator)
                    loss = nn.functional.cross_entropy(scores, labels[part])
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite training loss")
                    loss.backward()
                    norm = nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
                    gradient_norms.append(float(norm))
                    optimizer.step()
            row = {"epoch": epoch}
            for split, selected in (("train", train), ("validation", validation)):
                row[split], _ = evaluate(model, images, labels, selected,
                                         cfg["seed"] + (10 if split == "train" else 11), cfg)
            if gradient_norms:
                row["gradient_norm_mean"] = float(np.mean(gradient_norms))
            history.append(row)
            if row["validation"]["loss"] < best_loss:
                best_loss = row["validation"]["loss"]
                selected_epoch = epoch
                np.savez_compressed(run.export / "selected_weights.npz", **weights(model))
            print(f"epoch {epoch:02d}: train {row['train']['loss']:.4f} / {100*row['train']['accuracy']:.1f}%; "
                  f"validation {row['validation']['loss']:.4f} / {100*row['validation']['accuracy']:.1f}%", flush=True)
        training_seconds = time.perf_counter() - started
        np.savez_compressed(run.export / "final_weights.npz", **weights(model))
        with np.load(run.export / "selected_weights.npz") as saved:
            model.load_state_dict({name: torch.from_numpy(saved[name].copy()) for name in saved.files})
        inference_started = time.perf_counter()
        test_metrics, predictions = evaluate(model, images, labels, test, cfg["seed"] + 12, cfg)
        test_inference_seconds = time.perf_counter() - inference_started
        np.savez_compressed(run.export / "test_predictions.npz", indices=test, labels=target[test], predictions=predictions)
        write_json_atomic(run.export / "learning.json", {"epochs": history, "selected_epoch": selected_epoch, "test": test_metrics})
        run.record["execution"].update({
            "device": "cpu", "cpu_threads": cpu_threads, "machine": platform.machine(),
            "training_seconds": training_seconds, "training_and_test_seconds": time.perf_counter() - started,
            "test_inference_seconds": test_inference_seconds,
            "versions": {"torch": torch.__version__, "snntorch": snn.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__},
            "dataset": {"source": "sklearn.datasets.load_digits", "pixel_label_sha256": hashlib.sha256(raw.tobytes() + target.astype('<i8').tobytes()).hexdigest()},
        })
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    compute(args.cpu_threads)
