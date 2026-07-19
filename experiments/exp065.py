"""Experiment 065 — calibrate MNIST evidence loss with a matched vanilla ANN.

The ANN matches the trained PING baseline's visible architecture: 784 inputs,
one 1,024-unit hidden population (the number of excitatory PING cells), and 10
outputs. It is trained on clean grayscale MNIST. Held-out images are then
binarized and their foreground pixels independently retained with probability
q, producing the ANN calibration curve requested in issue #46.

Writing: writings/exp065.typ · record: artifacts/data/exp065/
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import theme  # noqa: E402
from helpers.cli import parse_meta  # noqa: E402
from helpers.datasets import load_mnist_split  # noqa: E402
from helpers.figsave import save_figure  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp065"
CHECKPOINT_ROOT = REPO / "temp" / "experiments" / SLUG / "ann"
MATCHED_CACHE_ROOT = REPO / "temp" / "experiments" / SLUG / "matched_masking"

N_INPUT = 784
N_HIDDEN = 1024
N_CLASSES = 10
SEEDS = (42, 43, 44)
EPOCHS = 15
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
BINARIZE_THRESHOLD = 0.0
MASK_DRAWS = 10
RETENTION_Q = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3,
               0.2, 0.1, 0.05, 0.02, 0.01, 0.0075, 0.005, 0.0025, 0.0)
CHANCE_ACCURACY = 1.0 / N_CLASSES
PING_T_MS = 200.0
MATCHED_Q = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.0)
MATCHED_IMAGES = 100
MATCHED_STREAM_SIZE = 10
MATCHED_RATE_HZ = 25.0
DIAGNOSTIC_Q = (1.0, 0.1, 0.02, 0.005, 0.0)

SCALE = {
    "dataset": "MNIST (full 70,000-sample corpus; stratified 80/20 split)",
    "model": "vanilla ANN, 784→1024→10",
    "hidden_units": N_HIDDEN,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "seeds": len(SEEDS),
    "mask_draws": MASK_DRAWS,
    "retention_levels": len(RETENTION_Q),
    "matched_mask_levels": len(MATCHED_Q),
    "matched_images_per_level": MATCHED_IMAGES,
    "compute": "local",
}


class MatchedANN(nn.Module):
    """One-hidden-layer ANN matched to the PING E-population width."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN),
            nn.ReLU(),
            nn.Linear(N_HIDDEN, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def binarize_foreground(x: torch.Tensor, threshold: float = BINARIZE_THRESHOLD) -> torch.Tensor:
    """Return a float binary image: foreground=1, background=0."""
    return (x > threshold).to(dtype=torch.float32)


def retain_foreground(
    binary: torch.Tensor, q: float, *, generator: torch.Generator,
) -> torch.Tensor:
    """Independently retain each foreground pixel with probability q.

    Sampling at every position is distributionally identical to sampling only
    foreground locations: multiplication makes every background draw a no-op.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must lie in [0, 1], got {q}")
    if q == 1.0:
        return binary.clone()
    if q == 0.0:
        return torch.zeros_like(binary)
    keep = torch.rand(binary.shape, generator=generator) < q
    return binary * keep.to(binary.dtype)


def _device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _checkpoint(seed: int) -> Path:
    return CHECKPOINT_ROOT / f"seed{seed}.pt"


def _checkpoint_config() -> dict:
    return {
        "n_input": N_INPUT,
        "n_hidden": N_HIDDEN,
        "n_classes": N_CLASSES,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
    }


@torch.inference_mode()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> float:
    model.eval()
    correct = 0
    for lo in range(0, len(x), BATCH_SIZE):
        xb = x[lo:lo + BATCH_SIZE].to(device)
        pred = model(xb).argmax(dim=1).cpu()
        correct += int((pred == y[lo:lo + BATCH_SIZE]).sum())
    return correct / len(y)


@torch.inference_mode()
def predictions(model: nn.Module, x: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    chunks = []
    for lo in range(0, len(x), BATCH_SIZE):
        chunks.append(model(x[lo:lo + BATCH_SIZE].to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(chunks)


def train_ann(seed: int, x_train: torch.Tensor, y_train: torch.Tensor, device: torch.device) -> dict:
    """Train one seed, cache its checkpoint, and return its loss trace."""
    torch.manual_seed(seed)
    model = MatchedANN().to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=BATCH_SIZE,
        shuffle=True, generator=generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    losses: list[float] = []
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(xb)
        losses.append(total / len(x_train))
        print(f"  seed {seed} epoch {epoch + 1:02d}/{EPOCHS}: loss={losses[-1]:.4f}")

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.cpu().state_dict(),
        "config": _checkpoint_config(),
        "seed": seed,
        "loss": losses,
    }, _checkpoint(seed))
    return {"seed": seed, "loss": losses}


def load_ann(seed: int, device: torch.device) -> tuple[MatchedANN, dict]:
    checkpoint = torch.load(_checkpoint(seed), map_location="cpu", weights_only=False)
    if checkpoint.get("config") != _checkpoint_config():
        raise RuntimeError(f"stale ANN checkpoint for seed {seed}; run without --skip-training")
    model = MatchedANN()
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device), checkpoint


def calibration(
    models: dict[int, MatchedANN], binary_test: torch.Tensor,
    y_test: torch.Tensor, device: torch.device,
) -> list[dict]:
    """Evaluate paired Bernoulli masks for every q and ANN seed."""
    rows: list[dict] = []
    for qi, q in enumerate(RETENTION_Q):
        per_seed: dict[int, list[float]] = {seed: [] for seed in models}
        retained: list[float] = []
        for draw in range(MASK_DRAWS):
            g = torch.Generator().manual_seed(65_000 + qi * 1_000 + draw)
            masked = retain_foreground(binary_test, q, generator=g)
            denom = float(binary_test.sum())
            retained.append(float(masked.sum()) / denom if denom else 0.0)
            for seed, model in models.items():
                per_seed[seed].append(accuracy(model, masked, y_test, device))
        seed_acc = [float(np.mean(per_seed[s])) for s in sorted(per_seed)]
        rows.append({
            "q": q,
            "masked_probability": 1.0 - q,
            "observed_foreground_retention": float(np.mean(retained)),
            "mean_visible_foreground_pixels": float(binary_test.sum(dim=1).mean()) * float(np.mean(retained)),
            "accuracy": float(np.mean(seed_acc)),
            "accuracy_sem": float(np.std(seed_acc, ddof=1) / math.sqrt(len(seed_acc))),
            "accuracy_by_seed": seed_acc,
            "n_images": len(binary_test),
            "mask_draws": MASK_DRAWS,
        })
        print(f"  q={q:>4.2f}: accuracy={100 * rows[-1]['accuracy']:.2f}%")
    return rows


def chance_bound(rows: list[dict]) -> dict | None:
    """Highest q whose seed-level 95% t interval contains ten-percent chance."""
    t95_df2 = 4.3026527299
    eligible = []
    for row in rows:
        half_width = t95_df2 * row["accuracy_sem"]
        if row["accuracy"] - half_width <= CHANCE_ACCURACY <= row["accuracy"] + half_width:
            eligible.append({**row, "ci95_half_width": half_width})
    return max(eligible, key=lambda r: r["q"]) if eligible else None


def matched_stimuli(
    binary_test: torch.Tensor, y_test: torch.Tensor, q: float,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """The fixed held-out examples and one paired mask draw for a retention q."""
    rng = np.random.default_rng(65_065)
    indices = rng.choice(len(y_test), MATCHED_IMAGES, replace=False)
    selected = binary_test[indices]
    qi = MATCHED_Q.index(q)
    generator = torch.Generator().manual_seed(650_650 + qi)
    return retain_foreground(selected, q, generator=generator), y_test[indices], indices


def _matched_cache(seed: int, q: float) -> Path:
    return MATCHED_CACHE_ROOT / f"seed{seed}" / f"q_{q:g}.json"


def compute_ping_mask_cell(
    seed: int, q: float, masked: torch.Tensor, labels: torch.Tensor,
) -> dict:
    """Evaluate one frozen PING seed on the shared masked-image cell."""
    cache = _matched_cache(seed, q)
    if cache.exists():
        row = json.loads(cache.read_text())
        if row.get("labels") == labels.tolist():
            return row

    import exp048

    train_dir, cfg, _x_test, _y_test = exp048._load_eval(seed=seed)
    w_out = exp048._load_w_out(train_dir)
    tau_out_ms = float(cfg.get("tau_out_ms", 2.0))
    tau_steps = int(round(PING_T_MS / exp048.DT))
    all_predictions = []
    for stream, lo in enumerate(range(0, MATCHED_IMAGES, MATCHED_STREAM_SIZE)):
        pixels = masked[lo:lo + MATCHED_STREAM_SIZE].numpy()
        generator = torch.Generator().manual_seed(650_000 + 100 * seed + stream)
        spike_input = exp048.encode_stream(pixels, PING_T_MS, MATCHED_RATE_HZ, generator)
        spike_e, _ = exp048._run_stream(train_dir, spike_input)
        logits = exp048.sliding_readout(spike_e, w_out, tau_out_ms, window_ms=PING_T_MS)
        n_here = len(pixels)
        ends = np.arange(1, n_here + 1) * tau_steps - 1
        all_predictions.extend(logits.argmax(axis=-1)[ends].tolist())

    pred = np.asarray(all_predictions)
    truth = labels.numpy()
    row = {
        "q": q,
        "train_seed": seed,
        "input_rate_hz": MATCHED_RATE_HZ,
        "n_correct": int((pred == truth).sum()),
        "n_total": len(truth),
        "accuracy": float((pred == truth).mean()),
        "labels": truth.tolist(),
        "predictions": pred.tolist(),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(row, indent=2) + "\n")
    return row


def confusion(labels: list[int], preds: list[int]) -> list[list[int]]:
    matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    np.add.at(matrix, (np.asarray(labels), np.asarray(preds)), 1)
    return matrix.tolist()


def matched_masking_rows(
    models: dict[int, MatchedANN], binary_test: torch.Tensor,
    y_test: torch.Tensor, device: torch.device,
) -> tuple[list[dict], dict[float, torch.Tensor]]:
    """Matched ANN–PING evaluation on identical examples and mask draws."""
    rows = []
    stimuli = {}
    for q in MATCHED_Q:
        masked, labels, indices = matched_stimuli(binary_test, y_test, q)
        stimuli[q] = masked
        ann_cells = []
        ping_cells = []
        for seed, model in models.items():
            ann_pred = predictions(model, masked, device)
            ann_cells.append({
                "seed": seed,
                "accuracy": float((ann_pred == labels.numpy()).mean()),
                "predictions": ann_pred.tolist(),
            })
            ping_cells.append(compute_ping_mask_cell(seed, q, masked, labels))

        ann_values = np.array([c["accuracy"] for c in ann_cells])
        ping_values = np.array([c["accuracy"] for c in ping_cells])
        repeated_labels = labels.tolist() * len(SEEDS)
        rows.append({
            "q": q,
            "observed_foreground_retention": float(masked.sum() / binary_test[indices].sum())
            if float(binary_test[indices].sum()) else 0.0,
            "mean_visible_foreground_pixels": float(masked.sum(dim=1).mean()),
            "n_images": MATCHED_IMAGES,
            "test_indices": indices.tolist(),
            "ann_accuracy": float(ann_values.mean()),
            "ann_accuracy_sem": float(ann_values.std(ddof=1) / math.sqrt(len(ann_values))),
            "ping_accuracy": float(ping_values.mean()),
            "ping_accuracy_sem": float(ping_values.std(ddof=1) / math.sqrt(len(ping_values))),
            "ann_by_seed": ann_cells,
            "ping_by_seed": ping_cells,
            "ann_confusion": confusion(
                repeated_labels, [p for cell in ann_cells for p in cell["predictions"]],
            ),
            "ping_confusion": confusion(
                repeated_labels, [p for cell in ping_cells for p in cell["predictions"]],
            ),
        })
        print(f"  matched q={q:g}: ANN={100 * rows[-1]['ann_accuracy']:.1f}% "
              f"PING={100 * rows[-1]['ping_accuracy']:.1f}%")
    return rows, stimuli


def plot_calibration(rows: list[dict], bound: dict | None, out: Path) -> None:
    theme.set_paper_mode()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    q = np.array([r["q"] for r in rows])
    acc = 100 * np.array([r["accuracy"] for r in rows])
    sem = 100 * np.array([r["accuracy_sem"] for r in rows])
    ax.plot(q, acc, color=theme.INK_BLACK, marker="o", lw=1.8)
    ax.fill_between(q, acc - sem, acc + sem, color=theme.INK_BLACK, alpha=0.15, linewidth=0)
    ax.axhline(100 * CHANCE_ACCURACY, color=theme.DEEP_RED, ls="--", lw=1.2,
               label="chance (10%)")
    if bound is not None:
        ax.axvline(bound["q"], color=theme.DEEP_RED, ls=":", lw=1.2,
                   label=f"chance bound q={bound['q']:.2g}")
    ax.set(xlabel="foreground retention probability q", ylabel="P(correct) (%)",
           xlim=(0, 1), ylim=(0, 101))
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out)
    plt.close(fig)


def plot_matched_masking(rows: list[dict], out: Path) -> None:
    theme.set_paper_mode()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    q = np.array([r["q"] for r in rows])
    for key, sem_key, label, color, marker in (
        ("ann_accuracy", "ann_accuracy_sem", "matched ANN", theme.INK_BLACK, "o"),
        ("ping_accuracy", "ping_accuracy_sem", "frozen PING", theme.DEEP_RED, "s"),
    ):
        values = 100 * np.array([r[key] for r in rows])
        sem = 100 * np.array([r[sem_key] for r in rows])
        ax.plot(q, values, color=color, marker=marker, lw=1.8, label=label)
        ax.fill_between(q, values - sem, values + sem, color=color,
                        alpha=0.12, linewidth=0)
    ax.axhline(100 * CHANCE_ACCURACY, color=theme.GREY_MID, ls="--", lw=1.2,
               label="chance (10%)")
    ax.set(xlabel="foreground retention probability q", ylabel="P(correct) (%)",
           xlim=(0, 1), ylim=(0, 101))
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out)
    plt.close(fig)


def plot_diagnostics(rows: list[dict], stimuli: dict[float, torch.Tensor], out: Path) -> None:
    theme.set_paper_mode()
    fig, axes = plt.subplots(len(DIAGNOSTIC_Q), 3, figsize=(6.5, 5.5),
                             gridspec_kw={"width_ratios": (1.8, 1, 1)})
    for ri, q in enumerate(DIAGNOSTIC_Q):
        row = next(r for r in rows if r["q"] == q)
        strip = np.concatenate([stimuli[q][i].reshape(28, 28).numpy() for i in range(5)], axis=1)
        axes[ri, 0].imshow(strip, cmap="gray", vmin=0, vmax=1)
        axes[ri, 0].set_title(f"q={q:g} · {row['mean_visible_foreground_pixels']:.1f} pixels")
        axes[ri, 0].axis("off")
        for ci, (key, title) in enumerate((("ann_confusion", "ANN"),
                                           ("ping_confusion", "PING")), start=1):
            matrix = np.asarray(row[key], dtype=float)
            matrix /= np.maximum(matrix.sum(axis=1, keepdims=True), 1)
            axes[ri, ci].imshow(matrix, cmap="Greys", vmin=0, vmax=1)
            axes[ri, ci].set_title(title)
            axes[ri, ci].set_xticks([])
            axes[ri, ci].set_yticks([])
            axes[ri, ci].set_ylabel("true")
            if ri == len(DIAGNOSTIC_Q) - 1:
                axes[ri, ci].set_xlabel("predicted")
    fig.tight_layout()
    save_figure(fig, out, formats=("png",))
    plt.close(fig)


def main() -> None:
    meta = parse_meta(sys.argv)
    if meta.plot_only:
        raise SystemExit("exp065 does not yet support --plot-only; use --skip-training")

    start = time.monotonic()
    run_id = next_run_id(SLUG)
    device = _device()
    theme.set_paper_mode()
    theme.apply()
    print(f"run_id={run_id} device={device} architecture={N_INPUT}→{N_HIDDEN}→{N_CLASSES}")

    x_tr, x_te, y_tr, y_te = load_mnist_split()
    x_train = torch.from_numpy(x_tr)
    y_train = torch.from_numpy(y_tr)
    x_test = torch.from_numpy(x_te)
    y_test = torch.from_numpy(y_te)

    if not meta.skip_training:
        for seed in SEEDS:
            if meta.only_missing and _checkpoint(seed).exists():
                print(f"[skip] checkpoint seed {seed}")
            else:
                train_ann(seed, x_train, y_train, device)

    models: dict[int, MatchedANN] = {}
    checkpoints: dict[int, dict] = {}
    for seed in SEEDS:
        models[seed], checkpoints[seed] = load_ann(seed, device)

    clean_accuracy = {str(seed): accuracy(model, x_test, y_test, device)
                      for seed, model in models.items()}
    binary_test = binarize_foreground(x_test)
    binary_accuracy = {str(seed): accuracy(model, binary_test, y_test, device)
                       for seed, model in models.items()}
    print("evaluating paired masking curve")
    rows = calibration(models, binary_test, y_test, device)
    bound = chance_bound(rows)
    print("evaluating matched ANN–PING foreground masking")
    matched_rows, matched_images = matched_masking_rows(
        models, binary_test, y_test, device,
    )

    with published_run(SLUG, run_id, scale=SCALE, skip_training=meta.skip_training) as (_, figures):
        plot_calibration(rows, bound, figures / "ann_masking_calibration")
        plot_matched_masking(matched_rows, figures / "matched_masking")
        plot_diagnostics(matched_rows, matched_images, figures / "masking_diagnostics")
        payload = {
            "status": "protocol complete",
            "config": {
                **_checkpoint_config(),
                "seeds": list(SEEDS),
                "binarize_threshold": BINARIZE_THRESHOLD,
                "mask_draws": MASK_DRAWS,
                "retention_q": list(RETENTION_Q),
                "matched_q": list(MATCHED_Q),
                "matched_images": MATCHED_IMAGES,
                "matched_rate_hz": MATCHED_RATE_HZ,
                "matched_presentation_ms": PING_T_MS,
            },
            "dataset": {"train_samples": len(x_train), "test_samples": len(x_test)},
            "parameter_count": sum(p.numel() for p in next(iter(models.values())).parameters()),
            "clean_accuracy_by_seed": clean_accuracy,
            "binary_accuracy_by_seed": binary_accuracy,
            "mean_original_foreground_pixels": float(binary_test.sum(dim=1).mean()),
            "chance_accuracy": CHANCE_ACCURACY,
            "chance_bound": bound,
            "masking_curve": rows,
            "matched_masking": {
                "protocol": "identical held-out examples and Bernoulli masks for ANN and PING",
                "rows": matched_rows,
            },
        }
        write_numbers(figures, run_id=run_id,
                      duration_s=time.monotonic() - start, payload=payload)
        print(f"wrote all protocol figures and {figures / 'numbers.json'}")


if __name__ == "__main__":
    main()
