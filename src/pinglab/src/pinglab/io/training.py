"""Training utilities: input encoding, epoch training and evaluation for SNN training."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def encode_rate(
    image: "torch.Tensor",
    *,
    T_steps: int,
    n_total: int,
    n_input: int,
    scale: float = 1.0,
) -> "torch.Tensor":
    """Encode a single image as a constant-rate current injection.

    Pixel intensities are scaled and broadcast over all T timesteps,
    injecting current only into the first n_input neurons. The remaining
    (n_total - n_input) neurons receive zero input.

    Args:
        image: Tensor of shape [1, H, W] or [H, W] or [N] (already flat).
        T_steps: Number of simulation timesteps.
        n_total: Total number of neurons in the network (N_E + N_I).
        n_input: Number of input neurons (must be <= n_total).
        scale: Multiplier applied to pixel values before injection.

    Returns:
        Tensor of shape [T_steps, n_total] with pixel * scale in [:, :n_input]
        and zeros in [:, n_input:].
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise ImportError("PyTorch backend requires torch to be installed") from exc

    pixels = image.detach().float().reshape(-1)  # [n_input]
    if pixels.shape[0] != n_input:
        raise ValueError(
            f"image flattens to {pixels.shape[0]} values but n_input={n_input}"
        )

    ext = torch.zeros(T_steps, n_total, dtype=torch.float32)
    ext[:, :n_input] = (pixels * scale).unsqueeze(0).expand(T_steps, -1)
    return ext


def train_epoch(
    dataloader: object,
    optimizer: object,
    forward_fn: "Callable[[torch.Tensor], torch.Tensor]",
    *,
    n_total_samples: int,
    device: str = "cpu",
) -> "tuple[float, list[float], list[float]]":
    """One training epoch with cross-entropy classification loss.

    Args:
        dataloader: Yields (X, y) batches.
        optimizer: PyTorch optimizer; ``step()`` and ``zero_grad()`` are called
            per batch.
        forward_fn: Maps an image batch X → logits [B, C].  Encapsulates all
            model and simulation details; built by the caller as a closure.
        n_total_samples: Total samples in the dataset (used for ETA / pct logging).
        device: Device string for moving labels to match logits.

    Returns:
        ``(avg_loss, iter_losses, iter_accs)`` — scalar average loss and
        per-batch series for plotting.
    """
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover
        raise ImportError("PyTorch backend requires torch to be installed") from exc

    total_loss = 0.0
    n_batches = 0
    samples_done = 0
    iter_losses: list[float] = []
    iter_accs: list[float] = []
    epoch_start = time.perf_counter()
    log_every = max(1, len(dataloader) // 10)  # ~10 log lines per epoch

    for batch_idx, (X, y) in enumerate(dataloader):
        y = y.to(device)
        batch_start = time.perf_counter()
        pred = forward_fn(X)  # [B, C]
        loss = F.cross_entropy(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_loss = loss.item()
        batch_acc = (pred.detach().argmax(1) == y).float().mean().item()
        iter_losses.append(batch_loss)
        iter_accs.append(batch_acc)
        total_loss += batch_loss
        n_batches += 1
        samples_done += len(X)

        if batch_idx % log_every == 0 or batch_idx == len(dataloader) - 1:
            elapsed = time.perf_counter() - epoch_start
            sps = samples_done / elapsed if elapsed > 0 else 0.0
            remaining = (n_total_samples - samples_done) / sps if sps > 0 else float("inf")
            pct = 100 * samples_done / n_total_samples
            batch_ms = 1000 * (time.perf_counter() - batch_start) / len(X)
            print(
                f"  [{pct:5.1f}%] batch {batch_idx:>4d}/{len(dataloader)-1}"
                f"  loss: {batch_loss:.4f}  acc: {100*batch_acc:.1f}%"
                f"  {sps:.1f} samp/s  {batch_ms:.0f} ms/samp"
                f"  ETA: {remaining:.0f}s",
                flush=True,
            )

    return total_loss / max(n_batches, 1), iter_losses, iter_accs


def eval_epoch(
    dataloader: object,
    forward_fn: "Callable[[torch.Tensor], torch.Tensor]",
    *,
    device: str = "cpu",
) -> "tuple[float, float]":
    """Evaluate classification accuracy and loss over a dataloader.

    Args:
        dataloader: Yields (X, y) batches.
        forward_fn: Maps X → logits [B, C].  Same closure used in
            :func:`train_epoch`.
        device: Device string for moving labels.

    Returns:
        ``(avg_loss, accuracy)`` over the full dataset.
    """
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover
        raise ImportError("PyTorch backend requires torch to be installed") from exc

    total_loss = 0.0
    correct = 0
    total = 0
    eval_start = time.perf_counter()
    print(f"  evaluating {len(dataloader.dataset)} samples...", flush=True)
    with torch.no_grad():
        for X, y in dataloader:
            y = y.to(device)
            pred = forward_fn(X)
            total_loss += F.cross_entropy(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()
            total += len(y)

    elapsed = time.perf_counter() - eval_start
    print(f"  eval done in {elapsed:.1f}s  ({total / elapsed:.1f} samp/s)", flush=True)
    return total_loss / max(len(dataloader), 1), correct / max(total, 1)
