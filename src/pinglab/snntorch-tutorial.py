"""Faithful reproduction of snnTorch Tutorial 5 (Training SNN with PyTorch).

Reference: https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html

This is a standalone script that uses the snntorch library directly — no
pinglab code in the path. Used to ground-truth our SNNTorchNet behaviour
against the canonical tutorial setup (plain nn.Linear + snn.Leaky +
Kaiming init + Poisson rate coding + CE on summed output spikes).

Architecture and hyperparameters match the tutorial. In addition to the
tutorial's training loop, this script saves hidden-layer raster snapshots
at init (untrained) and end (after training) so we can visually compare
against pinglab's end_d0s0.png outputs for the same configuration.

Run:  uv run python src/pinglab/snntorch-tutorial.py
Outputs:  src/artifacts/snntorch-tutorial/
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import spikegen, surrogate

# ── Hyperparameters (tutorial 5 values) ─────────────────────────────────
NUM_INPUTS = 28 * 28
NUM_HIDDEN = 1000
NUM_OUTPUTS = 10
NUM_STEPS = 25
BETA = 0.95
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
NUM_EPOCHS = 1
DEVICE = torch.device("cpu")

OUT_DIR = Path("src/artifacts/snntorch-tutorial")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class Net(nn.Module):
    """Two-layer feedforward LIF network: fc1 → Leaky → fc2 → Leaky."""

    def __init__(self):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(NUM_INPUTS, NUM_HIDDEN)
        self.lif1 = snn.Leaky(beta=BETA, spike_grad=spike_grad)
        self.fc2 = nn.Linear(NUM_HIDDEN, NUM_OUTPUTS)
        self.lif2 = snn.Leaky(beta=BETA, spike_grad=spike_grad)

    def forward(self, x):
        # x: (T, B, num_inputs) — pre-encoded Poisson spike train
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk1_rec = []
        spk2_rec = []
        mem2_rec = []

        for t in range(NUM_STEPS):
            cur1 = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return (torch.stack(spk1_rec, dim=0),
                torch.stack(spk2_rec, dim=0),
                torch.stack(mem2_rec, dim=0))


def make_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ])
    train_set = datasets.MNIST(
        root="/tmp/mnist", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(
        root="/tmp/mnist", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                             shuffle=False, drop_last=True)
    return train_loader, test_loader


def encode(images):
    """Poisson rate-code an image batch into (T, B, num_inputs) spikes."""
    flat = images.view(images.size(0), -1)
    return spikegen.rate(flat, num_steps=NUM_STEPS)


def evaluate(net, loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            spikes = encode(data)
            _, spk2_rec, _ = net(spikes)
            pred = spk2_rec.sum(dim=0).argmax(dim=1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
    net.train()
    return 100.0 * correct / total


def grab_hidden_raster(net, loader):
    """Run one digit-0 sample through the net and return (T, N_hidden) spikes."""
    net.eval()
    data, targets = next(iter(loader))
    data, targets = data.to(DEVICE), targets.to(DEVICE)
    zero_idx = (targets == 0).nonzero(as_tuple=True)[0]
    idx = zero_idx[0].item() if len(zero_idx) > 0 else 0
    sample = data[idx:idx + 1]
    spikes = encode(sample)
    with torch.no_grad():
        spk1_rec, _, _ = net(spikes)
    net.train()
    return spk1_rec.squeeze(1).cpu().numpy(), sample.cpu().numpy()


def save_raster(spikes, image, fname, title):
    """Save a raster figure: hidden spike activity over time + the input digit."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4),
                             gridspec_kw={"width_ratios": [4, 1]})

    T, N = spikes.shape
    times, neurons = np.where(spikes > 0)
    axes[0].scatter(times, neurons, s=0.5, c="black", marker="|")
    axes[0].set_xlim(0, T)
    axes[0].set_ylim(0, N)
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Hidden neuron index")
    active_frac = (spikes.any(axis=0)).mean()
    rate_per_step = spikes.sum() / (N * T)
    axes[0].set_title(f"{title}  —  active {active_frac:.0%}, "
                      f"mean rate {rate_per_step:.2f} spk/neuron/step")

    axes[1].imshow(image.squeeze(), cmap="gray_r")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title("Input")

    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  → {fname}")


def main():
    print("snnTorch tutorial 5 reproduction")
    print(f"  num_hidden={NUM_HIDDEN}  num_steps={NUM_STEPS}  beta={BETA}")
    print(f"  batch_size={BATCH_SIZE}  lr={LEARNING_RATE}  epochs={NUM_EPOCHS}")
    print()

    print("Loading MNIST...")
    train_loader, test_loader = make_loaders()
    print(f"  train batches={len(train_loader)}  test batches={len(test_loader)}")

    net = Net().to(DEVICE)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  parameters={n_params:,}")

    # Init raster (untrained)
    print("\nInit raster:")
    spikes_init, sample_img = grab_hidden_raster(net, test_loader)
    save_raster(spikes_init, sample_img, OUT_DIR / "init_raster.png",
                "Init (untrained)")

    init_acc = evaluate(net, test_loader)
    print(f"  init test accuracy: {init_acc:.2f}%")

    # Train
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE,
                                 betas=(0.9, 0.999))
    loss_fn = nn.CrossEntropyLoss()
    print(f"\nTraining for {NUM_EPOCHS} epoch(s)...")
    t0 = time.time()
    for epoch in range(NUM_EPOCHS):
        for i, (data, targets) in enumerate(train_loader):
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            spikes = encode(data)

            net.train()
            _, spk2_rec, _ = net(spikes)
            logits = spk2_rec.sum(dim=0)  # cross-entropy on summed spikes
            loss = loss_fn(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                with torch.no_grad():
                    pred = logits.argmax(dim=1)
                    batch_acc = (pred == targets).float().mean().item() * 100
                print(f"  ep {epoch+1}  batch {i:4d}/{len(train_loader)}  "
                      f"loss={loss.item():.3f}  batch_acc={batch_acc:.0f}%")
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.0f}s")

    # End state
    print("\nEnd state:")
    end_acc = evaluate(net, test_loader)
    print(f"  end test accuracy: {end_acc:.2f}%")
    spikes_end, sample_img = grab_hidden_raster(net, test_loader)
    save_raster(spikes_end, sample_img, OUT_DIR / "end_raster.png",
                f"End (trained, {end_acc:.1f}%)")

    print(f"\n{'='*40}")
    print(f"  init acc: {init_acc:.2f}%")
    print(f"  end  acc: {end_acc:.2f}%")
    print(f"  artifacts: {OUT_DIR}/")


if __name__ == "__main__":
    sys.exit(main())
