from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools"), str(REPO / "tools/snnsim")]

import matplotlib.pyplot as plt
import numpy as np
from experiments.exp076.recipe import N_CLASSES
from experiments.helpers import theme


def plot_training(metrics: dict, out_path: Path) -> None:
    theme.apply()
    rows = metrics["epochs"]
    epochs = np.asarray([row["ep"] for row in rows])
    train_loss = np.asarray([row["loss"] for row in rows])
    test_loss = np.asarray([row["test_loss"] for row in rows])
    accuracy = np.asarray([row["acc"] for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
    axes[0].plot(epochs, train_loss, marker="o", color=theme.INK_BLACK, label="train")
    axes[0].plot(
        epochs, test_loss, marker="o", color=theme.DEEP_RED, label="validation"
    )
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy")
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, accuracy, marker="o", color=theme.INK_BLACK)
    axes[1].axhline(
        100.0 / N_CLASSES,
        color=theme.DEEP_RED,
        linestyle="--",
        linewidth=1.0,
        label="chance",
    )
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation accuracy (%)")
    axes[1].set_ylim(0, max(30.0, float(accuracy.max()) * 1.2))
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(epochs)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_lifecycle_svg(out_path: Path) -> None:
    boxes = [
        ("Python", "graph + recipe"),
        ("Bundle", "compiled graph"),
        ("Train", "selected + final"),
        ("Replay", "bundle + legacy"),
        ("Parity", "one-step exact"),
    ]
    width, height = 900, 230
    box_w, box_h = 145, 74
    gap = 30
    x0, y0 = 35, 78
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}"
        ".title{font-size:17px;font-weight:700}.sub{font-size:12px;fill:#555}"
        ".box{fill:#fff;stroke:#222;stroke-width:1.3;rx:10}"
        ".arrow{stroke:#8f1d14;stroke-width:2;fill:none;marker-end:url(#arrow)}"
        "</style>",
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#8f1d14"/></marker></defs>',
        '<text x="35" y="35" class="title">Checkpoint replay and equivalence</text>',
        '<text x="35" y="55" class="sub">Protocol for the MNIST PING + MeanVoltage adapter.</text>',
    ]
    for idx, (title, sub) in enumerate(boxes):
        x = x0 + idx * (box_w + gap)
        svg.append(
            f'<rect class="box" x="{x}" y="{y0}" width="{box_w}" height="{box_h}"/>'
        )
        svg.append(f'<text x="{x + 14}" y="{y0 + 31}" class="title">{title}</text>')
        svg.append(f'<text x="{x + 14}" y="{y0 + 52}" class="sub">{sub}</text>')
        if idx < len(boxes) - 1:
            x1 = x + box_w + 4
            x2 = x + box_w + gap - 7
            y = y0 + box_h / 2
            svg.append(f'<path class="arrow" d="M {x1} {y} L {x2} {y}"/>')
    svg.append("</svg>")
    out_path.write_text("\n".join(svg))
