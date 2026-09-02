"""Publication-sized comparison figures for exp111."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def comparison_figure(test, path):
    rows = test["series"]
    x = np.asarray([row["x"] for row in rows], dtype=float)
    left = np.asarray([row["snnsim"] for row in rows], dtype=float)
    right = np.asarray([row["brian2"] for row in rows], dtype=float)
    labels = [row["label"] for row in rows]
    if test["id"] == "spike-perturbations":
        x = np.arange(len(rows), dtype=float)

    figure, (axis, residual) = plt.subplots(1, 2, figsize=(7.08, 2.95))
    axis.plot(x, left, "o-", color="#17212b", lw=1.25, ms=3.5, label="snnsim")
    axis.plot(x, right, "s--", color="#b3473d", lw=1.15, ms=3.2, label="Brian2")
    residual.axhline(0, color="#777777", lw=0.7)
    residual.plot(x, right - left, "o-", color="#5b6f82", lw=1.2, ms=3.5)
    x_label = (
        "integration timestep (ms)"
        if test["id"] == "timestep-robustness"
        else test["x_label"]
    )
    axis.set_xlabel(x_label)
    axis.set_ylabel(test["y_label"])
    residual.set_xlabel(x_label)
    residual.set_ylabel("Brian2 − snnsim")
    axis.legend(frameon=False, fontsize=7)
    for panel, letter in ((axis, "A"), (residual, "B")):
        panel.text(
            -0.16,
            1.03,
            letter,
            transform=panel.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )
        panel.spines[["top", "right"]].set_visible(False)
        panel.tick_params(labelsize=7)
        panel.xaxis.label.set_size(8)
        panel.yaxis.label.set_size(8)
    categorical = len(rows) <= 8 and np.allclose(x, np.arange(len(rows)))
    if categorical:
        axis.set_xticks(
            x,
            labels,
            rotation=25 if max(map(len, labels)) > 8 else 0,
            ha="right" if max(map(len, labels)) > 8 else "center",
        )
        residual.set_xticks(
            x,
            labels,
            rotation=25 if max(map(len, labels)) > 8 else 0,
            ha="right" if max(map(len, labels)) > 8 else "center",
        )
    figure.suptitle(test["title"], fontsize=9, y=1.01)
    figure.tight_layout(pad=0.7, w_pad=1.2)
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)
