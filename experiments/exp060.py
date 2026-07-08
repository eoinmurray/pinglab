"""Experiment runner for entry exp060 — strong vs weak coupling (analytic).

No simulation. Derives and plots two things straight from the balanced-network
scaling laws:

  1. how the mean recurrent input mu and its fluctuation sigma scale with the
     fan-in K under weak coupling (per-synapse weight w ~ 1/K) versus strong
     coupling (w ~ 1/sqrt(K)) — the figure that shows sigma collapsing under
     weak coupling but surviving at O(1) under strong;
  2. the per-synapse weight you must use to hold the coupling J = w*sqrt(K)
     fixed as the network is made sparser, w = J / sqrt((1-s) N_pre).

Companion to the canonical balanced state in exp058 and the strong/weak theory
note. Pure analytic — there is nothing to simulate.

Writing: writings/exp060.typ · figures + numbers.json: artifacts/data/exp060/
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
# Helpers + sibling runners live alongside this file under experiments/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import theme  # noqa: E402
from helpers.cli import parse_meta  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import stamp_figure  # noqa: E402

SLUG = "exp060"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)

# Illustrative constants (arbitrary units — the figures are about *scaling*,
# not absolute magnitudes).
N_PRE = 1024     # presynaptic pool size, matching the project's N_E
R = 10.0         # presynaptic rate (a.u.)
J0 = 1.0         # the O(1) coupling held fixed under strong coupling
W0 = 1.0         # the O(1) constant held fixed under weak coupling


def plot_input_scaling(out_path: Path, run_id: str) -> dict:
    """Mean mu and fluctuation sigma versus fan-in K, weak vs strong."""
    theme.apply()
    K = np.logspace(0.5, 4.0, 300)  # K from ~3 to 1e4

    # Per-synapse weight scalings.
    w_weak = W0 / K
    w_strong = J0 / np.sqrt(K)

    # mu ~ K w r, sigma ~ sqrt(K) w sqrt(r).
    mu_weak = K * w_weak * R
    sig_weak = np.sqrt(K) * w_weak * np.sqrt(R)
    mu_strong = K * w_strong * R
    sig_strong = np.sqrt(K) * w_strong * np.sqrt(R)

    fig, (ax_mu, ax_sig) = plt.subplots(1, 2, figsize=(8.0, 4.5), dpi=150)

    for ax, weak, strong, label in (
        (ax_mu, mu_weak, mu_strong, r"mean input  $\mu \approx K\,w\,r$"),
        (ax_sig, sig_weak, sig_strong, r"fluctuation  $\sigma \approx \sqrt{K}\,w\,\sqrt{r}$"),
    ):
        ax.loglog(K, strong, color=theme.INK_BLACK, lw=2.0,
                  label=r"strong  $w\!\sim\!1/\sqrt{K}$")
        ax.loglog(K, weak, color=theme.GREY_MID, lw=2.0, ls="--",
                  label=r"weak  $w\!\sim\!1/K$")
        ax.set_xlabel("fan-in  $K$", fontsize=theme.SIZE_LABEL)
        ax.set_title(label, fontsize=theme.SIZE_TITLE)
        ax.legend(fontsize=theme.SIZE_LEGEND, frameon=True)

    ax_mu.set_ylabel("mean input (a.u.)", fontsize=theme.SIZE_LABEL)
    ax_sig.set_ylabel("input fluctuation (a.u.)", fontsize=theme.SIZE_LABEL)

    # Annotate the punchline: sigma survives only under strong coupling.
    ax_sig.annotate("survives — $O(1)$", xy=(K[-1], sig_strong[-1]),
                    xytext=(0.30, 0.78), textcoords="axes fraction",
                    fontsize=theme.SIZE_ANNOTATION, color=theme.INK_BLACK)
    ax_sig.annotate("collapses — $\\propto 1/\\sqrt{K}$", xy=(K[-1], sig_weak[-1]),
                    xytext=(0.30, 0.22), textcoords="axes fraction",
                    fontsize=theme.SIZE_ANNOTATION, color=theme.GREY_DARK)

    fig.suptitle(
        "Strong coupling keeps the recurrent fluctuation alive as the network grows",
        fontsize=theme.SIZE_TITLE, x=0.01, ha="left",
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "sigma_strong_at_Kmax": float(sig_strong[-1]),
        "sigma_weak_at_Kmax": float(sig_weak[-1]),
        "sigma_ratio_strong_over_weak_at_Kmax": float(sig_strong[-1] / sig_weak[-1]),
        "K_max": float(K[-1]),
    }


def plot_weight_vs_sparsity(out_path: Path, run_id: str) -> dict:
    """Per-synapse weight needed to hold J fixed, versus sparsity s."""
    theme.apply()
    s = np.linspace(0.0, 0.99, 300)
    K = (1.0 - s) * N_PRE
    w = J0 / np.sqrt(K)  # = J0 / sqrt((1-s) N_pre)

    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    ax.plot(s, w, color=theme.INK_BLACK, lw=2.2)
    ax.set_xlabel("sparsity  $s$", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel(r"weight to hold $J$ fixed:  $w = J/\sqrt{(1-s)\,N_\mathrm{pre}}$",
                  fontsize=theme.SIZE_LABEL)
    ax.set_title("Sparser networks need stronger synapses to stay in the same regime",
                 fontsize=theme.SIZE_TITLE)

    # Top axis: the fan-in K the sparsity implies.
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ticks = np.array([0.0, 0.5, 0.75, 0.9, 0.99])
    ax_top.set_xticks(ticks)
    ax_top.set_xticklabels([f"{int(round((1 - t) * N_PRE))}" for t in ticks])
    ax_top.set_xlabel(r"fan-in  $K = (1-s)\,N_\mathrm{pre}$", fontsize=theme.SIZE_LABEL)

    fig.tight_layout()
    stamp_figure(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "w_at_s0": float(J0 / np.sqrt(N_PRE)),
        "w_at_s099": float(J0 / np.sqrt((1 - 0.99) * N_PRE)),
        "N_pre": N_PRE,
    }


def main() -> None:
    meta = parse_meta(sys.argv)
    t_start = time.monotonic()
    run_id = next_run_id(SLUG)
    print(f"experiment_run_id = {run_id} (analytic, no compute)")

    with published_run(
        SLUG, run_id, make_artifacts=False, plot_only=meta.plot_only,
    ) as (_artifacts, figures):
        scaling = plot_input_scaling(figures / "input_scaling.png", run_id)
        print(f"wrote {figures / 'input_scaling.png'}")
        weight = plot_weight_vs_sparsity(figures / "weight_vs_sparsity.png", run_id)
        print(f"wrote {figures / 'weight_vs_sparsity.png'}")

        duration_s = time.monotonic() - t_start
        write_numbers(
            figures, run_id=run_id, duration_s=duration_s,
            payload={
                "kind": "analytic",
                "config": {"N_pre": N_PRE, "r": R, "J0": J0, "W0": W0},
                "input_scaling": scaling,
                "weight_vs_sparsity": weight,
            },
        )
        print(f"wrote {figures / 'numbers.json'}")


if __name__ == "__main__":
    main()
