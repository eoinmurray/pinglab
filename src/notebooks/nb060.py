"""Notebook runner for entry 060 — strong vs weak coupling (analytic).

No simulation. Derives and plots two things straight from the balanced-network
scaling laws:

  1. how the mean recurrent input mu and its fluctuation sigma scale with the
     fan-in K under weak coupling (per-synapse weight w ~ 1/K) versus strong
     coupling (w ~ 1/sqrt(K)) — the figure that shows sigma collapsing under
     weak coupling but surviving at O(1) under strong;
  2. the per-synapse weight you must use to hold the coupling J = w*sqrt(K)
     fixed as the network is made sparser, w = J / sqrt((1-s) N_pre).

Companion to the canonical balanced state in nb058 and the strong/weak theory
note. Pure analytic — accepts the standard --tier/--modal-gpu flags for
consistency but ignores them (there is nothing to scale).

Notebook entry: src/docs/content/notebooks/nb060.mdx
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from helpers.fmt import format_duration  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import stamp_figure  # noqa: E402
from cli import theme  # noqa: E402

SLUG = "nb060"
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
    wipe_dir = "--no-wipe-dir" not in sys.argv
    t_start = time.monotonic()
    run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {run_id} (analytic, no compute)")

    prepare_run_dirs(SLUG, run_id, wipe=wipe_dir, make_artifacts=False)

    scaling = plot_input_scaling(FIGURES / "input_scaling.png", run_id)
    print(f"wrote {FIGURES / 'input_scaling.png'}")
    weight = plot_weight_vs_sparsity(FIGURES / "weight_vs_sparsity.png", run_id)
    print(f"wrote {FIGURES / 'weight_vs_sparsity.png'}")

    duration_s = time.monotonic() - t_start
    summary = {
        "notebook_run_id": run_id,
        "duration_s": round(duration_s, 2),
        "duration": format_duration(duration_s),
        "kind": "analytic",
        "config": {"N_pre": N_PRE, "r": R, "J0": J0, "W0": W0},
        "input_scaling": scaling,
        "weight_vs_sparsity": weight,
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {summary['duration']}")


if __name__ == "__main__":
    main()
