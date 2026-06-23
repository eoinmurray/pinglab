"""Notebook runner for entry 058 — PING cue combination.

Step 6(a) FIRST: the no-network negative control.

Before any PING run, ask the cheap question that decides whether the whole
experiment is meaningful: does the precision-addition identity

    1/Var(z_AB) == 1/Var(z_A) + 1/Var(z_B)

already hold when the population-vector decoder is run directly on the *raw
input spikes*, with no network at all? If it does, the test does not isolate
anything the PING dynamics contribute, and we need a sharper observable before
spending a single network run.

Construction (no randomness in the geometry, Poisson only in the draws):
    - scalar latent z ~ N(0, sigma_p^2), fixed within a trial
    - N_E cells with preferred values theta_i tiling the z-range
    - per-cell rate lambda_i(z) = lambda0 [ 1 + rho (b1 + b2) ],
      b_{k,i} = exp(-(theta_i - z)^2 / 2 w^2); both cues centred on the SAME z
    - conditions: A alone (b1), B alone (b2), AB (b1 + b2). Both single cues
      share the same rate profile here; their independence is in the draws.
    - each "cycle" is a CYCLE_MS window; decode z by population vector
      (count-weighted centre of mass over theta_i) per window = one sample.
    - Var over windows within a fixed-z trial is the represented width.

House rules (ar016): only --tier and --modal-gpu are CLI args; every other
knob is a hardcoded literal here. The notebook is the recipe.
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

from cli import theme  # noqa: E402
from helpers.modal import parse_modal_gpu  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.tier import parse_tier  # noqa: E402

SLUG = "nb058"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)

# --- Hardcoded recipe ---------------------------------------------------
N_E = 200                  # tuned input cells
Z_HALF_RANGE = 3.0         # theta_i tile [-3, +3] (≈ ±3 sigma_p)
THETA = np.linspace(-Z_HALF_RANGE, Z_HALF_RANGE, N_E)
W_TUNE = 0.5               # tuning width (z units)
LAMBDA0_HZ = 10.0          # baseline floor rate, fixed across conditions
RHO = 8.0                  # bump contrast (one fixed moderate value)
SIGMA_P = 1.0              # prior standard deviation
Z_CLIP = 1.8               # keep the bump inside the tiled range
CYCLE_MS = 25.0            # one decode window = one "gamma cycle" sample
SEED = 0

TIER_CONFIG = {
    "tiny": {"n_trials": 30, "t_ms": 1000.0},
    "small": {"n_trials": 80, "t_ms": 2000.0},
    "medium": {"n_trials": 200, "t_ms": 2000.0},
    "large": {"n_trials": 400, "t_ms": 4000.0},
    "huge": {"n_trials": 800, "t_ms": 4000.0},
}
DEFAULT_TIER = "small"


def bump(z: float) -> np.ndarray:
    """Gaussian tuning kernel b_i(z) over the preferred values theta_i."""
    return np.exp(-((THETA - z) ** 2) / (2.0 * W_TUNE ** 2))


def condition_rates(z: float) -> dict[str, np.ndarray]:
    """Per-cell rate (Hz) for each condition at latent z.

    Both cues are centred on the same z, so A and B share a rate profile;
    AB sums the two bumps on the shared baseline floor.
    """
    b = bump(z)
    return {
        "A": LAMBDA0_HZ * (1.0 + RHO * b),
        "B": LAMBDA0_HZ * (1.0 + RHO * b),
        "AB": LAMBDA0_HZ * (1.0 + RHO * (b + b)),
    }


def decode_cycles(rate: np.ndarray, n_cycles: int, rng) -> np.ndarray:
    """Population-vector decode of z, one estimate per cycle window.

    Draws independent Poisson counts per cell per CYCLE_MS window at the given
    rate and returns the count-weighted centre of mass over theta_i.
    """
    mu = rate * CYCLE_MS / 1000.0                      # expected counts/cell/cycle
    counts = rng.poisson(mu, size=(n_cycles, N_E)).astype(np.float64)
    totals = counts.sum(axis=1)
    weighted = counts @ THETA
    with np.errstate(invalid="ignore", divide="ignore"):
        est = np.where(totals > 0, weighted / totals, np.nan)
    return est                                          # (n_cycles,)


def run_control(n_trials: int, t_ms: float):
    """Decode each condition on raw input across trials; return per-trial
    variances and the z used."""
    rng = np.random.default_rng(SEED)
    n_cycles = int(t_ms / CYCLE_MS)
    zs = np.clip(rng.normal(0.0, SIGMA_P, n_trials), -Z_CLIP, Z_CLIP)
    var = {"A": [], "B": [], "AB": []}
    example = None
    for k, z in enumerate(zs):
        rates = condition_rates(float(z))
        est = {c: decode_cycles(rates[c], n_cycles, rng) for c in ("A", "B", "AB")}
        for c in ("A", "B", "AB"):
            var[c].append(float(np.nanvar(est[c])))
        if k == n_trials // 2:                          # one mid-range trial to show
            example = {"z": float(z), "est": est}
    return zs, {c: np.array(v) for c, v in var.items()}, example


# --- figures ------------------------------------------------------------
def plot_tuning_geometry(out_path: Path) -> None:
    """Step 0 reference: rate vs theta for the conditions, and bump translation."""
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.0, 6.75), dpi=150)
    # Left: A/B (identical) vs AB at z = 0.
    rates = condition_rates(0.0)
    axl.plot(THETA, rates["A"], color=theme.INK_BLACK, lw=1.6, label="A or B (one cue)")
    axl.plot(THETA, rates["AB"], color=theme.DEEP_RED, lw=1.6, label="AB (both cues)")
    axl.axhline(LAMBDA0_HZ, color=theme.GREY_MID, lw=0.8, ls=":")
    axl.text(THETA[0], LAMBDA0_HZ, " baseline $\\lambda_0$", color=theme.GREY_MID,
             va="bottom", ha="left", fontsize=theme.SIZE_LABEL - 1)
    axl.set_xlabel("preferred value $\\theta_i$")
    axl.set_ylabel("input rate $\\lambda_i$ (Hz)")
    axl.set_title("Bump on the floor at $z = 0$ — AB is the clean sum",
                  loc="left", fontsize=theme.SIZE_LABEL)
    axl.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper right")
    axl.spines["top"].set_visible(False)
    axl.spines["right"].set_visible(False)
    # Right: AB bump translating with z.
    greys = [theme.GREY_LIGHT, theme.GREY_MID, theme.INK_BLACK]
    for z, col in zip((-1.2, 0.0, 1.2), greys):
        axr.plot(THETA, condition_rates(z)["AB"], color=col, lw=1.6,
                 label=f"$z = {z:+.1f}$")
    axr.set_xlabel("preferred value $\\theta_i$")
    axr.set_ylabel("input rate $\\lambda_i$ (Hz)")
    axr.set_title("AB bump translates with the latent $z$",
                  loc="left", fontsize=theme.SIZE_LABEL)
    axr.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper right")
    axr.spines["top"].set_visible(False)
    axr.spines["right"].set_visible(False)
    fig.suptitle("Step 0 — tuning geometry (noise-free reference)",
                 fontsize=theme.SIZE_TITLE, x=0.07, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_decode_scatter(example: dict, out_path: Path) -> None:
    """Step 4 sanity: per-cycle estimate distribution for one trial."""
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    z = example["z"]
    bins = np.linspace(z - 2.0, z + 2.0, 50)
    styles = {"A": (theme.DEEP_RED, "cue A"), "B": (theme.AMBER, "cue B"),
              "AB": (theme.INK_BLACK, "AB")}
    for c, (col, lab) in styles.items():
        est = example["est"][c]
        est = est[np.isfinite(est)]
        ax.hist(est, bins=bins, histtype="step", color=col, lw=1.6,
                label=f"{lab}  (Var = {np.var(est):.3f})")
    ax.axvline(z, color=theme.GREY_DARK, lw=1.0, ls="--")
    ax.text(z, ax.get_ylim()[1] * 0.96, " true $z$", color=theme.GREY_DARK,
            va="top", ha="left", fontsize=theme.SIZE_LABEL - 1)
    ax.set_xlabel("per-cycle estimate $\\hat z_t$")
    ax.set_ylabel("cycle count")
    ax.set_title("Step 4 — per-cycle decode for one trial (AB is narrower)",
                 loc="left", fontsize=theme.SIZE_TITLE)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_precision_test(var: dict, out_path: Path) -> dict:
    """Step 6(a): no-network precision-addition scatter vs the identity line."""
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"
    x = 1.0 / var["A"] + 1.0 / var["B"]
    y = 1.0 / var["AB"]
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    slope = float(np.sum(x * y) / np.sum(x * x))       # best-fit through origin
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    lim = float(max(x.max(), y.max())) * 1.05
    ax.plot([0, lim], [0, lim], color=theme.GREY_MID, lw=1.0, ls="--",
            label="identity (optimal)")
    ax.scatter(x, y, s=20, c=theme.INK_BLACK, alpha=0.7, linewidths=0,
               label=f"no-network control (slope {slope:.2f})")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("$1/\\mathrm{Var}(\\hat z_A) + 1/\\mathrm{Var}(\\hat z_B)$")
    ax.set_ylabel("$1/\\mathrm{Var}(\\hat z_{AB})$")
    ax.set_title("Step 6(a) — does COM-on-input already pass? (no PING)",
                 loc="left", fontsize=theme.SIZE_TITLE)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"slope_through_origin": slope, "n_points": int(good.sum())}


def main() -> None:
    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    parse_modal_gpu(sys.argv)
    wipe_dir = "--no-wipe-dir" not in sys.argv
    cfg = TIER_CONFIG[tier]

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {notebook_run_id} tier={tier} "
          f"n_trials={cfg['n_trials']} t_ms={cfg['t_ms']}")
    prepare_run_dirs(SLUG, notebook_run_id, wipe=wipe_dir, make_artifacts=False)

    zs, var, example = run_control(cfg["n_trials"], cfg["t_ms"])

    tuning_dst = FIGURES / "tuning_geometry.png"
    plot_tuning_geometry(tuning_dst)
    print(f"wrote {tuning_dst}")

    decode_dst = FIGURES / "decode_scatter.png"
    plot_decode_scatter(example, decode_dst)
    print(f"wrote {decode_dst}")

    test_dst = FIGURES / "control_precision_test.png"
    stats = plot_precision_test(var, test_dst)
    print(f"wrote {test_dst}  (slope through origin = {stats['slope_through_origin']:.3f})")

    med = {c: float(np.nanmedian(var[c])) for c in ("A", "B", "AB")}
    print(f"  median Var: A={med['A']:.3f} B={med['B']:.3f} AB={med['AB']:.3f}")

    duration_s = time.monotonic() - t_start
    summary = {
        "notebook_run_id": notebook_run_id,
        "duration_s": round(duration_s, 1),
        "tier": tier,
        "n_trials": cfg["n_trials"],
        "t_ms": cfg["t_ms"],
        "n_e": N_E,
        "lambda0_hz": LAMBDA0_HZ,
        "rho": RHO,
        "w_tune": W_TUNE,
        "sigma_p": SIGMA_P,
        "cycle_ms": CYCLE_MS,
        "median_var": med,
        "precision_test": stats,
        "example_z": example["z"],
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  duration: {duration_s:.1f}s")


if __name__ == "__main__":
    main()
