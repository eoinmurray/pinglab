"""Notebook runner for entry 039 — CUBA-PING in two synaptic modes.

Standalone runner. Builds a current-based PING network from scratch in
PyTorch — no training, no oscilloscope CLI, no Modal — and probes its
dynamics under spatially uniform Poisson input at varying rates, in two
synaptic configurations:

1. **Instant synapses**: each presynaptic spike contributes its weight
   to the post-synaptic current at the next step, then vanishes. No
   τ_AMPA / τ_GABA filters.
2. **Exponential synapses**: standard τ_AMPA = 2 ms exponential decay
   on excitatory currents, τ_GABA = 9 ms on inhibitory.

Side-by-side comparison lets us see what the synaptic filters add vs
what the LIF membrane time constant already provides.

Figures land in /figures/notebooks/nb039/.

Notebook entry: src/docs/src/pages/notebooks/nb039.mdx
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from pinglab import theme  # noqa: E402

SLUG = "nb039"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG

# ── Architecture (matches PING from nb035 except for synapses) ──
N_E: int = 1024
N_I: int = 256
N_IN: int = 784

# Trial timing.
T_MS: float = 200.0
DT: float = 0.1
N_STEPS: int = int(T_MS / DT)

# LIF neuron — single set of constants for both populations.
TAU_M_MS: float = 20.0
V_REST: float = -70.0
V_TH: float = -50.0
V_RESET: float = -70.0
R_M: float = 200.0     # scales weight×spike → mV per ms; tuned so a few
                       # input spikes per ms push V from rest to threshold

# Weight initialisation. Means follow the canonical PING recipe so
# numerical comparisons against the COBA baselines stay meaningful.
W_IN_MEAN: float = 1.2
W_IN_STD: float = 0.12
W_IN_SPARSITY: float = 0.95
W_EI_MEAN: float = 1.0
W_EI_STD: float = 0.1
W_IE_MEAN: float = 2.0
W_IE_STD: float = 0.2

INPUT_RATES_HZ: tuple[float, ...] = (5.0, 25.0, 100.0)
FI_RATES_HZ: tuple[float, ...] = tuple(np.linspace(0.0, 100.0, 21))
SEED: int = 42

# Synapse timescales for the exponential-synapse condition (match the
# COBA-PING values used in nb035–nb038).
TAU_AMPA_MS: float = 2.0
TAU_GABA_MS: float = 9.0


def build_weights(seed: int) -> dict[str, torch.Tensor]:
    """Return W_in, W_ei, W_ie tensors — all non-negative; the sign of
    the I→E pathway is applied at the current-summation step."""
    g = torch.Generator().manual_seed(seed)
    w_in_dense = (
        torch.randn(N_IN, N_E, generator=g) * W_IN_STD + W_IN_MEAN
    ).clamp_min(0.0)
    mask = (torch.rand(N_IN, N_E, generator=g) > W_IN_SPARSITY).float()
    w_in = w_in_dense * mask

    w_ei = (
        torch.randn(N_E, N_I, generator=g) * W_EI_STD + W_EI_MEAN
    ).clamp_min(0.0)
    w_ie = (
        torch.randn(N_I, N_E, generator=g) * W_IE_STD + W_IE_MEAN
    ).clamp_min(0.0)
    return {"W_in": w_in, "W_ei": w_ei, "W_ie": w_ie}


def simulate(
    W: dict[str, torch.Tensor],
    input_rate_hz: float,
    seed: int,
    *,
    synapse_mode: str = "instant",
) -> dict[str, np.ndarray]:
    """One trial. Spatially uniform Poisson input at input_rate_hz on
    every channel for the full T_MS window.

    synapse_mode: "instant" — each presynaptic spike contributes its
    weight to the post-synaptic current at the next step then vanishes.
    "exp" — each pathway carries an exponentially-decaying synaptic
    current with τ_AMPA on excitation and τ_GABA on inhibition."""
    if synapse_mode not in ("instant", "exp"):
        raise ValueError(f"unknown synapse_mode {synapse_mode!r}")

    g = torch.Generator().manual_seed(seed)
    p_in = input_rate_hz * DT / 1000.0
    V_E = torch.full((N_E,), V_REST)
    V_I = torch.full((N_I,), V_REST)
    s_E_prev = torch.zeros(N_E)
    s_I_prev = torch.zeros(N_I)
    spk_E = torch.zeros(N_STEPS, N_E)
    spk_I = torch.zeros(N_STEPS, N_I)

    if synapse_mode == "exp":
        d_ampa = float(np.exp(-DT / TAU_AMPA_MS))
        d_gaba = float(np.exp(-DT / TAU_GABA_MS))
        # Per-spike scale: (1-d) so steady-state current under constant
        # spike rate matches the instant case (each spike's instant
        # contribution W replaced by an exponentially-decaying tail
        # with the same time integral W).
        s_ampa = 1.0 - d_ampa
        s_gaba = 1.0 - d_gaba
        I_E_exc = torch.zeros(N_E)   # AMPA-filtered: input (no E→E)
        I_E_inh = torch.zeros(N_E)   # GABA-filtered: I→E
        I_I_exc = torch.zeros(N_I)   # AMPA-filtered: E→I

    for t in range(N_STEPS):
        x = (torch.rand(N_IN, generator=g) < p_in).float()
        if synapse_mode == "instant":
            I_E = x @ W["W_in"] - s_I_prev @ W["W_ie"]
            I_I = s_E_prev @ W["W_ei"]
        else:
            I_E_exc = d_ampa * I_E_exc + s_ampa * (x @ W["W_in"])
            I_E_inh = d_gaba * I_E_inh + s_gaba * (s_I_prev @ W["W_ie"])
            I_I_exc = d_ampa * I_I_exc + s_ampa * (s_E_prev @ W["W_ei"])
            I_E = I_E_exc - I_E_inh
            I_I = I_I_exc
        V_E = V_E + (DT / TAU_M_MS) * (-(V_E - V_REST) + R_M * I_E)
        V_I = V_I + (DT / TAU_M_MS) * (-(V_I - V_REST) + R_M * I_I)
        s_E = (V_E >= V_TH).float()
        s_I = (V_I >= V_TH).float()
        spk_E[t] = s_E
        spk_I[t] = s_I
        V_E = torch.where(s_E.bool(), torch.full_like(V_E, V_RESET), V_E)
        V_I = torch.where(s_I.bool(), torch.full_like(V_I, V_RESET), V_I)
        s_E_prev, s_I_prev = s_E, s_I

    return {"spk_E": spk_E.numpy(), "spk_I": spk_I.numpy()}


def population_rate_hz(spk: np.ndarray) -> float:
    """Mean per-cell firing rate in Hz over the trial."""
    n_cells = spk.shape[1]
    total_spikes = float(spk.sum())
    return total_spikes / (n_cells * T_MS / 1000.0) if n_cells else 0.0


def population_psth(spk: np.ndarray, bin_ms: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (time_bin_centers_ms, population_rate_hz_per_bin)."""
    steps_per_bin = max(1, int(round(bin_ms / DT)))
    n_bins = spk.shape[0] // steps_per_bin
    truncated = spk[: n_bins * steps_per_bin]
    binned = truncated.reshape(n_bins, steps_per_bin, spk.shape[1]).sum(axis=(1, 2))
    rate = binned / (spk.shape[1] * bin_ms / 1000.0)
    centers = (np.arange(n_bins) + 0.5) * bin_ms
    return centers, rate


def plot_raster(
    sample: dict[str, np.ndarray], title: str, out_path: Path,
) -> None:
    theme.apply()
    fig, (ax_e, ax_i, ax_p) = plt.subplots(
        3, 1, figsize=(8.0, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [4, 1, 1.6]},
    )
    spk_e, spk_i = sample["spk_E"], sample["spk_I"]
    t_ms = np.arange(spk_e.shape[0]) * DT
    e_idx, e_t = np.where(spk_e.T)
    ax_e.scatter(
        t_ms[e_t], e_idx, s=1.0, c=theme.INK_BLACK, marker="|", linewidths=0.5,
    )
    ax_e.set_ylabel("E neuron")
    ax_e.set_ylim(0, N_E)
    ax_e.set_title(title, fontsize=theme.SIZE_TITLE)
    i_idx, i_t = np.where(spk_i.T)
    ax_i.scatter(
        t_ms[i_t], i_idx, s=1.0, c=theme.DEEP_RED, marker="|", linewidths=0.5,
    )
    ax_i.set_ylabel("I neuron")
    ax_i.set_ylim(0, N_I)
    # PSTH overlay: 1 ms bins on a single axis showing E and I rates.
    cx_e, py_e = population_psth(spk_e, bin_ms=1.0)
    cx_i, py_i = population_psth(spk_i, bin_ms=1.0)
    ax_p.plot(cx_e, py_e, color=theme.INK_BLACK, lw=1.2, label="E")
    ax_p.plot(cx_i, py_i, color=theme.DEEP_RED, lw=1.2, label="I")
    ax_p.set_xlabel("time (ms)")
    ax_p.set_ylabel("pop. rate (Hz)")
    ax_p.legend(fontsize=theme.SIZE_CAPTION, frameon=False, loc="upper right")
    ax_p.set_xlim(0, T_MS)
    fig.tight_layout()
    fig.text(
        0.995, 0.005, f"nb039-{int(time.time())}",
        ha="right", va="bottom",
        fontsize=theme.SIZE_CAPTION, color=theme.LABEL, family="monospace",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fi_curve(rows: list[dict], out_path: Path) -> None:
    theme.apply()
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    xs = [r["input_rate_hz"] for r in rows]
    ys_e = [r["rate_e_hz"] for r in rows]
    ys_i = [r["rate_i_hz"] for r in rows]
    ax.plot(xs, ys_e, marker="o", color=theme.INK_BLACK, lw=1.5, label="E")
    ax.plot(xs, ys_i, marker="s", color=theme.DEEP_RED, lw=1.5, label="I")
    ax.set_xlabel("Input Poisson rate per channel (Hz)")
    ax.set_ylabel("Per-cell mean firing rate (Hz)")
    ax.set_title("CUBA-PING — instant-synapse population f–I curve")
    ax.legend(fontsize=theme.SIZE_LABEL, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.text(
        0.995, 0.005, f"nb039-{int(time.time())}",
        ha="right", va="bottom",
        fontsize=theme.SIZE_CAPTION, color=theme.LABEL, family="monospace",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_mode(
    W: dict[str, torch.Tensor],
    *,
    synapse_mode: str,
    fig_tag: str,
) -> dict:
    """Run rasters + f-I sweep for one synaptic mode and return the
    rows + summary."""
    print(f"[nb039:{synapse_mode}]")
    raster_rows: list[dict] = []
    for r in INPUT_RATES_HZ:
        sample = simulate(W, input_rate_hz=r, seed=SEED + 1,
                          synapse_mode=synapse_mode)
        rate_e = population_rate_hz(sample["spk_E"])
        rate_i = population_rate_hz(sample["spk_I"])
        out = FIGURES / f"raster__{fig_tag}__rate{int(r)}.png"
        plot_raster(
            sample,
            f"CUBA-PING ({synapse_mode} synapses) — input {r:g} Hz / channel "
            f"(E={rate_e:.1f} Hz, I={rate_i:.1f} Hz)",
            out,
        )
        print(
            f"  raster: input={r:>5.1f} Hz  "
            f"E={rate_e:6.2f} Hz  I={rate_i:6.2f} Hz  → {out.name}"
        )
        raster_rows.append({
            "input_rate_hz": float(r),
            "rate_e_hz": float(rate_e),
            "rate_i_hz": float(rate_i),
        })

    fi_rows: list[dict] = []
    for r in FI_RATES_HZ:
        sample = simulate(W, input_rate_hz=float(r), seed=SEED + 2,
                          synapse_mode=synapse_mode)
        rate_e = population_rate_hz(sample["spk_E"])
        rate_i = population_rate_hz(sample["spk_I"])
        fi_rows.append({
            "input_rate_hz": float(r),
            "rate_e_hz": float(rate_e),
            "rate_i_hz": float(rate_i),
        })
    fi_out = FIGURES / f"fi_curve__{fig_tag}.png"
    plot_fi_curve(fi_rows, fi_out)
    print(f"  fi-curve: {len(fi_rows)} points → {fi_out.name}")

    return {"rasters": raster_rows, "fi_curve": fi_rows}


def main() -> None:
    wipe_dir = "--no-wipe-dir" not in sys.argv
    if wipe_dir and FIGURES.exists():
        print(f"[wipe] {FIGURES.relative_to(REPO)}")
        shutil.rmtree(FIGURES)
    FIGURES.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    t_start = time.monotonic()
    print(f"[nb039] CUBA-PING, seed={SEED}")
    W = build_weights(SEED)
    sparsity_check = float((W["W_in"] > 0).float().mean())
    print(
        f"  N_E={N_E}, N_I={N_I}, N_IN={N_IN}; "
        f"W_in fill≈{sparsity_check:.3f} (target {1 - W_IN_SPARSITY:.3f})"
    )

    instant = run_mode(W, synapse_mode="instant", fig_tag="instant")
    exp = run_mode(W, synapse_mode="exp", fig_tag="exp")

    duration_s = time.monotonic() - t_start
    summary = {
        "slug": SLUG,
        "duration_s": round(duration_s, 1),
        "config": {
            "n_e": N_E, "n_i": N_I, "n_in": N_IN,
            "t_ms": T_MS, "dt": DT, "tau_m_ms": TAU_M_MS,
            "v_th": V_TH, "v_reset": V_RESET, "v_rest": V_REST, "r_m": R_M,
            "tau_ampa_ms": TAU_AMPA_MS, "tau_gaba_ms": TAU_GABA_MS,
            "w_in_mean": W_IN_MEAN, "w_in_std": W_IN_STD,
            "w_in_sparsity": W_IN_SPARSITY,
            "w_ei_mean": W_EI_MEAN, "w_ei_std": W_EI_STD,
            "w_ie_mean": W_IE_MEAN, "w_ie_std": W_IE_STD,
            "seed": SEED,
        },
        "instant": instant,
        "exp": exp,
        "success_criteria": [
            {
                "label": f"{mode} rasters rendered at every input rate",
                "passed": all(
                    (FIGURES / f"raster__{mode}__rate{int(r)}.png").exists()
                    for r in INPUT_RATES_HZ
                ),
                "detail": ", ".join(f"{int(r)}Hz" for r in INPUT_RATES_HZ),
            }
            for mode in ("instant", "exp")
        ] + [
            {
                "label": f"{mode} f-I curve rendered",
                "passed": (FIGURES / f"fi_curve__{mode}.png").exists(),
                "detail": f"{len(FI_RATES_HZ)} points",
            }
            for mode in ("instant", "exp")
        ],
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {duration_s:.1f}s")


if __name__ == "__main__":
    main()
