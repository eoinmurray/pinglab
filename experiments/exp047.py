"""Runner for exp047 — what sets the PING rate.

Sweeps the per-synapse I→E weight W^IE (x-axis) against the inhibitory pool size
N_I (lines) on an untrained PING network, reading each point's population E/I rate
from the snn tool. The finding is anchor-free: the rate is a function of W^IE, not
N_I, so the N_I curves collapse onto one master curve. The tool emits the rates;
this runner owns the sweep loop, renders the figure, and aggregates numbers.json.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import sh

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "snn" / "tool.py"
TEMP = ROOT / "temp" / "snn" / "sim"
ARTIFACTS = ROOT / "artifacts" / "data" / "exp047"

# Independent sweeps: per-synapse weight (x-axis) and pool size (lines).
W_IE_VALUES = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
N_I_SWEEP = [16, 64, 256]
W_IE_DEFAULT = 2.0  # the 1:4-init value; marked on the figure, anchors nothing.

# Fixed untrained-PING scale (matches the canonical biophysical init).
FIXED = dict(
    n_hidden=1024, n_in=784, ei_strength=1.0, w_in=1.2, w_in_sparsity=0.95,
    input_rate=25.0, n_batch=4, t_ms=200.0, dt=0.1, seed=42,
)


def run_point(n_inh: int, w_ie: float) -> dict:
    """One (N_I, W^IE) sim via the snn tool; returns the population E/I rates."""
    sh.uv.run(
        "python", str(TOOL), "sim",
        "--n-inh", str(n_inh), "--ei-ratio", str(w_ie),
        "--n-hidden", str(FIXED["n_hidden"]), "--n-in", str(FIXED["n_in"]),
        "--ei-strength", str(FIXED["ei_strength"]), "--w-in", str(FIXED["w_in"]),
        "--w-in-sparsity", str(FIXED["w_in_sparsity"]),
        "--input-rate", str(FIXED["input_rate"]), "--n-batch", str(FIXED["n_batch"]),
        "--t-ms", str(FIXED["t_ms"]), "--dt", str(FIXED["dt"]), "--seed", str(FIXED["seed"]),
        _fg=True,
    )
    out = json.loads((TEMP / "output.json").read_text())
    return {
        "n_inh": int(n_inh), "w_ie": float(w_ie),
        "r_e_hz": float(out["rate_e_hz"]), "r_i_hz": float(out["rate_i_hz"]),
    }


def plot_summary(rows_by_ni: dict, dst: Path) -> None:
    """E (left) and I (right) per-cell rate vs W^IE, one line per N_I. The lines
    overlap — the rate is set by the per-synapse weight, not the pool size."""
    fig, (ax_e, ax_i) = plt.subplots(1, 2, figsize=(10, 4.5))
    palette = ["#b2182b", "#ef8a62", "#1a1a1a"]
    markers = ["s", "^", "o"]
    n_is = sorted(rows_by_ni)
    for ax, key, ylabel in ((ax_e, "r_e_hz", "E rate (Hz / cell)"),
                            (ax_i, "r_i_hz", "I rate (Hz / cell)")):
        for n_inh, col, mk in zip(n_is, palette, markers):
            rows = sorted(rows_by_ni[n_inh], key=lambda r: r["w_ie"])
            ax.plot([r["w_ie"] for r in rows], [r[key] for r in rows],
                    mk + "-", color=col, lw=1.6, ms=6, label=f"$N_I = {n_inh}$")
        ax.axvline(W_IE_DEFAULT, ls=":", color="#888", lw=0.8)
        ax.set_xlabel("per-synapse $W^{IE}$ (μS)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 16.5)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_e.legend(frameon=False, loc="upper right")
    fig.suptitle("Rate is set by the per-synapse weight, not the pool size")
    fig.tight_layout()
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=120)
    plt.close(fig)


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    rows_by_ni: dict[int, list[dict]] = {}
    for n_inh in N_I_SWEEP:
        print(f"--- N_I = {n_inh} ---")
        rows = []
        for w_ie in W_IE_VALUES:
            r = run_point(n_inh, w_ie)
            rows.append(r)
            print(f"  W^IE={w_ie:>5.2f} μS  E={r['r_e_hz']:6.2f} Hz  I={r['r_i_hz']:6.2f} Hz")
        rows_by_ni[n_inh] = rows

    plot_summary(rows_by_ni, ARTIFACTS / "rate_vs_w_ie.png")
    print(f"rendered rate_vs_w_ie.png -> {ARTIFACTS}")

    numbers = {
        "config": {
            **FIXED, "w_ie_values": W_IE_VALUES, "n_i_sweep": N_I_SWEEP,
            "w_ie_default": W_IE_DEFAULT,
        },
        "rows_by_n_i": {str(k): v for k, v in rows_by_ni.items()},
    }
    (ARTIFACTS / "numbers.json").write_text(json.dumps(numbers, indent=2) + "\n")
    print(f"wrote {ARTIFACTS / 'numbers.json'}")


if __name__ == "__main__":
    main()
