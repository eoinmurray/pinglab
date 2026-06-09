"""COBA streaming control for nb048.

Runs the τ-sweep and (τ, rate) grid on the trained COBA baselines from
nb025 (3 seeds × no I-loop), then produces side-by-side comparison
plots against the existing PING data persisted in nb048/numbers.json.

The point: if COBA streams as well as PING, the streaming result is
about the trained mem-mean readout, not the gamma cycle. If COBA is
worse, the gamma-cycle structure is doing real work in the streaming
regime.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src" / "pinglab"))
sys.path.insert(0, str(REPO / "src" / "pinglab" / "notebooks"))

import nb048  # noqa: E402
from pinglab import theme  # noqa: E402

ARCH = "coba"  # trained nb025 baselines: coba__off__seed{42,43,44}


def coba_dir(seed: int) -> Path:
    return (
        REPO / "src" / "artifacts" / "notebooks" / "nb025"
        / f"coba__off__seed{seed}"
    )


def load_coba(device, seed: int):
    """Mirrors nb048._load_trained_full but reads COBA configs."""
    import models as M
    from config import build_net, patch_dt
    from cli import load_dataset, seed_everything

    train_dir = coba_dir(seed)
    if not (train_dir / "weights.pth").exists():
        raise SystemExit(f"missing coba seed {seed} at {train_dir}")
    cfg = json.loads((train_dir / "config.json").read_text())
    seed_everything(int(cfg.get("seed", 42)))
    M.T_ms = float(cfg["t_ms"])
    patch_dt(float(cfg["dt"]))
    hidden_sizes = cfg.get("hidden_sizes") or [int(cfg["n_hidden"])]
    M.N_HID = hidden_sizes[-1]
    M.N_INH = hidden_sizes[-1] // 4
    M.HIDDEN_SIZES = list(hidden_sizes)
    _, X_te, _, y_te = load_dataset(
        cfg["dataset"], max_samples=int(cfg["max_samples"]), split=True
    )
    M.N_IN = 784 if cfg["dataset"] == "mnist" else 64
    w_in_cfg = cfg.get("w_in")
    w_in_arg = (
        (float(w_in_cfg[0]), float(w_in_cfg[1]))
        if isinstance(w_in_cfg, list) and len(w_in_cfg) >= 2
        else None
    )
    net = build_net(
        cfg["model"],
        w_in=w_in_arg,
        w_in_sparsity=float(cfg.get("w_in_sparsity") or 0.0),
        ei_strength=float(cfg.get("ei_strength") or 0.0),
        ei_ratio=float(cfg.get("ei_ratio") or 2.0),
        sparsity=float(cfg.get("sparsity") or 0.0),
        device=device,
        randomize_init=not bool(cfg.get("kaiming_init", False)),
        dales_law=bool(cfg.get("dales_law", True)),
        hidden_sizes=hidden_sizes,
        readout_mode=cfg.get("readout_mode", "mem-mean"),
    )
    if hasattr(net, "readout_mode"):
        net.readout_mode = cfg.get("readout_mode", "mem-mean")
    state = torch.load(train_dir / "weights.pth", map_location=device)
    net.load_state_dict(state, strict=False)
    net.eval()
    net.recording = True
    return net, cfg, X_te, y_te


# ── Plots: side-by-side heatmap (PING | COBA) + difference panel ────
def plot_heatmap_comparison(
    ping_rows: list[dict], coba_rows: list[dict],
    out_path: Path, run_id: str,
) -> None:
    theme.apply()
    taus = sorted(set(r["tau_ms"] for r in ping_rows))
    rates = sorted(set(r["input_rate_hz"] for r in ping_rows))

    def to_grid(rows):
        g = np.zeros((len(rates), len(taus)), dtype=np.float32)
        for r in rows:
            i = rates.index(r["input_rate_hz"])
            j = taus.index(r["tau_ms"])
            g[i, j] = r["acc"]
        return g

    g_ping = to_grid(ping_rows)
    g_coba = to_grid(coba_rows)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.5), dpi=150)
    panels = [
        (axes[0], g_ping, "magma", "PING accuracy (%)", 0, 100),
        (axes[1], g_coba, "magma", "COBA accuracy (%)", 0, 100),
    ]
    for ax, g, cmap_name, label, vmin, vmax in panels:
        im = ax.imshow(
            g, origin="lower", aspect="auto", cmap=cmap_name,
            vmin=vmin, vmax=vmax,
        )
        ax.set_xticks(range(len(taus)))
        ax.set_xticklabels([f"{t:g}" for t in taus])
        ax.set_yticks(range(len(rates)))
        ax.set_yticklabels([f"{r:g}" for r in rates])
        ax.set_xlabel(r"$\tau$ (ms)", fontsize=theme.SIZE_LABEL)
        if ax is axes[0]:
            ax.set_ylabel("Input rate (Hz)", fontsize=theme.SIZE_LABEL)
        for i in range(len(rates)):
            for j in range(len(taus)):
                v = g[i, j]
                txt = f"{v:.0f}" if cmap_name == "magma" else f"{v:+.0f}"
                # White text on dark cells (low magma values OR strong reds/blues).
                bright = (
                    v < 55 if cmap_name == "magma"
                    else abs(v) > 22
                )
                ax.text(
                    j, i, txt, ha="center", va="center",
                    fontsize=theme.SIZE_LABEL - 1,
                    color=("white" if bright else theme.INK_BLACK),
                )
        fig.colorbar(im, ax=ax, shrink=0.85).set_label(
            label, fontsize=theme.SIZE_LABEL,
        )
        ax.set_title(label.replace(" (%)", "").replace(" (pp)", ""),
                     fontsize=theme.SIZE_TITLE)
    fig.suptitle(
        "Streaming accuracy: PING vs COBA across $(\\tau,$ input rate$)$",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    fig.text(
        0.995, 0.005, run_id,
        ha="right", va="bottom", family="monospace",
        fontsize=theme.SIZE_CAPTION, color=theme.LABEL,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tau_comparison(
    ping_rows: list[dict], coba_rows: list[dict],
    out_path: Path, run_id: str,
) -> None:
    theme.apply()
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)
    styles = [
        ("ping", False, theme.INK_BLACK, "o", "-", "PING constant (25 Hz)"),
        ("ping", True, theme.DEEP_RED, "s", "-", "PING rate-compensated"),
        ("coba", False, theme.INK_BLACK, "o", "--", "COBA constant (25 Hz)"),
        ("coba", True, theme.DEEP_RED, "s", "--", "COBA rate-compensated"),
    ]
    for arch, comp, color, marker, ls, label in styles:
        rows = ping_rows if arch == "ping" else coba_rows
        sub = sorted(
            [r for r in rows if bool(r["rate_compensate"]) == comp],
            key=lambda r: r["tau_ms"],
        )
        if not sub:
            continue
        ax.errorbar(
            [r["tau_ms"] for r in sub], [r["acc"] for r in sub],
            yerr=[r.get("acc_sem", 0.0) for r in sub],
            marker=marker, color=color, ls=ls, lw=1.5, capsize=4,
            label=label,
        )
    ax.set_xlabel(r"Segment duration $\tau$ (ms)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Per-segment accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax.set_ylim(0, 100)
    ax.axhline(10.0, color=theme.GREY_MID, lw=0.5, ls=":", alpha=0.6)
    ax.axvline(28.0, color=theme.AMBER, lw=0.7, ls="--", alpha=0.8)
    ax.text(
        28.0, 92, " ≈ 1 gamma cycle (PING only)",
        fontsize=theme.SIZE_ANNOTATION, color=theme.AMBER, va="top",
    )
    ax.set_xticks(nb048.TAU_SWEEP_MS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="lower right")
    fig.suptitle(
        "Streaming accuracy vs $\\tau$ — PING vs COBA",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    fig.text(
        0.995, 0.005, run_id,
        ha="right", va="bottom", family="monospace",
        fontsize=theme.SIZE_CAPTION, color=theme.LABEL,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    from cli import _auto_device
    device = _auto_device()
    print(f"[coba-control] device={device}  seeds={nb048.SEEDS}")

    run_id = "coba-control"
    rows_constant: list[dict] = []
    rows_comp: list[dict] = []
    grid_rows: list[dict] = []
    t0 = time.monotonic()
    for sd in nb048.SEEDS:
        net, cfg, X_te, y_te = load_coba(device, sd)
        print(f"[tau-sweep] coba constant input  seed {sd}")
        rows_constant += nb048.run_tau_sweep(
            net, cfg, X_te, y_te, device, train_seed=sd,
            n_streams=20, n_per_stream=10, rate_compensate=False,
        )
        print(f"[tau-sweep] coba rate-compensated  seed {sd}")
        rows_comp += nb048.run_tau_sweep(
            net, cfg, X_te, y_te, device, train_seed=sd,
            n_streams=20, n_per_stream=10, rate_compensate=True,
        )
        print(f"[grid-sweep] coba τ × rate  seed {sd}")
        grid_rows += nb048.run_grid_sweep(
            net, cfg, X_te, y_te, device, train_seed=sd,
            n_streams=40, n_per_stream=10,
        )

    tau_agg = (
        nb048.aggregate_tau_rows(rows_constant)
        + nb048.aggregate_tau_rows(rows_comp)
    )
    grid_agg = nb048.aggregate_grid_rows(grid_rows)

    # Load existing PING aggregates from nb048's numbers.json.
    numbers_path = nb048.FIGURES / "numbers.json"
    data = json.loads(numbers_path.read_text())
    ping_tau = data["tau_sweep_agg"]
    ping_grid = data["grid_sweep_agg"]

    plot_tau_comparison(
        ping_tau, tau_agg,
        nb048.FIGURES / "acc_vs_tau__ping_vs_coba.png", run_id,
    )
    plot_heatmap_comparison(
        ping_grid, grid_agg,
        nb048.FIGURES / "acc_grid__ping_vs_coba.png", run_id,
    )

    # Persist coba data alongside the ping aggregates.
    data["coba_tau_sweep_per_seed"] = rows_constant + rows_comp
    data["coba_tau_sweep_agg"] = tau_agg
    data["coba_grid_sweep_per_seed"] = grid_rows
    data["coba_grid_sweep_agg"] = grid_agg
    numbers_path.write_text(json.dumps(data, indent=2) + "\n")

    print(f"\nduration: {time.monotonic() - t0:.1f}s")
    print(f"wrote {nb048.FIGURES / 'acc_vs_tau__ping_vs_coba.png'}")
    print(f"wrote {nb048.FIGURES / 'acc_grid__ping_vs_coba.png'}")
    print(f"updated {numbers_path}")


if __name__ == "__main__":
    main()
