"""Notebook runner for entry 037 — spike-stream perturbations of
trained PING.

Standalone runner with no cross-notebook helpers. Trains coba / ping
baseline cells (θ_u = off, three seeds), then runs:
- hidden-spike perturbation sweep (drop + Poisson add) against the
  trained baselines; and
- τ_GABA sweep (inference-time mutation of the inhibitory decay
  constant on trained PING).

Figures land in /figures/notebooks/nb037/ and the success-criteria
summary in nb037/numbers.json.

Notebook entry: src/docs/src/pages/notebooks/nb037.mdx
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from helpers.figsave import save_figure  # noqa: E402
from helpers.fmt import format_duration  # noqa: E402
from helpers.modal import BatchDispatcher, parse_modal_gpu  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import stamp_figure  # noqa: E402
from helpers import theme  # noqa: E402
from nb022 import cell_dir as shared_cell_dir, cell_name  # noqa: E402

SLUG = "nb037"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
OSCILLOSCOPE = REPO / "src" / "cli/cli.py"

MAX_SAMPLES = 500
T_MS = 200.0
DT_TRAIN = 0.1

# Baseline (θ_u = off) cells are trained at multiple seeds so the
# headline bar chart and learning curves can show mean ± SEM. The θ_u
# sweep cells stay single-seed — the frontier *shape* is dominated by
# the regulariser, not the seed.
SEEDS_BASELINE: list[int] = [42, 43, 44]
SEED_SWEEP: int = 42
BASELINE_EPOCHS: int = 30  # overrides the baked epochs for baseline cells

# Inference-time ei_strength sweep on the coba__off__seed42 baseline.
# Subsumes the now-retired nb019 — trains nothing new; just runs the
# already-trained coba weights forward through the test set with a
# fresh ping-arch I-loop at progressively higher ei_strength.
EI_SWEEP: list[float] = [round(0.1 * i, 1) for i in range(11)]  # 0.0–1.0
EI_RASTER: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
EI_RASTER_SAMPLE_IDX: int = 0
EI_RASTER_N_E_PLOT: int = 200
EI_RASTER_N_I_PLOT: int = 64

# Hidden-spike perturbation sweep levels. drop = fraction of hidden spikes
# removed (0.0–1.0); add = extra spikes injected per cell over the trial
# (0.0–40.0). Restored from the committed nb037 numbers.json after a refactor
# dropped these module constants while leaving the code that references them.
PERTURB_DROP_LEVELS: list[float] = [round(0.1 * i, 1) for i in range(11)]  # 0.0–1.0
PERTURB_ADD_LEVELS: list[float] = [float(2 * i) for i in range(21)]        # 0.0–40.0
PERTURB_RASTER_DROP_LEVELS: list[float] = [0.0, 0.5, 1.0]
PERTURB_RASTER_ADD_LEVELS: list[float] = [0.0, 20.0, 40.0]

# θ_u sweep grid in spikes-per-trial. None = no penalty (baseline).
# At T = 200 ms, spikes/trial × 5 = Hz. The grid spans from no
# pressure (off → ~80 Hz coba baseline) down to 1 Hz —
# below ping's natural 5 Hz and into the regime where every model
# loses accuracy.
THETA_U_GRID: list[float | None] = [None, 5.0, 2.0, 1.0, 0.5, 0.2]
FR_STRENGTH_UPPER = 1e-3

MODELS = ["coba", "ping"]

MODEL_RECIPES: dict[str, dict] = {
    "coba": {
        "__build_as": "ping",
        "--ei-strength": "0",
        "--v-grad-dampen": "1000",
        "--w-in": "0.3",
        "--w-in-sparsity": "0.95",
        "--readout": "mem-mean",
        "--surrogate-slope": "1",
        "--readout-w-out-scale": "100",
        "--lr": "0.0004",
        "--batch-size": "256",
    },
    "ping": {
        "__build_as": "ping",
        "--ei-strength": "1",
        "--v-grad-dampen": "1000",
        "--w-in": "1.2",
        "--w-in-sparsity": "0.95",
        "--readout": "mem-mean",
        "--surrogate-slope": "1",
        "--readout-w-out-scale": "500",
        "--lr": "0.0004",
        "--batch-size": "256",
    },
}

MODEL_COLORS = {
    "coba": theme.DEEP_RED,
    "ping": theme.INK_BLACK,
}
MODEL_MARKERS = {"coba": "s", "ping": "D"}

# Run scale — stamped into the manifest by run_dirs.prepare and rendered as
# the Methods table via RunScale; the mdx never restates these numbers.
SCALE = {
    "dataset": "mnist",
    "max_samples": MAX_SAMPLES,
    "epochs": BASELINE_EPOCHS,
    "t_ms": T_MS,
    "dt_ms": DT_TRAIN,
    "batch_size": 256,
    "seeds": len(SEEDS_BASELINE),
    "cells": len(MODELS) * len(THETA_U_GRID),
    "grid": "2 models × 6 θ_u values",
}

def theta_label(theta_u: float | None) -> str:
    """Filesystem-safe label for an out-dir."""
    if theta_u is None:
        return "off"
    s = f"{theta_u:g}".replace(".", "p")
    return f"tu{s}"


def theta_display(theta_u: float | None) -> str:
    """Human label for plots / numbers.json."""
    if theta_u is None:
        return "off"
    return f"{theta_u:g}"


def theta_hz(theta_u: float | None) -> float | None:
    if theta_u is None:
        return None
    return theta_u * (1000.0 / T_MS)


def seeds_for(theta_u: float | None) -> list[int]:
    """Baseline cells run all seeds; sweep cells stay single-seed."""
    return list(SEEDS_BASELINE) if theta_u is None else [SEED_SWEEP]


def cell_dir(model: str, theta_u: float | None, seed: int) -> Path:
    """Trained cell — now the shared nb022 cell (train-once / reuse-many).
    nb022 owns the θ_u sweep; this notebook only consumes it."""
    return shared_cell_dir(cell_name(model, theta_u, seed))


def baseline_dir(model: str, seed: int = SEEDS_BASELINE[0]) -> Path:
    return cell_dir(model, None, seed)


def build_train_args(
    model: str, theta_u: float | None, seed: int, out_dir: Path
) -> list[str]:
    recipe = MODEL_RECIPES[model]
    args = [
        "train",
        "--model", recipe["__build_as"],
        "--dataset", "mnist",
        "--max-samples", str(MAX_SAMPLES),
        "--epochs", str(BASELINE_EPOCHS),
        "--t-ms", str(T_MS),
        "--dt", str(DT_TRAIN),
        "--seed", str(seed),
        "--out-dir", str(out_dir),
        "--wipe-dir",
    ]
    for k, v in recipe.items():
        if k.startswith("__"):
            continue
        if v is True:
            args.append(k)
        elif v is not None:
            args += [k, v]
    if theta_u is not None:
        args += [
            "--fr-reg-upper-theta", str(theta_u),
            "--fr-reg-upper-strength", str(FR_STRENGTH_UPPER),
        ]
    return args


def load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text())


def load_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())



def capture_perturbation_raster(
    train_dir: Path, mode: str, level, sample_idx: int = 0
) -> dict:
    """Single-trial raster with the hidden-spike perturbation active, via the CLI
    snapshot (`sim --infer --perturb-mode M --perturb-level L --sample-index N`)."""
    cfg = json.loads((train_dir / "config.json").read_text())
    lvl = list(level) if isinstance(level, (list, tuple)) else [level]
    out_dir = (ARTIFACTS / "perturb_raster" / f"{mode}_{'_'.join(str(x) for x in lvl)}_s{sample_idx}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "uv", "run", "python", str(OSCILLOSCOPE), "sim", "--infer",
            "--load-config", str((train_dir / "config.json").resolve()),
            "--load-weights", str((train_dir / "weights.pth").resolve()),
            "--perturb-mode", mode,
            "--perturb-level", *[str(x) for x in lvl],
            "--sample-index", str(sample_idx),
            "--out-dir", str(out_dir),
        ],
        cwd=REPO,
        check=True,
    )
    d = np.load(out_dir / "snapshot.npz")
    e_full, i_full = d["spk_e"], d["spk_i"]
    if e_full.ndim == 3:
        e_full = e_full[:, 0, :]
    if i_full.ndim == 3:
        i_full = i_full[:, 0, :]
    t_sec = float(cfg["t_ms"]) / 1000.0
    e_rate_hz = float(e_full.sum() / (e_full.shape[1] * t_sec))
    rng = np.random.default_rng(0)
    e_idx = np.sort(rng.choice(e_full.shape[1], EI_RASTER_N_E_PLOT, replace=False))
    i_idx = np.sort(rng.choice(i_full.shape[1], EI_RASTER_N_I_PLOT, replace=False))
    return {
        "mode": mode,
        "level": (list(float(x) for x in level) if isinstance(level, (list, tuple)) else float(level)),
        "e_rate_hz": e_rate_hz,
        "label": int(d["label"]),
        "e": e_full[:, e_idx].astype(bool),
        "i": i_full[:, i_idx].astype(bool),
        "dt": float(cfg["dt"]),
        "t_ms": float(cfg["t_ms"]),
    }


def plot_perturbation_rasters(
    samples: list[dict], out_path: Path, run_id: str, level_fmt: str, title: str
) -> None:
    """Stacked single-trial rasters across perturbation levels for one mode."""
    theme.apply()
    n = len(samples)
    n_e = EI_RASTER_N_E_PLOT
    n_i = EI_RASTER_N_I_PLOT
    gap = 6
    fig, axes = plt.subplots(
        n, 1, figsize=(6.9, 3.88),
        sharex=True, gridspec_kw={"hspace": 0.18},
    )
    if n == 1:
        axes = [axes]
    for i, (ax, s) in enumerate(zip(axes, samples)):
        T = s["e"].shape[0]
        t_axis = np.arange(T) * s["dt"]
        e_t, e_n = np.where(s["e"])
        i_t, i_n = np.where(s["i"])
        ax.scatter(
            t_axis[e_t], e_n,
            s=2.0, c=theme.INK_BLACK, marker="|", linewidths=0.4,
        )
        ax.scatter(
            t_axis[i_t], i_n + n_e + gap,
            s=2.0, c=theme.DEEP_RED, marker="|", linewidths=0.4,
        )
        ax.set_ylim(-2, n_e + n_i + gap + 2)
        ax.set_yticks([n_e / 2, n_e + gap + n_i / 2])
        ax.set_yticklabels(["E", "I"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, s["t_ms"])
        ax.text(
            1.012, 0.5,
            level_fmt.format(level=s["level"]) + f"\nE = {s['e_rate_hz']:.1f} Hz",
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=theme.SIZE_LABEL,
        )
        if i == 0:
            ax.set_title(title)
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense raster: PNG + PDF
    plt.close(fig)




def run_perturbation_sweep(train_dir: Path, mode: str, level) -> dict:
    """Evaluate the test set under a hidden-spike perturbation, via the CLI.

    Runs `sim --infer --perturb-mode M --perturb-level L`; the perturbation is
    applied inside the CLI's forward loop (models.py hook). acc + hidden E rate
    come from metrics.json.
    """
    cfg = json.loads((train_dir / "config.json").read_text())
    lvl = list(level) if isinstance(level, (list, tuple)) else [level]
    out_dir = (ARTIFACTS / "perturb" / f"{mode}_{'_'.join(str(x) for x in lvl)}_{train_dir.name}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "uv", "run", "python", str(OSCILLOSCOPE), "sim", "--infer",
            "--load-config", str((train_dir / "config.json").resolve()),
            "--load-weights", str((train_dir / "weights.pth").resolve()),
            "--perturb-mode", mode,
            "--perturb-level", *[str(x) for x in lvl],
            "--out-dir", str(out_dir),
        ],
        cwd=REPO,
        check=True,
    )
    m = json.loads((out_dir / "metrics.json").read_text())
    rates = m.get("rates_hz", {})
    hid = max((k for k in rates if k.startswith("hid")), default=None)
    return {
        "mode": mode,
        "level": level,
        "acc": float(m["best_acc"]),
        "e_rate_hz": float(rates.get(hid, 0.0)) if hid else 0.0,
        "n_total": int(m.get("n_total", 0)),
    }


def plot_perturbation_curves(
    points: list[dict], out_path: Path, run_id: str,
    add_pct_rows: list[dict] | None = None,
) -> None:
    """Two-panel accuracy plot: drop on the left, add on the right.

    Left panel: drop probability (Bernoulli spike mask).
    Right panel: Poisson add. If `add_pct_rows` is provided, the right
    panel uses percentage-of-baseline-rate from those rows (the fair
    comparison). Otherwise it falls back to absolute Hz from `points`.
    """
    theme.apply()
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.15), sharey=True)
    use_pct = add_pct_rows is not None and len(add_pct_rows) > 0

    # Left panel: drop (as % of spikes dropped)
    ax_drop = axes[0]
    for model in MODELS:
        rows = [
            p for p in points if p["model"] == model and p["mode"] == "drop"
        ]
        rows.sort(key=lambda p: p["level"])
        ax_drop.plot(
            [p["level"] * 100 for p in rows], [p["acc"] for p in rows],
            marker=MODEL_MARKERS[model], markersize=5, linewidth=1.4,
            color=MODEL_COLORS[model], label=model,
        )
    ax_drop.set_xlabel("Spikes dropped (% of emitted)",
                       fontsize=theme.SIZE_LABEL)
    ax_drop.set_title("Drop — Bernoulli spike mask",
                      fontsize=theme.SIZE_LABEL, loc="left", pad=4)
    ax_drop.set_xlim(-2, 102)
    ax_drop.axhline(10.0, ls="--", color=theme.MUTED, lw=0.7, alpha=0.6)
    ax_drop.text(
        0.02, 12, "chance", transform=ax_drop.get_yaxis_transform(),
        fontsize=theme.SIZE_ANNOTATION, color=theme.MUTED, va="bottom",
    )

    # Right panel: add
    ax_add = axes[1]
    if use_pct:
        for model in MODELS:
            rows = sorted(
                [r for r in add_pct_rows if r["model"] == model],
                key=lambda r: r["pct"],
            )
            ax_add.plot(
                [r["pct"] * 100 for r in rows], [r["acc"] for r in rows],
                marker=MODEL_MARKERS[model], markersize=5, linewidth=1.4,
                color=MODEL_COLORS[model], label=model,
            )
        ax_add.set_xlabel(
            "Added Poisson noise (% of baseline rate)",
            fontsize=theme.SIZE_LABEL,
        )
        ax_add.set_title(
            "Add — Poisson noise as % of baseline",
            fontsize=theme.SIZE_LABEL, loc="left", pad=4,
        )
        ax_add.set_xlim(-2, 102)
    else:
        for model in MODELS:
            rows = sorted(
                [p for p in points
                 if p["model"] == model and p["mode"] == "add"],
                key=lambda p: p["level"],
            )
            ax_add.plot(
                [p["level"] for p in rows], [p["acc"] for p in rows],
                marker=MODEL_MARKERS[model], markersize=5, linewidth=1.4,
                color=MODEL_COLORS[model], label=model,
            )
        ax_add.set_xlabel("Poisson rate (Hz / neuron)",
                          fontsize=theme.SIZE_LABEL)
        ax_add.set_title("Add — Poisson noise injection",
                          fontsize=theme.SIZE_LABEL, loc="left", pad=4)
    ax_add.axhline(10.0, ls="--", color=theme.MUTED, lw=0.7, alpha=0.6)

    for ax in axes:
        ax.set_ylim(0, 100)
        ax.tick_params(labelsize=theme.SIZE_TICK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_locator(plt.matplotlib.ticker.MultipleLocator(20))
        ax.grid(True, axis="y", alpha=0.15, linewidth=0.5)
    ax_drop.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax_add.legend(
        loc="upper right", fontsize=theme.SIZE_LEGEND, frameon=False,
    )
    fig.suptitle(
        "Hidden-spike perturbation — accuracy vs perturbation level",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)  # line/curve plot: SVG + PDF
    plt.close(fig)


# ── inference helpers used by perturbation sweep ────────────────────


def main() -> None:
    # Publication profile: every figure this notebook writes is a print-sized
    # vector, emitted as both SVG (docs) and PDF (manuscript) by save_figure.
    theme.set_paper_mode(True)

    modal_gpu = parse_modal_gpu(sys.argv)
    skip_training = "--skip-training" in sys.argv
    wipe_dir = "--no-wipe-dir" not in sys.argv

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    n_cells = len(MODELS) * len(THETA_U_GRID)
    print(
        f"notebook_run_id = {notebook_run_id} cells={n_cells}"
        + ("  [skip-training]" if skip_training else "")
    )

    prepare_run_dirs(
        SLUG, notebook_run_id, wipe=wipe_dir, skip_training=skip_training,
        make_artifacts=False,
        scale=SCALE,
        host=f"modal:{modal_gpu}" if modal_gpu else "local",
    )

    # Training lives in nb022 now (train-once / reuse-many). This notebook
    # consumes the shared cells via cell_dir → nb022.load_cell.

    rows: list[dict] = []
    for model in MODELS:
        for theta_u in THETA_U_GRID:
            for seed in seeds_for(theta_u):
                run_dir = cell_dir(model, theta_u, seed)
                if not (run_dir / "metrics.json").exists():
                    raise SystemExit(f"missing metrics: {run_dir / 'metrics.json'}")
                metrics = load_metrics(run_dir)
                last = metrics["epochs"][-1]
                rows.append(
                    {
                        "model": model,
                        "theta_u": theta_u,
                        "theta_display": theta_display(theta_u),
                        "theta_u_hz": theta_hz(theta_u),
                        "seed": seed,
                        "best_acc": float(metrics["best_acc"]),
                        "best_epoch": int(metrics["best_epoch"]),
                        "final_acc": float(last["acc"]),
                        "rate_e": float(last.get("rate_e") or 0.0),
                    }
                )

    print("  results:")
    for r in rows:
        theta_str = (
            f"θ_u={r['theta_display']:>4} ({r['theta_u_hz']:>4.1f} Hz)"
            if r["theta_u"] is not None
            else "θ_u= off"
        )
        print(
            f"    {r['model']:<5}  {theta_str}  "
            f"acc(final)={r['final_acc']:6.2f}%  best={r['best_acc']:6.2f}%  "
            f"rate_e={r['rate_e']:6.1f} Hz"
        )

    # Hidden-layer perturbation sweep: drop spikes (Bernoulli mask) and
    # add Poisson noise spikes, applied inside the forward loop so the
    # I-population and readout both react.
    print("[perturb] hidden-spike drop + add sweep (coba, ping)")
    perturb_rows: list[dict] = []
    for model in MODELS:
        train_dir = baseline_dir(model)
        for mode, levels in (
            ("drop", PERTURB_DROP_LEVELS),
            ("add", PERTURB_ADD_LEVELS),
        ):
            for level in levels:
                res = run_perturbation_sweep(train_dir, mode, level)
                res["model"] = model
                perturb_rows.append(res)
                print(
                    f"  {model:<5} {mode:<4} level={level:>5.2f}  "
                    f"acc={res['acc']:5.2f}%  E={res['e_rate_hz']:6.2f} Hz"
                )
    plot_perturbation_curves(
        perturb_rows, FIGURES / "perturbation_curves", notebook_run_id
    )
    print(f"wrote {FIGURES / 'perturbation_curves'}.{{svg,pdf}}")

    # Stacked-raster snapshots of each trained baseline under both
    # perturbation modes, six panels per (model, mode). Same MNIST digit 0
    # sample 0 as the other rasters so the panels read against the
    # unperturbed baselines (Figures 4-5).
    for model in MODELS:
        train_dir = baseline_dir(model)
        drop_samples = [
            capture_perturbation_raster(train_dir, "drop", lvl, 0)
            for lvl in PERTURB_RASTER_DROP_LEVELS
        ]
        plot_perturbation_rasters(
            drop_samples,
            FIGURES / f"perturb_rasters__drop__{model}",
            notebook_run_id,
            level_fmt="p(drop) = {level:.1f}",
            title=(
                f"E (black) and I (red) spikes — trained {model} with "
                "hidden-spike drop"
            ),
        )
        print(f"wrote {FIGURES / f'perturb_rasters__drop__{model}'}.{{png,pdf}}")
        add_samples = [
            capture_perturbation_raster(train_dir, "add", lvl, 0)
            for lvl in PERTURB_RASTER_ADD_LEVELS
        ]
        plot_perturbation_rasters(
            add_samples,
            FIGURES / f"perturb_rasters__add__{model}",
            notebook_run_id,
            level_fmt="r(add) = {level:g} Hz",
            title=(
                f"E (black) and I (red) spikes — trained {model} with "
                "hidden-spike Poisson noise added"
            ),
        )
        print(f"wrote {FIGURES / f'perturb_rasters__add__{model}'}.{{png,pdf}}")

    duration_s = time.monotonic() - t_start
    train_cfg = load_config(baseline_dir(MODELS[0]))
    summary = {
        "notebook_run_id": notebook_run_id,
        "git_sha": train_cfg.get("git_sha"),
        "duration_s": round(duration_s, 1),
        "duration": format_duration(duration_s),
        "config": {
            "dataset": "mnist",
            "models": MODELS,
            "theta_u_grid_spikes": [t for t in THETA_U_GRID if t is not None],
            "theta_u_grid_hz": [
                theta_hz(t) for t in THETA_U_GRID if t is not None
            ],
            "max_samples": MAX_SAMPLES,
            "epochs": BASELINE_EPOCHS,
            "t_ms": T_MS,
            "dt": DT_TRAIN,
            "seeds_baseline": SEEDS_BASELINE,
            "seed_sweep": SEED_SWEEP,
            "fr_strength_upper": FR_STRENGTH_UPPER,
        },
        "baseline_results": rows,
        "perturbation": perturb_rows,
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {summary['duration']}")



if __name__ == "__main__":
    main()
