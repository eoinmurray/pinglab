"""Notebook runner for entry 063 — Training.

The single place the gamma-gated-sparsity collection trains its canonical
networks. Every cell is trained once here, to a shared artifact root
(src/artifacts/notebooks/training/), and the analysis notebooks load those
weights with `load_cell` (imported from this module) instead of retraining
their own. This replaces the collection's older "standalone runner, no
cross-notebook helpers" rule with a train-once / reuse-many policy (see ar016).

Phase 1 owns the canonical coba / ping θ_u spike-budget sweep that nb025
defines and that nb024 / nb036 / nb037 / nb038 each used to retrain
independently: θ_u ∈ {off, 5, 2, 1, 0.5, 0.2}, baselines (off) at seeds
42/43/44 and sweep cells single-seed. Standard: 100 epochs, dt = 0.1 ms,
T = 200 ms, MNIST. (nb044's dt sweep is the one documented exception that
varies dt; it owns its own training.)

Outputs a per-cell accuracy / E-rate summary plus a manifest (numbers.json)
recording exactly which cells were trained and the git sha — the contract
the analysis notebooks rely on.

Notebook entry: src/docs/content/notebooks/nb022.mdx
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from helpers import theme  # noqa: E402
from helpers.fmt import format_duration  # noqa: E402
from helpers.modal import BatchDispatcher, parse_modal_gpu  # noqa: E402
from helpers.operating_point import TAU_GABA_GAMMA_MS  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import stamp_figure  # noqa: E402

SLUG = "nb022"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)

# ── Canonical training registry (the hub the collection reuses) ──────
# Analysis notebooks import `load_cell` / `cell_dir` from this module rather
# than retraining; this entry is the single producer of the shared cells.
TRAINING_ROOT = REPO / "src" / "artifacts" / "notebooks" / "training"
PINGLAB_CLI = REPO / "src" / "cli" / "cli.py"

EPOCHS_STANDARD = 100          # standard depth (dt sweep in nb044 is the exception)
DT_MS = 0.1
T_MS = 200.0
SEEDS_BASELINE = [42, 43, 44]
SEED_SWEEP = 42
THETA_U_GRID: list[float | None] = [None, 5.0, 2.0, 1.0, 0.5, 0.2]
FR_STRENGTH_UPPER = 1e-3
TAU_AMPA_MS = 2.0          # AMPA decay — fixed across the collection (no CLI knob)
# GABA decay that puts the loop in gamma (≈ 44 Hz); the standard for every
# family except the τ_GABA sweep. Single source of truth in helpers so the
# whole collection moves together (see helpers/operating_point.py).
TAU_GABA_GAMMA = TAU_GABA_GAMMA_MS

# Canonical recipe (from nb025 — the reference training).
# Gradient dampening (--v-grad-dampen) is loop-specific: [nb064](nb064) showed
# PING needs it (its BPTT gradient explodes through the E→I→E loop without it),
# while COBA is insensitive and trains identically at dampening 1. So COBA trains
# with NO dampening (1) and PING keeps the stabiliser (1000). Training COBA
# without the crutch keeps the two architectures honest — COBA earns its accuracy
# on the bare feedforward gradient.
MODEL_RECIPES: dict[str, dict] = {
    "coba": {
        "__build_as": "ping",
        "--ei-strength": "0",
        "--v-grad-dampen": "1",
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
MODELS = ["coba", "ping"]
MODEL_COLORS = {"coba": theme.DEEP_RED, "ping": theme.INK_BLACK}
MODEL_MARKERS = {"coba": "s", "ping": "D"}

# ping recipe without the fixed --ei-strength, for the init family (nb049),
# whose whole point is to vary ei-strength + recurrent trainability per cell.
MODEL_RECIPES["ping_init"] = {
    k: v for k, v in MODEL_RECIPES["ping"].items() if k != "--ei-strength"
}

# nb049 init conditions: (ei_strength, trainable W_EI, trainable W_IE).
INIT_CONDITIONS: dict[str, tuple] = {
    "frozen_ping": ("1", False, False),
    "trainable_ping_init": ("1", True, True),
    "trainable_zero_init": ("0", True, True),
    "trainable_small_init": ("0.1", True, True),
}


TAU_GABA_SWEEP = (4.5, 6.0, 9.0, 12.0, 18.0, 27.0)   # nb041
DT_SWEEP_MS = (0.05, 0.1, 0.25, 0.5, 1.0)             # nb044 (the dt exception)
MNIST_POOL = 70000                                   # 60k train + 10k test pool
CANONICAL_MAX_SAMPLES = MNIST_POOL                    # the canonical run sees all of it
SUBSET_MAX_SAMPLES = MNIST_POOL // 10                 # 10% of MNIST for the sweeps
MAX_SAMPLES = 100                                     # plumbing cap on every cell
EPOCHS = 2                                            # plumbing depth on every cell
BATCH_SIZE = 256                                      # fixed across every recipe


def theta_label(theta_u: float | None) -> str:
    if theta_u is None:
        return "off"
    return "tu" + f"{theta_u:g}".replace(".", "p")


def theta_display(theta_u: float | None) -> str:
    return "off" if theta_u is None else f"{theta_u:g}"


def seeds_for(theta_u: float | None) -> list[int]:
    """Baselines run all seeds; sweep cells stay single-seed (matches nb025)."""
    return list(SEEDS_BASELINE) if theta_u is None else [SEED_SWEEP]


def cell_name(model: str, theta_u: float | None, seed: int) -> str:
    """θ_u cell name. Matches nb025/nb036/nb037/nb038, so migrating those is a
    path repoint, not a rename."""
    if theta_u is None:
        return f"{model}__off__seed{seed}"
    return f"{model}__{theta_label(theta_u)}"


def _label(x: float) -> str:
    return f"{x:g}".replace(".", "p")


# ── Cell registry: one spec per trained cell, tagged by family ───────
# Each spec carries the family-specific bits (dt override + extra flags) so a
# single build_train_args reproduces every family. Names match the existing
# per-notebook artifacts, so folding the already-trained cells in is a move,
# not a retrain.

def _theta_u_cells() -> list[dict]:
    cells = []
    for m in MODELS:
        for tu in THETA_U_GRID:
            extra = ([] if tu is None else
                     ["--fr-reg-upper-theta", str(tu),
                      "--fr-reg-upper-strength", str(FR_STRENGTH_UPPER)])
            for s in seeds_for(tu):
                cells.append({
                    "name": cell_name(m, tu, s), "model": m, "family": "theta_u",
                    "tag": theta_display(tu), "seed": s, "dt_ms": DT_MS,
                    "tau_gaba": TAU_GABA_GAMMA, "extra": extra,
                })
    return cells


def _tau_gaba_cells() -> list[dict]:
    return [
        {"name": f"ping__tg{_label(tau)}__seed{s}", "model": "ping",
         "family": "tau_gaba", "tag": f"τ={tau:g}", "seed": s, "dt_ms": DT_MS,
         "tau_gaba": tau, "extra": []}
        for tau in TAU_GABA_SWEEP for s in SEEDS_BASELINE
    ]


def _dt_cells() -> list[dict]:
    # The dt sweep is the documented exception that varies dt by design.
    return [
        {"name": f"ping__dt{_label(dt)}__seed{s}", "model": "ping",
         "family": "dt", "tag": f"dt={dt:g}", "seed": s, "dt_ms": dt,
         "tau_gaba": TAU_GABA_GAMMA, "extra": []}
        for dt in DT_SWEEP_MS for s in SEEDS_BASELINE
    ]


def _canonical_cells() -> list[dict]:
    # The canonical reference: θ_u = off, trained on ALL of MNIST (not the
    # subset the other families use) once the full standard is restored.
    return [
        {"name": f"{m}__canonical__seed{s}", "model": m, "family": "canonical",
         "tag": "off · all MNIST", "seed": s, "dt_ms": DT_MS, "extra": [],
         "tau_gaba": TAU_GABA_GAMMA, "max_samples": CANONICAL_MAX_SAMPLES}
        for m in MODELS for s in SEEDS_BASELINE
    ]


def _init_cells() -> list[dict]:
    cells = []
    for cond, (ei, t_ei, t_ie) in INIT_CONDITIONS.items():
        extra = ["--ei-strength", ei]
        if t_ei:
            extra.append("--trainable-w-ei")
        if t_ie:
            extra.append("--trainable-w-ie")
        for s in SEEDS_BASELINE:
            cells.append({
                "name": f"{cond}__seed{s}", "model": "ping_init",
                "family": "init", "tag": cond, "seed": s, "dt_ms": DT_MS,
                "tau_gaba": TAU_GABA_GAMMA, "extra": extra,
            })
    return cells


def _theta_u_3seed_cells() -> list[dict]:
    """3-seed extension of the θ_u sweep: interior θ_u values re-run across
    all baseline seeds so the ar009 Figure 3 accuracy–rate frontier can be
    plotted with across-seed bands (currently the interior is single-seed,
    a disclosure in ar009 §2.3 we'd like to remove). The 'off' baseline is
    not included here — _theta_u_cells already runs it at all baseline seeds.

    NOT YET WIRED INTO CANONICAL_CELLS — see ar010 (manuscript-driven TODO
    article, item 3 after the items-1-and-2 merge). To enable training, append
    ` + _theta_u_3seed_cells()` to the CANONICAL_CELLS expression below.
    Names use the {model}__tu{val}__seedN convention so they sit alongside
    rather than overwrite the existing single-seed sweep cells.
    """
    cells = []
    for m in MODELS:
        for tu in THETA_U_GRID:
            if tu is None:
                continue
            extra = [
                "--fr-reg-upper-theta", str(tu),
                "--fr-reg-upper-strength", str(FR_STRENGTH_UPPER),
            ]
            for s in SEEDS_BASELINE:
                cells.append({
                    "name": f"{m}__{theta_label(tu)}__seed{s}",
                    "model": m, "family": "theta_u_3seed",
                    "tag": theta_display(tu), "seed": s, "dt_ms": DT_MS,
                    "tau_gaba": TAU_GABA_GAMMA, "extra": extra,
                })
    return cells


# Available but not enabled — see docstring above. Importable from outside
# the module for inspection / dry-run; flip the switch by adding to
# CANONICAL_CELLS below.
THETA_U_3SEED_CELLS = _theta_u_3seed_cells()


CANONICAL_CELLS = (_canonical_cells() + _theta_u_cells() + _tau_gaba_cells()
                   + _dt_cells() + _init_cells())

# Run scale — stamped into the manifest by run_dirs.prepare and rendered as
# the Methods table via RunScale; the mdx never restates these numbers.
SCALE = {
    "dataset": "mnist",
    "max_samples": MAX_SAMPLES,
    "epochs": EPOCHS,
    "t_ms": T_MS,
    "dt_ms": DT_MS,
    "batch_size": BATCH_SIZE,
    "seeds": len(SEEDS_BASELINE),
    "cells": len(CANONICAL_CELLS),
    "grid": "2 architectures × 5 families",
}


def cell_dir(name: str) -> Path:
    """Shared per-cell artifact directory."""
    return TRAINING_ROOT / name


def load_cell(name: str) -> Path:
    """Return a trained cell's directory, or fail loudly if this notebook has
    not been run. Analysis notebooks call this instead of training."""
    d = cell_dir(name)
    if not (d / "weights.pth").exists():
        raise SystemExit(
            f"missing trained cell '{name}' at {d.relative_to(REPO)}; "
            "run nb022 (Training) first to produce the shared cells."
        )
    return d


def build_train_args(spec: dict, out_dir: Path,
                     max_samples: int, epochs: int) -> list[str]:
    """CLI `train` args for one registry cell, across all families."""
    recipe = MODEL_RECIPES[spec["model"]]
    ms = spec.get("max_samples") or max_samples   # canonical cells override
    args = [
        "train",
        "--model", recipe["__build_as"],
        "--dataset", "mnist",
        "--max-samples", str(ms),
        "--epochs", str(epochs),
        "--t-ms", str(T_MS),
        "--dt", str(spec["dt_ms"]),
        "--tau-gaba", str(spec["tau_gaba"]),
        "--seed", str(spec["seed"]),
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
    args += spec["extra"]
    return args


# ── Runner ───────────────────────────────────────────────────────────

# This runner is pinned to the plumbing scale: every cell is capped to
# MAX_SAMPLES samples and EPOCHS epochs so the whole registry can be trained
# in minutes to check the wiring. The full per-family standard (canonical =
# all MNIST, sweeps = 10%, EPOCHS_STANDARD) lives in the constants above and
# is restored by editing cell_samples_epochs when a real run is scheduled.


def cell_samples_epochs(spec: dict) -> tuple[int, int]:
    """Per-cell (max_samples, epochs): the fixed plumbing cap for every cell."""
    return MAX_SAMPLES, EPOCHS


def _json_safe(o):
    """Replace non-finite floats (NaN/inf from untrained cells) with None so
    the manifest is valid JSON the docs loader can parse."""
    if isinstance(o, float):
        return o if (o == o and abs(o) != float("inf")) else None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_json_safe(v) for v in o]
    return o


def load_metrics(d: Path) -> dict:
    p = d / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


def final_rates(d: Path) -> tuple[float, float]:
    """Last-epoch E / I rate (Hz) from metrics.jsonl, if present."""
    p = d / "metrics.jsonl"
    if not p.exists():
        return float("nan"), float("nan")
    lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
    if not lines:
        return float("nan"), float("nan")
    row = json.loads(lines[-1])
    return (float(row.get("test_rate_e", row.get("rate_e", float("nan")))),
            float(row.get("test_rate_i", row.get("rate_i", float("nan")))))


FAMILY_COLORS = {
    "canonical": theme.GREY_DARK,
    "theta_u": theme.INK_BLACK,
    "theta_u_3seed": theme.INK_BLACK,
    "tau_gaba": theme.DEEP_RED,
    "dt": theme.ELECTRIC_CYAN,
    "init": theme.AMBER,
}


def training_curve(d: Path) -> tuple[list[int], list[float]]:
    """Per-epoch (epoch, test accuracy) from a cell's metrics.jsonl."""
    p = d / "metrics.jsonl"
    if not p.exists():
        return [], []
    eps, accs = [], []
    for ln in p.read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        if "ep" in r and "acc" in r:
            eps.append(int(r["ep"]))
            accs.append(float(r["acc"]))
    return eps, accs


FAMILY_ORDER = ["canonical", "theta_u", "theta_u_3seed", "tau_gaba", "dt", "init"]
FAMILY_LABELS = {
    "canonical": "Canonical reference",
    "theta_u": "θ_u spike-budget sweep",
    "theta_u_3seed": "θ_u spike-budget sweep (3-seed, ar010 item 3)",
    "tau_gaba": "τ_GABA ladder",
    "dt": "Δt sweep",
    "init": "Init variants",
}


def plot_family_curves(family: str, cells: list[dict],
                       out_path: Path, run_id: str) -> int:
    """One figure for one family: each cell's test-accuracy learning curve,
    coloured by the swept value. Returns the number of cells actually drawn."""
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D

    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"
    tags = list(dict.fromkeys(c["tag"] for c in cells))  # ordered unique
    # cm.viridis exists at runtime; the matplotlib stub omits it (false positive).
    colours = {t: cm.viridis(i / max(1, len(tags) - 1))  # ty: ignore[unresolved-attribute]
               for i, t in enumerate(tags)}
    # ping (and ping-init) solid, coba dashed — distinguishes the two models
    # in families that train both (θ_u, canonical).
    linestyle = {"coba": "--", "ping": "-", "ping_init": "-"}
    models = list(dict.fromkeys(c["model"] for c in cells))

    fig, ax = plt.subplots(figsize=(12.0, 6.75), dpi=150)
    n = 0
    for c in cells:
        eps, accs = training_curve(cell_dir(c["name"]))
        if eps:
            ax.plot(eps, accs, lw=1.1, color=colours[c["tag"]],
                    ls=linestyle.get(c["model"], "-"), alpha=0.85)
            n += 1
    handles = [Line2D([0], [0], color=colours[t], lw=2.4, label=t) for t in tags]
    leg1 = ax.legend(handles=handles, frameon=False, fontsize=theme.SIZE_LEGEND,
                     ncol=2, loc="lower right", title="swept value")
    ax.add_artist(leg1)
    if len(models) > 1:
        mh = [Line2D([0], [0], color=theme.MUTED, lw=2.0,
                     ls=linestyle.get(m, "-"), label="ping" if m == "ping_init" else m)
              for m in models]
        ax.legend(handles=mh, frameon=False, fontsize=theme.SIZE_LEGEND,
                  loc="lower center", title="model")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 100)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"{FAMILY_LABELS[family]} — {n}/{len(cells)} cells trained",
                 loc="left", fontweight="semibold", fontsize=theme.SIZE_LABEL)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return n


def main() -> None:
    modal_gpu = parse_modal_gpu(sys.argv)
    skip_training = "--skip-training" in sys.argv
    only_missing = "--only-missing" in sys.argv

    t_start = time.monotonic()
    run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {run_id} "
          f"cells={len(CANONICAL_CELLS)}"
          + ("  [skip-training]" if skip_training else ""))
    # Wipe only this entry's figures, never the shared TRAINING_ROOT.
    prepare_run_dirs(SLUG, run_id, wipe=True, skip_training=skip_training,
                     make_artifacts=False, scale=SCALE,
                     host=f"modal:{modal_gpu}" if modal_gpu else "local")

    if not skip_training:
        TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
        dispatcher = BatchDispatcher(modal_gpu, REPO, PINGLAB_CLI)
        for c in CANONICAL_CELLS:
            out = cell_dir(c["name"])
            if only_missing and (out / "metrics.json").exists():
                print(f"[skip] {c['name']} already trained")
                continue
            ms, ep = cell_samples_epochs(c)
            gpu_override = "A100" if modal_gpu in ("T4", "L4", "A10G") else None
            print(f"[train] {c['name']} (n={ms}, {ep} ep) → {out.relative_to(REPO)}"
                  + (f"  [modal:{modal_gpu}]" if modal_gpu else ""))
            dispatcher.submit(
                build_train_args(c, out, ms, ep),
                out, gpu_override=gpu_override,
            )
        dispatcher.drain()

    rows = []
    for c in CANONICAL_CELLS:
        d = cell_dir(c["name"])
        m = load_metrics(d)
        re, ri = final_rates(d)
        rows.append({
            "name": c["name"], "model": c["model"], "family": c["family"],
            "tag": c["tag"], "seed": c["seed"],
            "acc": float(m.get("best_acc", float("nan"))),
            "best_epoch": m.get("best_epoch"), "rate_e": re, "rate_i": ri,
        })
        print(f"  {c['name']:<22} acc={rows[-1]['acc']:5.1f}%  "
              f"E={re:5.1f}Hz I={ri:5.1f}Hz")

    # One training-curve figure per family. Untrained families get no figure,
    # so the entry's <Figure> shows its "not generated yet" placeholder.
    family_status = {}
    for fam in FAMILY_ORDER:
        fcells = [c for c in CANONICAL_CELLS if c["family"] == fam]
        n_trained = sum(1 for c in fcells
                        if (cell_dir(c["name"]) / "metrics.jsonl").exists())
        family_status[fam] = {"cells": len(fcells), "trained": n_trained}
        out = FIGURES / f"curves__{fam}.png"
        if n_trained:
            plot_family_curves(fam, fcells, out, run_id)
            print(f"wrote {out}")
        else:
            print(f"[not trained] {fam} — no figure (placeholder shown)")

    duration_s = time.monotonic() - t_start
    # rows is a heterogeneous list of dicts, so r["name"] widens to a union;
    # coerce to str for cell_dir (the value is always the cell-name string).
    git_sha = next((c for c in (load_metrics(cell_dir(str(r["name"]))).get("config", {})
                                 for r in rows) if c), {}).get("git_sha")
    summary = {
        "notebook_run_id": run_id,
        "git_sha": git_sha,
        "duration_s": round(duration_s, 1),
        "duration": format_duration(duration_s),
        "standard": {"epochs": EPOCHS_STANDARD, "dt_ms": DT_MS, "t_ms": T_MS,
                     "dataset": "mnist",
                     "max_samples_canonical": CANONICAL_MAX_SAMPLES,
                     "max_samples_sweeps": SUBSET_MAX_SAMPLES},
        "training_root": str(TRAINING_ROOT.relative_to(REPO)),
        "families": FAMILY_ORDER,
        "family_status": family_status,
        "n_cells": len(CANONICAL_CELLS),
        "cells": rows,
    }
    (FIGURES / "numbers.json").write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {summary['duration']}")


if __name__ == "__main__":
    main()
