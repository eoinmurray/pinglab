"""Notebook runner for entry 021 — L2 (MSE) loss across COBA / PING.

Trains each model under two loss modes:

  ce  — cross-entropy on cumulative-membrane logits (the default across
        nb011–12).
  mse — L2 between logits and one-hot targets (Bohte 2002, Lee 2016).

Four cells (2 models × 2 loss modes). The CE-calibrated readout scales
(100 for coba, 500 for ping) put logits in O(100), which CE absorbs
through softmax but MSE chases as raw magnitude error — collapses the
network. A scale sweep ∈ {1, 2, 5, 10} (see numbers.json history) put
the optimum at scale=5 for both coba/mse and ping/mse, so we pin that
value for the MSE cells.

Notebook entry: src/docs/src/pages/notebooks/nb021.mdx
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _modal import BatchDispatcher, parse_modal_gpu  # noqa: E402
from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from _tier import parse_tier  # noqa: E402
from pinglab import theme  # noqa: E402

SLUG = "nb021"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
OSCILLOSCOPE = REPO / "src" / "pinglab" / "cli/__main__.py"

TIER_CONFIG = {
    "extra small": dict(max_samples=100, epochs=1),
    "small": dict(max_samples=500, epochs=5),
    "medium": dict(max_samples=2000, epochs=10),
    "large": dict(max_samples=5000, epochs=40),
    "extra large": dict(max_samples=10000, epochs=40),
}
DEFAULT_TIER = "small"
T_MS = 200.0
DT_TRAIN = 0.1
SEED = 42

MODELS = ["coba", "ping"]
LOSS_MODES = ["ce", "mse"]

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

# MSE on one-hot targets needs logits in O(1); the CE-calibrated readout
# scales (100 for coba, 500 for ping) over-shoot under L2. A sweep across
# {1, 2, 5, 10} put the optimum at 5 for both coba/mse and ping/mse —
# pinned here.
MSE_READOUT_SCALE = "5"
LOSS_RECIPE_OVERRIDES: dict[tuple[str, str], dict[str, str | bool | None]] = {
    ("coba", "mse"): {"--readout-w-out-scale": MSE_READOUT_SCALE},
    ("ping", "mse"): {"--readout-w-out-scale": MSE_READOUT_SCALE},
}

MODEL_COLORS = {
    "coba": theme.AMBER,
    "ping": theme.ELECTRIC_CYAN,
}
LOSS_HATCH = {"ce": "", "mse": "//"}

MIN_ACC_BY_TIER = {
    "extra small": 15.0,
    "small": 30.0,
    "medium": 50.0,
    "large": 70.0,
    "extra large": 70.0,
}


def cell_dir(model: str, loss_mode: str) -> Path:
    return ARTIFACTS / f"{model}__{loss_mode}"


def build_train_args(
    model: str, loss_mode: str, tier: str, out_dir: Path
) -> list[str]:
    recipe = MODEL_RECIPES[model]
    args = [
        "train",
        "--model", recipe["__build_as"],
        "--dataset", "mnist",
        "--max-samples", str(TIER_CONFIG[tier]["max_samples"]),
        "--epochs", str(TIER_CONFIG[tier]["epochs"]),
        "--t-ms", str(T_MS),
        "--dt", str(DT_TRAIN),
        "--seed", str(SEED),
        "--observe", "video",
        "--frame-rate", "1",
        "--loss", loss_mode,
        "--out-dir", str(out_dir),
        "--wipe-dir",
    ]
    merged: dict = {**recipe, **LOSS_RECIPE_OVERRIDES.get((model, loss_mode), {})}
    for k, v in merged.items():
        if k.startswith("__"):
            continue
        if v is True:
            args.append(k)
        elif v is not None:
            args += [k, v]
    return args


def load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text())


def load_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())


def _stamp(fig, run_id: str) -> None:
    fig.text(
        0.995, 0.005, run_id,
        ha="right", va="bottom",
        fontsize=theme.SIZE_CAPTION,
        color=theme.LABEL, family="monospace",
    )


def plot_accuracy_bars(rows: list[dict], out_path: Path, run_id: str) -> None:
    """Grouped bar chart: final test accuracy by (model, loss_mode)."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = list(range(len(MODELS)))
    width = 0.38
    for i, loss_mode in enumerate(LOSS_MODES):
        offsets = [xi + (i - 0.5) * width for xi in x]
        ys = []
        for model in MODELS:
            r = next(
                row for row in rows
                if row["model"] == model and row["loss_mode"] == loss_mode
            )
            ys.append(r["final_acc"])
        ax.bar(
            offsets, ys, width,
            color=[MODEL_COLORS[m] for m in MODELS],
            edgecolor=theme.INK_BLACK,
            hatch=LOSS_HATCH[loss_mode],
            label=loss_mode,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("test accuracy (%, final epoch)")
    ax.set_title("CE vs L2 loss — COBA and PING")
    ax.set_ylim(0, 100)
    ax.legend(title="loss")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def evaluate_success(rows: list[dict], tier: str, figures: Path) -> list[dict]:
    floor = float(MIN_ACC_BY_TIER[tier])
    figs_root = figures.parents[2]

    def artifact(name: str, label: str) -> dict:
        path = figures / name
        ok = path.exists() and path.stat().st_size > 0
        href = "/" + str(path.relative_to(figs_root)) if ok else None
        return {
            "label": label,
            "passed": bool(ok),
            "detail": f"{path.name} ({path.stat().st_size} bytes)"
            if ok else f"missing {path.name}",
            "detail_href": href,
        }

    crits: list[dict] = [artifact("accuracy.png", "accuracy figure rendered")]
    for model in MODELS:
        for loss_mode in LOSS_MODES:
            crits.append(
                artifact(
                    f"training__{model}__{loss_mode}.mp4",
                    f"{model}/{loss_mode}: training video",
                )
            )
    for model in MODELS:
        ce = next(
            r for r in rows if r["model"] == model and r["loss_mode"] == "ce"
        )
        crits.append(
            {
                "label": f"{model} ce baseline acc ≥ {floor:.0f}% ({tier} floor)",
                "passed": bool(ce["best_acc"] >= floor),
                "detail": f"{model}/ce={ce['best_acc']:.2f}%",
            }
        )
    for model in MODELS:
        mse = next(
            r for r in rows if r["model"] == model and r["loss_mode"] == "mse"
        )
        crits.append(
            {
                "label": f"{model} mse trains above chance",
                "passed": bool(mse["best_acc"] >= 15.0),
                "detail": f"{model}/mse={mse['best_acc']:.2f}%",
            }
        )
    return crits


def _format_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def copy_video(run_dir: Path, out_path: Path) -> None:
    src = run_dir / "training.mp4"
    if not src.exists():
        raise SystemExit(f"missing training video: {src}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out_path)
    print(f"wrote {out_path}")


def main() -> None:
    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    modal_gpu = parse_modal_gpu(sys.argv)
    skip_training = "--skip-training" in sys.argv
    wipe_dir = "--no-wipe-dir" not in sys.argv

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    n_cells = len(MODELS) * len(LOSS_MODES)
    print(
        f"notebook_run_id = {notebook_run_id} tier={tier} "
        f"cells={n_cells}"
        + ("  [skip-training]" if skip_training else "")
    )

    if wipe_dir:
        if skip_training:
            if FIGURES.exists():
                print(f"[wipe] {FIGURES.relative_to(REPO)}")
                shutil.rmtree(FIGURES)
        else:
            for d in (ARTIFACTS, FIGURES):
                if d.exists():
                    print(f"[wipe] {d.relative_to(REPO)}")
                    shutil.rmtree(d)
    FIGURES.mkdir(parents=True, exist_ok=True)
    persist_run_id(SLUG, notebook_run_id)

    if not skip_training:
        dispatcher = BatchDispatcher(modal_gpu, REPO, OSCILLOSCOPE)
        for model in MODELS:
            for loss_mode in LOSS_MODES:
                out = cell_dir(model, loss_mode)
                build_as = MODEL_RECIPES[model]["__build_as"]
                gpu_override = None
                if modal_gpu in ("T4", "L4", "A10G") and build_as == "ping":
                    gpu_override = "A100"
                print(
                    f"[train] {model}/{loss_mode} → "
                    f"{out.relative_to(REPO)}"
                    + (f"  [modal:{modal_gpu}]" if modal_gpu else "")
                )
                dispatcher.submit(
                    build_train_args(model, loss_mode, tier, out),
                    out,
                    gpu_override=gpu_override,
                )
        dispatcher.drain()

    rows: list[dict] = []
    for model in MODELS:
        for loss_mode in LOSS_MODES:
            run_dir = cell_dir(model, loss_mode)
            if not (run_dir / "metrics.json").exists():
                raise SystemExit(f"missing metrics: {run_dir / 'metrics.json'}")
            metrics = load_metrics(run_dir)
            last = metrics["epochs"][-1]
            rows.append(
                {
                    "model": model,
                    "loss_mode": loss_mode,
                    "best_acc": float(metrics["best_acc"]),
                    "best_epoch": int(metrics["best_epoch"]),
                    "final_acc": float(last["acc"]),
                    "rate_e": float(last.get("rate_e") or 0.0),
                }
            )
            copy_video(
                run_dir,
                FIGURES / f"training__{model}__{loss_mode}.mp4",
            )

    print("  results:")
    for r in rows:
        print(
            f"    {r['model']:<5}  loss={r['loss_mode']:<3}  "
            f"acc(final)={r['final_acc']:6.2f}%  best={r['best_acc']:6.2f}%  "
            f"rate_e={r['rate_e']:6.1f} Hz"
        )

    plot_accuracy_bars(rows, FIGURES / "accuracy.png", notebook_run_id)
    print(f"wrote {FIGURES / 'accuracy.png'}")

    duration_s = time.monotonic() - t_start
    train_cfg = load_config(cell_dir(MODELS[0], LOSS_MODES[0]))
    crits = evaluate_success(rows, tier, FIGURES)
    summary = {
        "notebook_run_id": notebook_run_id,
        "git_sha": train_cfg.get("git_sha"),
        "duration_s": round(duration_s, 1),
        "duration": _format_duration(duration_s),
        "tier": tier,
        "config": {
            "tier": tier,
            "dataset": "mnist",
            "models": MODELS,
            "loss_modes": LOSS_MODES,
            "mse_readout_scale": MSE_READOUT_SCALE,
            "max_samples": TIER_CONFIG[tier]["max_samples"],
            "epochs": TIER_CONFIG[tier]["epochs"],
            "t_ms": T_MS,
            "dt": DT_TRAIN,
            "seed": SEED,
        },
        "results": rows,
        "success_criteria": crits,
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {summary['duration']}")

    for c in crits:
        mark = "pass" if c["passed"] else "FAIL"
        print(f"  [{mark}] {c['label']} — {c['detail']}")
    if any(not c["passed"] for c in crits):
        sys.exit(1)


if __name__ == "__main__":
    main()
