"""Notebook runner for entry 043 — rate-maximising pressure test.

Trains PING and COBA-no-loop networks across a signed rate-target axis:
upper-bound (penalty for firing above θ_u) → baseline (no regulariser)
→ lower-bound (penalty for firing below θ_l). The lower-bound side is
the new direction — gradient descent is *rewarded* for emitting more
spikes. The forbidding-claim signature is PING flatlining at its
structural bound while COBA tracks the target in both directions.

Reuses the symmetric --fr-reg-lower-* CLI added to oscilloscope train.

Notebook entry: src/docs/src/pages/notebooks/nb043.mdx
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _modal import BatchDispatcher, parse_modal_gpu  # noqa: E402
from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from _tier import parse_tier  # noqa: E402
from pinglab import theme  # noqa: E402

SLUG = "nb043"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
OSCILLOSCOPE = REPO / "src" / "pinglab" / "cli" / "__main__.py"

T_MS = 200.0
DT_TRAIN = 0.1
SPIKES_PER_TRIAL_TO_HZ = 1000.0 / T_MS  # at T=200ms, 1 spike/trial = 5 Hz

SEED: int = 42

TIER_CONFIG = {
    "extra small": dict(max_samples=100, epochs=2),
    "small": dict(max_samples=500, epochs=10),
    # Medium tier upgraded 2026-06-03 to match nb024's convergence
    # finding. The PING-reward cells (the load-bearing ones) were
    # still climbing at epoch 30; 100 epochs gives the converged
    # rate-vs-accuracy trade-off claimed by the entry.
    "medium": dict(max_samples=2000, epochs=100),
    "large": dict(max_samples=5000, epochs=100),
    "extra large": dict(max_samples=10000, epochs=100),
}
DEFAULT_TIER = "small"

# Signed rate-target axis. Positive = upper-bound (penalty above θ_u);
# negative = lower-bound (penalty below θ_l, "reward for spikes"); 0 =
# no regulariser. Stored as spikes/trial; ×5 gives Hz at T=200ms.
TARGETS_SPIKES: tuple[float, ...] = (
    +1.0,    # upper-bound at  5 Hz target
    +2.0,    # upper-bound at 10 Hz target
    +5.0,    # upper-bound at 25 Hz target
     0.0,    # baseline (no regulariser)
    -4.0,    # lower-bound at 20 Hz target
    -10.0,   # lower-bound at 50 Hz target
    -20.0,   # lower-bound at 100 Hz target
)
FR_STRENGTH = 1e-3

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

MODEL_COLORS = {"coba": theme.DEEP_RED, "ping": theme.INK_BLACK}
MODEL_MARKERS = {"coba": "s", "ping": "D"}


def target_label(target_spikes: float) -> str:
    """Filesystem-safe label."""
    if target_spikes == 0:
        return "off"
    sign = "u" if target_spikes > 0 else "l"
    s = f"{abs(target_spikes):g}".replace(".", "p")
    return f"{sign}{s}"


def cell_dir(model: str, target_spikes: float) -> Path:
    return ARTIFACTS / f"{model}__{target_label(target_spikes)}"


def build_train_args(
    model: str, target_spikes: float, tier: str, out_dir: Path
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
    if target_spikes > 0:
        args += [
            "--fr-reg-upper-theta", str(target_spikes),
            "--fr-reg-upper-strength", str(FR_STRENGTH),
        ]
    elif target_spikes < 0:
        args += [
            "--fr-reg-lower-theta", str(abs(target_spikes)),
            "--fr-reg-lower-strength", str(FR_STRENGTH),
        ]
    return args


def load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text())


def load_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())


def _stamp(fig, run_id: str) -> None:
    fig.text(
        0.995, 0.005, run_id,
        ha="right", va="bottom",
        fontsize=theme.SIZE_CAPTION, color=theme.LABEL, family="monospace",
    )


def plot_pressure_sweep(
    rows: list[dict], out_path: Path, run_id: str,
    ping_baseline_rate: float,
) -> None:
    """E rate (y, left axis) and accuracy (y, right axis) vs signed
    rate-target axis (x). One curve per model; horizontal line at the
    PING structural bound."""
    theme.apply()
    fig, (ax_rate, ax_acc) = plt.subplots(
        2, 1, figsize=(8.0, 6.0), dpi=150, sharex=True,
        gridspec_kw={"hspace": 0.12, "height_ratios": [1.6, 1.0]},
    )
    for model in MODELS:
        sub = [r for r in rows if r["model"] == model]
        sub.sort(key=lambda r: r["target_hz_signed"])
        xs = [r["target_hz_signed"] for r in sub]
        ys_rate = [r["e_rate_hz"] for r in sub]
        ys_acc = [r["acc"] for r in sub]
        ax_rate.plot(
            xs, ys_rate, marker=MODEL_MARKERS[model], markersize=6,
            linewidth=1.4, color=MODEL_COLORS[model], label=model,
        )
        ax_acc.plot(
            xs, ys_acc, marker=MODEL_MARKERS[model], markersize=6,
            linewidth=1.4, color=MODEL_COLORS[model], label=model,
        )
    # Structural bound annotation on the E-rate panel.
    ax_rate.axhline(
        ping_baseline_rate, color=theme.MUTED, lw=0.7, ls="--", alpha=0.7,
    )
    xlim = ax_rate.get_xlim()
    ax_rate.text(
        xlim[1], ping_baseline_rate + 0.5,
        f"  PING structural bound ≈ {ping_baseline_rate:.1f} Hz",
        fontsize=theme.SIZE_ANNOTATION, color=theme.MUTED,
        ha="right", va="bottom",
    )
    ax_rate.axvline(0, color=theme.GREY_MID, lw=0.5, ls=":")
    ax_acc.axvline(0, color=theme.GREY_MID, lw=0.5, ls=":")
    ax_acc.axhline(10.0, color=theme.GREY_MID, lw=0.5, ls=":", alpha=0.6)
    ax_rate.set_ylabel("Hidden E rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax_acc.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax_acc.set_xlabel(
        "Signed rate target (Hz) — positive = upper-bound (penalty); "
        "negative = lower-bound (reward)",
        fontsize=theme.SIZE_LABEL,
    )
    ax_acc.set_ylim(0, 100)
    for ax in (ax_rate, ax_acc):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=theme.SIZE_TICK)
    ax_rate.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper left")
    fig.suptitle(
        "Rate-maximising pressure — E rate and accuracy vs signed rate target",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curves(out_path: Path, run_id: str) -> None:
    """Per-cell training-trajectory curves so convergence is auditable.
    2 rows (model = coba, ping) × 2 cols (test acc, test E rate).
    Colour by signed rate target on a diverging scale: blue = upper-
    bound (penalty), red = lower-bound (reward), grey = baseline."""
    theme.apply()
    fig, axes = plt.subplots(
        len(MODELS), 2, figsize=(10.0, 6.0), dpi=150, sharex=True,
        gridspec_kw={"hspace": 0.15, "wspace": 0.18},
    )
    cmap_neg = plt.get_cmap("Reds")
    cmap_pos = plt.get_cmap("Blues")
    negs = sorted([t for t in TARGETS_SPIKES if t < 0])
    poss = sorted([t for t in TARGETS_SPIKES if t > 0])
    def color_for(target: float):
        if target == 0:
            return theme.GREY_MID
        if target > 0:
            idx = poss.index(target)
            return cmap_pos(0.4 + 0.55 * idx / max(1, len(poss) - 1))
        idx = negs.index(target)
        return cmap_neg(0.4 + 0.55 * (len(negs) - 1 - idx) / max(1, len(negs) - 1))

    for row, model in enumerate(MODELS):
        ax_acc = axes[row][0]
        ax_rate = axes[row][1]
        for target in TARGETS_SPIKES:
            mfile = cell_dir(model, target) / "metrics.json"
            if not mfile.exists():
                continue
            m = json.loads(mfile.read_text())
            eps = [e["ep"] for e in m["epochs"]]
            accs = [e.get("acc", 0) for e in m["epochs"]]
            rates = [e.get("test_rate_e", 0) for e in m["epochs"]]
            target_hz = target * SPIKES_PER_TRIAL_TO_HZ
            label = (
                f"target = {target_hz:+g} Hz"
                if target != 0 else "no regulariser"
            )
            color = color_for(target)
            ax_acc.plot(eps, accs, color=color, lw=1.0, alpha=0.85, label=label)
            ax_rate.plot(eps, rates, color=color, lw=1.0, alpha=0.85)
        ax_acc.set_ylabel(f"{model}\nTest accuracy (%)", fontsize=theme.SIZE_LABEL)
        ax_rate.set_ylabel("Test E rate (Hz)", fontsize=theme.SIZE_LABEL)
        ax_acc.set_ylim(0, 100)
        for ax in (ax_acc, ax_rate):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.15, lw=0.4)
        if row == 0:
            ax_acc.legend(
                fontsize=theme.SIZE_LEGEND, frameon=False, ncol=2,
                loc="lower right",
            )
    axes[-1][0].set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
    axes[-1][1].set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
    fig.suptitle(
        "Per-cell training curves — convergence check across signed-target sweep",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_success(
    rows: list[dict], ping_baseline_rate: float, figures: Path
) -> list[dict]:
    figs_root = figures.parents[2]

    def artifact(name: str, label: str) -> dict:
        path = figures / name
        ok = path.exists() and path.stat().st_size > 0
        href = "/" + str(path.relative_to(figs_root)) if ok else None
        return {
            "label": label,
            "passed": bool(ok),
            "detail": (
                f"{path.name} ({path.stat().st_size} bytes)"
                if ok else f"missing {path.name}"
            ),
            "detail_href": href,
        }

    crits: list[dict] = [artifact("pressure_sweep.png", "pressure sweep rendered")]

    coba_baseline = next(
        r for r in rows if r["model"] == "coba" and r["target_spikes"] == 0
    )
    # Comparator sanity — COBA baseline has to actually be a working
    # classifier with non-trivial firing, else the "rate climbs under
    # reward" check is degenerate. nb025's medium-tier COBA reaches
    # ≈ 88% / 88 Hz; for smaller tiers we just require above-chance.
    crits.append({
        "label": "COBA baseline accuracy > 20% (comparator non-degenerate)",
        "passed": bool(coba_baseline["acc"] > 20.0),
        "detail": (
            f"COBA baseline acc = {coba_baseline['acc']:.2f}%, "
            f"E = {coba_baseline['e_rate_hz']:.2f} Hz"
        ),
    })

    # Forbidding signature: the COBA span across the signed axis is at
    # least 5× the PING span. The headline isn't "absolute COBA peak vs
    # baseline" — it's that one curve responds to the Lagrangian and the
    # other does not.
    def span(model: str) -> tuple[float, float, float]:
        rates = [r["e_rate_hz"] for r in rows if r["model"] == model]
        lo, hi = min(rates), max(rates)
        return lo, hi, (hi / max(lo, 1e-3))

    ping_lo, ping_hi, ping_ratio = span("ping")
    coba_lo, coba_hi, coba_ratio = span("coba")

    crits.append({
        "label": "PING span across sweep stays within 2× of baseline",
        "passed": bool(ping_ratio <= 2.0),
        "detail": (
            f"PING E range = [{ping_lo:.2f}, {ping_hi:.2f}] Hz "
            f"(span {ping_ratio:.2f}×)"
        ),
    })
    crits.append({
        "label": "COBA span across sweep covers at least 5×",
        "passed": bool(coba_ratio >= 5.0),
        "detail": (
            f"COBA E range = [{coba_lo:.2f}, {coba_hi:.2f}] Hz "
            f"(span {coba_ratio:.2f}×)"
        ),
    })
    crits.append({
        "label": "COBA span ≥ 5× PING span (architectures differentiated)",
        "passed": bool(coba_ratio >= 5.0 * ping_ratio),
        "detail": f"COBA span {coba_ratio:.2f}× vs PING span {ping_ratio:.2f}×",
    })
    return crits


def _format_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def main() -> None:
    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    modal_gpu = parse_modal_gpu(sys.argv)
    skip_training = "--skip-training" in sys.argv
    only_missing = "--only-missing" in sys.argv
    wipe_dir = "--no-wipe-dir" not in sys.argv

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    n_cells = len(MODELS) * len(TARGETS_SPIKES)
    print(
        f"notebook_run_id = {notebook_run_id} tier={tier} cells={n_cells}"
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
            for target in TARGETS_SPIKES:
                out = cell_dir(model, target)
                if only_missing and (out / "metrics.json").exists():
                    print(
                        f"[skip] {model}/target={target:+g} spikes → "
                        f"already trained at {out.relative_to(REPO)}"
                    )
                    continue
                target_hz_signed = target * SPIKES_PER_TRIAL_TO_HZ
                print(
                    f"[train] {model}/target={target:+g} spikes "
                    f"({target_hz_signed:+.1f} Hz) → {out.relative_to(REPO)}"
                    + (f"  [modal:{modal_gpu}]" if modal_gpu else "")
                )
                dispatcher.submit(
                    build_train_args(model, target, tier, out),
                    out,
                )
        dispatcher.drain()

    rows: list[dict] = []
    for model in MODELS:
        for target in TARGETS_SPIKES:
            run_dir = cell_dir(model, target)
            if not (run_dir / "metrics.json").exists():
                raise SystemExit(f"missing metrics: {run_dir / 'metrics.json'}")
            metrics = load_metrics(run_dir)
            last = metrics["epochs"][-1]
            rows.append({
                "model": model,
                "target_spikes": float(target),
                "target_hz_signed": float(target * SPIKES_PER_TRIAL_TO_HZ),
                "regulariser_mode": (
                    "upper" if target > 0
                    else "lower" if target < 0
                    else "off"
                ),
                "best_acc": float(metrics["best_acc"]),
                "best_epoch": int(metrics["best_epoch"]),
                "final_acc": float(last["acc"]),
                "acc": float(last["acc"]),
                "e_rate_hz": float(last.get("rate_e") or 0.0),
            })

    print("  results:")
    for r in rows:
        print(
            f"    {r['model']:<5}  target={r['target_hz_signed']:+6.1f} Hz "
            f"({r['regulariser_mode']:<5})  "
            f"acc={r['acc']:5.2f}%  E={r['e_rate_hz']:6.2f} Hz"
        )

    # PING baseline (target = 0) sets the structural-bound annotation.
    ping_baseline_rate = next(
        r["e_rate_hz"] for r in rows
        if r["model"] == "ping" and r["target_spikes"] == 0
    )

    plot_pressure_sweep(
        rows, FIGURES / "pressure_sweep.png", notebook_run_id, ping_baseline_rate,
    )
    print(f"wrote {FIGURES / 'pressure_sweep.png'}")
    plot_training_curves(FIGURES / "training_curves.png", notebook_run_id)
    print(f"wrote {FIGURES / 'training_curves.png'}")

    duration_s = time.monotonic() - t_start
    train_cfg = load_config(cell_dir(MODELS[0], 0.0))
    crits = evaluate_success(rows, ping_baseline_rate, FIGURES)
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
            "targets_spikes_per_trial": list(TARGETS_SPIKES),
            "targets_hz_signed": [
                t * SPIKES_PER_TRIAL_TO_HZ for t in TARGETS_SPIKES
            ],
            "fr_strength": FR_STRENGTH,
            "max_samples": TIER_CONFIG[tier]["max_samples"],
            "epochs": TIER_CONFIG[tier]["epochs"],
            "t_ms": T_MS,
            "dt": DT_TRAIN,
            "seed": SEED,
        },
        "ping_baseline_rate_hz": ping_baseline_rate,
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
