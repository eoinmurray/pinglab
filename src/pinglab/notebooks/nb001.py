"""Notebook runner for entry 001 — plotting style guide.

Generates a fixed set of placeholder plots covering the chart types
used across pinglab notebooks (line, bar, scatter, histogram with
theoretical overlay, heatmap, spike raster). The data is intentionally
synthetic — this notebook is a *style* surface, not a result. Iterate
on theme.py and the per-plot styling here; whatever lands as the
final brand-aligned look is the canonical pinglab plot style.

Notebook entry: src/docs/src/pages/notebooks/nb001.mdx
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from _tier import parse_tier  # noqa: E402

SLUG = "nb001"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
DEFAULT_TIER = "small"
TIER_CONFIG = {
    "small": dict(),
    "medium": dict(),
    "large": dict(),
}
FIGSIZE = (8, 4.5)
SEED = 42


def _setup() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import theme  # type: ignore[import]

    theme.apply()


def fig_line() -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    epochs = np.arange(1, 41)
    train = 1 - 0.85 * np.exp(-epochs / 8) + rng.normal(0, 0.01, size=epochs.shape)
    test = 1 - 0.80 * np.exp(-epochs / 8) + rng.normal(0, 0.015, size=epochs.shape)
    ax.plot(epochs, train * 100, label="train")
    ax.plot(epochs, test * 100, label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Line: training curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = FIGURES / "line.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_line_many() -> Path:
    """12 lines, ordinal — too many for the 5-color cycle, so colors come
    from the brand colormap instead. Brightness encodes the parameter."""
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    epochs = np.arange(1, 41)
    cmap = plt.get_cmap("pinglab_brand")
    n_series = 12
    lrs = np.geomspace(1e-4, 1e-1, n_series)
    for i, lr in enumerate(lrs):
        plateau = 0.92 - 0.35 * abs(np.log10(lr) + 2.5) / 3
        plateau = np.clip(plateau, 0.45, 0.92)
        tau = 6 + 8 * abs(np.log10(lr) + 2.5)
        curve = plateau * (1 - np.exp(-epochs / tau)) + rng.normal(0, 0.008, size=epochs.shape)
        # Skip the colormap's extreme-light end so series stay legible on white.
        color = cmap(0.15 + 0.8 * i / (n_series - 1))
        ax.plot(epochs, curve * 100, color=color, linewidth=1.4)

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.matplotlib.colors.LogNorm(vmin=lrs[0], vmax=lrs[-1]),
    )
    fig.colorbar(sm, ax=ax, label="learning rate")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Line: many series (sweep)")
    out = FIGURES / "line_many.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_bar() -> Path:
    import matplotlib.pyplot as plt
    import numpy as np
    import theme  # type: ignore[import]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    models = ["cuba", "coba", "ping"]
    windows = np.arange(4)
    width = 0.25
    base = np.array([
        [25.4, 26.4, 28.5, 27.0],
        [23.8, 27.1, 27.9, 28.4],
        [24.0, 29.5, 29.2, 30.1],
    ])
    for i, m in enumerate(models):
        ax.bar(windows + (i - 1) * width, base[i], width, label=m)
    ax.axhline(10, linestyle="--", linewidth=1.0, color=theme.DEEP_RED, label="chance")
    ax.set_xticks(windows)
    ax.set_xticklabels([f"w{k}" for k in windows])
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Bar: grouped categorical")
    ax.legend()
    out = FIGURES / "bar.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_scatter() -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    import theme  # type: ignore[import]

    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    n = 80
    x = rng.uniform(0, 100, size=n)
    y = 0.6 * x + rng.normal(0, 8, size=n)
    sizes = rng.uniform(20, 120, size=n)
    ax.scatter(x, y, s=sizes, alpha=0.55, color=theme.CAT_BLUE, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("firing rate (Hz)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Scatter: two continuous variables")
    ax.grid(True, alpha=0.3)
    out = FIGURES / "scatter.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_hist() -> Path:
    import matplotlib.pyplot as plt
    import numpy as np
    import theme  # type: ignore[import]

    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    T = 80.0
    n = 4000
    speeds = np.sqrt(-2 * T * np.log(rng.uniform(1e-9, 1, size=n)))
    bins = np.linspace(0, 30, 30)
    ax.hist(speeds, bins=bins, alpha=0.5, label="samples")
    v = np.linspace(0, 30, 200)
    pdf = (v / T) * np.exp(-v * v / (2 * T))
    pdf_scaled = pdf * n * (bins[1] - bins[0])
    ax.plot(v, pdf_scaled, color=theme.DEEP_RED, linewidth=2.0, label="theory (Maxwell-Boltzmann)")
    ax.set_xlabel("speed")
    ax.set_ylabel("count")
    ax.set_title("Histogram: data + theoretical overlay")
    ax.legend()
    out = FIGURES / "histogram.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_heatmap() -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    train_dts = np.array([0.1, 0.25, 0.5, 1.0])
    eval_dts = np.array([0.05, 0.1, 0.25, 0.5, 1.0, 1.5])
    base = 88 - 2 * np.abs(np.log10(eval_dts[None, :] / train_dts[:, None]))
    noise = rng.normal(0, 1.5, size=base.shape)
    grid = np.clip(base + noise, 60, 95)
    im = ax.imshow(grid, aspect="auto", origin="upper")
    ax.set_xticks(range(len(eval_dts)))
    ax.set_xticklabels([f"{d:g}" for d in eval_dts])
    ax.set_yticks(range(len(train_dts)))
    ax.set_yticklabels([f"{d:g}" for d in train_dts])
    ax.set_xlabel("eval dt (ms)")
    ax.set_ylabel("train dt (ms)")
    ax.set_title("Heatmap: 2D parameter sweep")
    fig.colorbar(im, ax=ax, label="accuracy (%)")
    out = FIGURES / "heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_raster() -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    import theme  # type: ignore[import]

    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    n_neurons = 200
    T = 1.0
    dt = 0.001
    times = np.arange(0, T, dt)
    rate_hz = 6 + 4 * np.sin(2 * np.pi * 4 * times)
    spikes = rng.uniform(size=(len(times), n_neurons)) < rate_hz[:, None] * dt
    t_idx, n_idx = np.where(spikes)
    ax.scatter(times[t_idx], n_idx, s=1.2, c=theme.INK, alpha=0.7)
    ax.set_xlim(0, T)
    ax.set_ylim(-1, n_neurons)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("neuron")
    ax.set_title("Raster: spike trains")
    out = FIGURES / "raster.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    notebook_run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {notebook_run_id}")

    if "--no-wipe-dir" not in sys.argv:
        for path in (ARTIFACTS, FIGURES):
            if path.exists():
                print(f"[wipe] {path.relative_to(REPO)}")
                shutil.rmtree(path)
    FIGURES.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    _setup()
    t_start = time.monotonic()
    for fn in (fig_line, fig_line_many, fig_bar, fig_scatter, fig_hist, fig_heatmap, fig_raster):
        out = fn()
        print(f"  → {out.relative_to(REPO)}")
    duration_s = time.monotonic() - t_start

    persist_run_id(SLUG, notebook_run_id)
    summary = {
        "notebook_run_id": notebook_run_id,
        "duration_s": duration_s,
        "duration": f"{int(duration_s // 60)}m {int(duration_s % 60):02d}s",
        "config": {"figsize": list(FIGSIZE), "seed": SEED},
        "run_finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {(FIGURES / 'numbers.json').relative_to(REPO)}")


if __name__ == "__main__":
    main()
