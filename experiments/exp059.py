"""Notebook runner for entry 059 — what the SHD dataset looks like.

The first entry in the Spiking Heidelberg Digits collection: before training a
network on SHD, look at the raw data. SHD is event-based audio — each spoken
digit is a list of (spike time, cochlear channel) events over 700 channels, no
image, no dense array. Two figures, both straight from the raw HDF5 (no model,
no training):

  Figure 1 (class gallery): one utterance per class, 0-9 in German then English,
    each as a spike raster (time × channel). Shows the across-class structure —
    every word paints a different time-frequency signature.
  Figure 2 (within-class spread): four different utterances of one digit, side
    by side — the speaker-to-speaker variability the classifier must generalise
    over.

Writing: writings/exp059.typ · figures + numbers.json: artifacts/data/exp059/
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import theme  # noqa: E402
from helpers.cli import parse_meta  # noqa: E402
from helpers.datasets import SHD_LABELS, load_shd_events  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp059"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)

# Which split to sample from. The train split is the larger pool and these are
# only illustrative utterances, not a held-out measurement, so train is fine.
SPLIT = "train"
SHD_N_IN = 700  # cochlear channels

# Figure 1: one utterance per class, laid out 2 rows × 5. Which physical sample
# of each class — the k-th matching utterance — is fixed here so the gallery is
# reproducible run to run.
GALLERY_CLASSES = list(range(20))  # all 20 (German 0-9 then English 0-9)
GALLERY_SAMPLE_IDX = 0  # first matching utterance of each class

# Figure 2: within-class spread — this class, its first N utterances.
SPREAD_CLASS = 0  # "null"
SPREAD_N = 4

# Gallery layout: 20 panels laid out ncols wide (nrows falls out of the count).
GALLERY_NCOLS = 5

# Firing-rate regulariser constants (Cramer et al. upper-rate bound) that the
# NEXT entries use. Kept here so the writeup can read them off numbers.json
# rather than hand-typing them into prose.
RATE_TARGET_THETA_U = 100  # target upper bound on a unit's firing rate
RATE_WEIGHT_S_U = 0.06  # strength of the penalty applied above that bound

SCALE = {
    "dataset": "shd",
    "split": SPLIT,
    "n_channels": SHD_N_IN,
    "n_classes": len(SHD_LABELS),
    "gallery_panels": len(GALLERY_CLASSES),
    "spread_class": SHD_LABELS[SPREAD_CLASS],
    "spread_panels": SPREAD_N,
}


def _nth_index(labels: np.ndarray, cls: int, n: int) -> int | None:
    """Index of the n-th utterance with label `cls` (None if fewer than n+1)."""
    hits = np.flatnonzero(labels == cls)
    return int(hits[n]) if n < len(hits) else None


def _raster(ax, units: np.ndarray, times_s: np.ndarray, xlim_ms: float) -> None:
    """Draw one utterance as a spike raster: time (ms) on x, channel on y."""
    ax.scatter(
        times_s * 1000.0, units,
        s=1.0, marker="s", linewidths=0,
        color=theme.INK_BLACK, alpha=0.55, rasterized=True,
    )
    ax.set_xlim(0, xlim_ms)
    ax.set_ylim(0, SHD_N_IN)
    ax.tick_params(labelsize=theme.SIZE_TICK - 1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_gallery(events, labels, out_path: Path) -> dict:
    """One utterance per class as a raster, 2 rows × 5 (twice, for both languages
    stacked into a 4×5 grid). Returns per-panel event counts for numbers.json."""
    theme.apply()
    idxs = [(c, _nth_index(labels, c, GALLERY_SAMPLE_IDX)) for c in GALLERY_CLASSES]
    idxs = [(c, i) for c, i in idxs if i is not None]

    # Common time axis so panels are visually comparable.
    xlim_ms = max(
        float(events[i][1].max()) * 1000.0 for _, i in idxs
    ) * 1.02

    ncols = GALLERY_NCOLS
    nrows = int(np.ceil(len(idxs) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.5, 3.66), sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()
    counts = {}
    for ax, (cls, i) in zip(axes, idxs):
        units, times_s = events[i]
        _raster(ax, units, times_s, xlim_ms)
        ax.set_title(
            f"{cls}  {SHD_LABELS[cls]}",
            fontsize=theme.SIZE_LABEL - 1, pad=2,
        )
        counts[SHD_LABELS[cls]] = int(len(units))
    for ax in axes[len(idxs):]:
        ax.set_visible(False)

    # One shared pair of axis labels rather than 20.
    fig.supxlabel("time (ms)", fontsize=theme.SIZE_LABEL)
    fig.supylabel("cochlear channel", fontsize=theme.SIZE_LABEL)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return counts


def plot_within_class(events, labels, out_path: Path) -> dict:
    """SPREAD_N different utterances of one digit, side by side — the raw
    speaker variability. Returns the sample indices + event counts used."""
    theme.apply()
    idxs = [
        _nth_index(labels, SPREAD_CLASS, n) for n in range(SPREAD_N)
    ]
    idxs = [i for i in idxs if i is not None]
    xlim_ms = max(float(events[i][1].max()) * 1000.0 for i in idxs) * 1.02

    fig, axes = plt.subplots(
        1, len(idxs), figsize=(6.5, 3.66), sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()
    used = {}
    for k, (ax, i) in enumerate(zip(axes, idxs)):
        units, times_s = events[i]
        _raster(ax, units, times_s, xlim_ms)
        ax.set_title(f"utterance {k + 1}", fontsize=theme.SIZE_LABEL - 1, pad=2)
        if k == 0:
            ax.set_ylabel("cochlear channel", fontsize=theme.SIZE_LABEL)
        used[f"sample_{i}"] = int(len(units))

    fig.supxlabel("time (ms)", fontsize=theme.SIZE_LABEL)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return used


def main() -> None:
    meta = parse_meta(sys.argv)

    t_start = time.monotonic()
    run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {run_id}")

    with published_run(
        SLUG, run_id, make_artifacts=False, scale=SCALE, plot_only=meta.plot_only,
    ) as (_artifacts, figures):
        print(f"[data] loading SHD {SPLIT} split (events, not binned)")
        events, labels = load_shd_events(split=SPLIT)
        print(f"  {len(labels)} utterances · {len(set(labels.tolist()))} classes")

        gallery_dst = figures / "class_gallery.png"
        gallery_counts = plot_gallery(events, labels, gallery_dst)
        print(f"wrote {gallery_dst}")

        spread_dst = figures / "within_class_spread.png"
        spread_counts = plot_within_class(events, labels, spread_dst)
        print(f"wrote {spread_dst}")

        # Dataset-level summary numbers the writing can read off.
        ev_per_utt = np.array([len(events[i][0]) for i in range(len(events))])
        # Per-utterance duration = time of the last spike (events carry times in s).
        dur_per_utt = np.array(
            [float(events[i][1].max()) for i in range(len(events))]
        )
        n_classes = int(len(set(labels.tolist())))
        digits_per_lang = n_classes // 2  # 10 spoken digits per language

        payload = {
            "split": SPLIT,
            "n_utterances": int(len(labels)),
            "n_classes": n_classes,
            "n_channels": SHD_N_IN,
            "events_per_utterance_median": float(np.median(ev_per_utt)),
            "events_per_utterance_min": int(ev_per_utt.min()),
            "events_per_utterance_max": int(ev_per_utt.max()),
            "duration_min_s": float(dur_per_utt.min()),
            "duration_median_s": float(np.median(dur_per_utt)),
            "duration_max_s": float(dur_per_utt.max()),
            "gallery_event_counts": gallery_counts,
            "within_class_event_counts": spread_counts,
            # Recipe constants + descriptive ranges the prose/captions cite, so no
            # number is hand-typed into the writeup.
            "config": {
                "digit_min": 0,
                "digit_max": digits_per_lang - 1,  # 9
                "german_label_min": 0,
                "german_label_max": digits_per_lang - 1,  # 9
                "english_label_min": digits_per_lang,  # 10
                "english_label_max": n_classes - 1,  # 19
                "channel_min": 0,  # cochlear channels are 0-indexed
                "spread_class": SPREAD_CLASS,  # 0 ("null")
                "spread_panels": SPREAD_N,  # 4
                "gallery_ncols": GALLERY_NCOLS,  # 5
                "gallery_nrows": int(np.ceil(len(GALLERY_CLASSES) / GALLERY_NCOLS)),
                "rate_target_theta_u": RATE_TARGET_THETA_U,  # 100
                "rate_weight_s_u": RATE_WEIGHT_S_U,  # 0.06
            },
        }

        duration_s = time.monotonic() - t_start
        write_numbers(figures, run_id=run_id, duration_s=duration_s, payload=payload)
        print(f"wrote {figures / 'numbers.json'}")
        print(f"  duration: {duration_s:.1f}s")


if __name__ == "__main__":
    main()
