"""Notebook runner for entry 023 — PING fundamentals.

Runs the oscilloscope in *image* mode twice — once with the recurrent
loop disabled (*--ei-strength 0*, "coba") and once with it active
(*--ei-strength 1.5*, "ping") — and renders dedicated per-cell raster
plots from the saved spike arrays. Everything else is held fixed
(MNIST digit 0 sample 0, --input-rate 50, --t-ms 400, --w-in 1.5 0.3).

Two PNGs are written: raster__coba.png and raster__ping.png.

Notebook entry: src/docs/src/pages/notebooks/nb023.mdx
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _modal import parse_modal_gpu  # noqa: E402
from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from _tier import parse_tier  # noqa: E402

SLUG = "nb023"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
OSCILLOSCOPE = REPO / "src" / "pinglab" / "oscilloscope/__main__.py"
SCOPE_OUT_PNG = REPO / "src" / "artifacts" / "oscilloscope" / "snapshot.png"
SCOPE_OUT_NPZ = REPO / "src" / "artifacts" / "oscilloscope" / "snapshot.npz"

COMMON_ARGS = [
    "image",
    "--model", "ping",
    "--input", "dataset",
    "--dataset", "mnist",
    "--digit", "0",
    "--sample", "0",
    "--w-in", "1.5", "0.3",
    "--input-rate", "50",
    "--t-ms", "400",
]
CELLS: dict[str, dict] = {
    "coba": {
        "args": ["--ei-strength", "0"],
        "title": "COBA — no recurrent loop (ei-strength = 0)",
    },
    "ping": {
        "args": ["--ei-strength", "1.5"],
        "title": "PING — recurrent loop active (ei-strength = 1.5)",
    },
}

TIER_CONFIG = {
    "extra small": {},
    "small": {},
    "medium": {},
    "large": {},
    "extra large": {},
}
DEFAULT_TIER = "small"


def plot_raster(npz_path: Path, out_path: Path, title: str) -> None:
    data = np.load(npz_path)
    spk_e = data["spk_e"]
    spk_i = data["spk_i"]
    dt = float(data["dt"])
    T = spk_e.shape[0]
    t_ms = np.arange(T) * dt

    has_i = spk_i.size > 0 and spk_i.shape[0] == T and spk_i.any()

    if has_i:
        fig, (ax_e, ax_i) = plt.subplots(
            2, 1, figsize=(8, 4.5), sharex=True,
            gridspec_kw={"height_ratios": [4, 1]},
        )
    else:
        fig, ax_e = plt.subplots(1, 1, figsize=(8, 4.5))
        ax_i = None

    e_idx, e_t = np.where(spk_e.T)
    ax_e.scatter(t_ms[e_t], e_idx, s=1.0, c="black", marker="|", linewidths=0.5)
    ax_e.set_ylabel("E neuron")
    ax_e.set_ylim(0, spk_e.shape[1])
    ax_e.set_xlim(0, T * dt)
    ax_e.set_title(title)

    if has_i:
        i_idx, i_t = np.where(spk_i.T)
        ax_i.scatter(t_ms[i_t], i_idx, s=1.0, c="C3", marker="|", linewidths=0.5)
        ax_i.set_ylabel("I neuron")
        ax_i.set_ylim(0, spk_i.shape[1])
        ax_i.set_xlim(0, T * dt)
        ax_i.set_xlabel("time (ms)")
    else:
        ax_e.set_xlabel("time (ms)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    modal_gpu = parse_modal_gpu(sys.argv)
    wipe_dir = "--no-wipe-dir" not in sys.argv

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {notebook_run_id} tier={tier}")

    if wipe_dir:
        for d in (ARTIFACTS, FIGURES):
            if d.exists():
                print(f"[wipe] {d.relative_to(REPO)}")
                shutil.rmtree(d)
    FIGURES.mkdir(parents=True, exist_ok=True)
    persist_run_id(SLUG, notebook_run_id)

    rasters: dict[str, Path] = {}
    for cell, spec in CELLS.items():
        for p in (SCOPE_OUT_PNG, SCOPE_OUT_NPZ):
            if p.exists():
                p.unlink()
        scope_argv = [*COMMON_ARGS, *spec["args"]]
        cmd = ["uv", "run", "python", str(OSCILLOSCOPE), *scope_argv]
        print(f"[scope] {cell}: {' '.join(scope_argv)}")
        subprocess.run(cmd, cwd=REPO, check=True)
        if not SCOPE_OUT_NPZ.exists():
            raise SystemExit(f"oscilloscope did not produce {SCOPE_OUT_NPZ}")
        dst = FIGURES / f"raster__{cell}.png"
        plot_raster(SCOPE_OUT_NPZ, dst, spec["title"])
        rasters[cell] = dst
        print(f"wrote {dst}")

    duration_s = time.monotonic() - t_start
    figs_root = FIGURES.parents[2]
    summary = {
        "notebook_run_id": notebook_run_id,
        "duration_s": round(duration_s, 1),
        "tier": tier,
        "config": {
            "tier": tier,
            "model": "ping",
            "input": "mnist d0 s0",
            "cells": list(CELLS),
            "t_ms": 400,
            "input_rate_hz": 50,
            "modal_gpu": modal_gpu,
        },
        "results": [],
        "success_criteria": [
            {
                "label": f"{cell} raster rendered",
                "passed": dst.exists() and dst.stat().st_size > 0,
                "detail": f"{dst.name} ({dst.stat().st_size} bytes)" if dst.exists() else "missing",
                "detail_href": "/" + str(dst.relative_to(figs_root)) if dst.exists() else None,
            }
            for cell, dst in rasters.items()
        ],
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {duration_s:.1f}s")


if __name__ == "__main__":
    main()
