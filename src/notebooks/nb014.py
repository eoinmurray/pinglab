"""Notebook runner for entry 013b — Eval-Δt transport across encoder modes.

Second part of the split nb013 trilogy. Loads nb013's trained
checkpoints, runs the eval-Δt sweep over each of the three input
encoder modes (upsample / downsample / resample), and emits the
accuracy-vs-eval-Δt and firing-rate-vs-eval-Δt figures.

Writes:
  * dt_sweep.png — accuracy vs eval-dt (the money plot)
  * firing_rates.png — mean hidden firing rate vs eval-dt
  * dt_sweep_<regime>_<model>.mp4 — per-sweep inference videos
  * numbers.json — per-mode (ref, min, max) accuracy & rates

Requires: nb013's trained checkpoints under
  src/artifacts/notebooks/nb013/<regime>/<model>/train/

Notebook entry: src/docs/src/pages/notebooks/nb014.mdx
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "notebooks"))

import _dt_stability_lib as nb013  # noqa: E402
from _modal import BatchDispatcher, parse_modal_gpu  # noqa: E402
from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from _tier import parse_tier  # noqa: E402

SLUG = "nb014"
# Sweep artifacts live under this entry's ARTIFACTS; checkpoints are read
# from nb013's path.
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
NB013_ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / "nb013"


def main() -> None:
    nb013.theme.apply()
    wipe_dir = "--no-wipe-dir" not in sys.argv
    modal_gpu = parse_modal_gpu(sys.argv)
    nb013.TIER = parse_tier(
        sys.argv, choices=nb013.TIER_CONFIG.keys(), default=nb013.DEFAULT_TIER,
    )
    # Point sweep_model() at this entry's ARTIFACTS so the per-eval-dt
    # results land under nb014/, while training checkpoints stay at
    # nb013/.
    nb013.ARTIFACTS = ARTIFACTS
    nb013.FIGURES = FIGURES

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {notebook_run_id} tier={nb013.TIER}")

    if wipe_dir:
        for d in (ARTIFACTS, FIGURES):
            if d.exists():
                print(f"[wipe] {d.relative_to(REPO)}")
                shutil.rmtree(d)
    FIGURES.mkdir(parents=True, exist_ok=True)
    persist_run_id(SLUG, notebook_run_id)

    # Reconstruct train_dirs from nb013's artifact tree.
    regime_train_dirs: dict[float, dict[str, Path]] = {}
    for dt_train in nb013.DT_TRAINS:
        td: dict[str, Path] = {}
        for m in nb013.MODELS:
            d = NB013_ARTIFACTS / nb013._regime_key(dt_train) / m / "train"
            if not (d / "weights.pth").exists():
                raise SystemExit(
                    f"missing nb013 checkpoint at {d.relative_to(REPO)}; "
                    "run nb013 first."
                )
            td[m] = d
        regime_train_dirs[dt_train] = td

    dispatcher = BatchDispatcher(modal_gpu, REPO, nb013.OSCILLOSCOPE)

    regime_sweep_dirs: dict[float, dict[str, dict[str, Path]]] = {}
    for dt_train in nb013.DT_TRAINS:
        td = regime_train_dirs[dt_train]
        sd: dict[str, dict[str, Path]] = {}
        for m in nb013.MODELS:
            sd[m] = {
                mode: nb013.sweep_model(m, dt_train, td[m], mode, dispatcher)
                for mode in nb013.ENCODER_MODES
            }
        regime_sweep_dirs[dt_train] = sd
    dispatcher.drain()

    for dt_train, sd in regime_sweep_dirs.items():
        for m, modes in sd.items():
            for mode, d in modes.items():
                if not (d / "results.json").exists():
                    raise SystemExit(
                        f"dt-sweep did not produce {d / 'results.json'}"
                    )

    nb013.plot_dt_sweep(
        regime_sweep_dirs, FIGURES / "dt_sweep.png", notebook_run_id,
    )
    print(f"wrote {(FIGURES / 'dt_sweep.png').relative_to(REPO)}")
    nb013.plot_firing_rates(
        regime_sweep_dirs, FIGURES / "firing_rates.png", notebook_run_id,
    )
    fr = FIGURES / "firing_rates.png"
    if fr.exists():
        print(f"wrote {fr.relative_to(REPO)}")

    # Copy inference videos under the new figures dir (skip train videos —
    # those belong to nb013).
    nb013.copy_videos({}, regime_sweep_dirs, FIGURES, notebook_run_id)

    numbers_path = FIGURES / "numbers.json"
    duration_s = time.monotonic() - t_start
    summary = nb013.write_numbers(
        regime_train_dirs,
        regime_sweep_dirs,
        out_path=numbers_path,
        notebook_run_id=notebook_run_id,
        duration_s=duration_s,
        init_match={},
    )
    numbers_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {numbers_path.relative_to(REPO)}  duration={summary['duration']}")


if __name__ == "__main__":
    main()
    sys.exit(0)
