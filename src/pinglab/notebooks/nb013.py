"""Notebook runner for entry 013a — Δt training equivalence.

First part of the split nb013 trilogy. Trains every model
(standard-snn, snntorch-library, cuba, coba, ping) at two training-Δt
regimes (0.1 ms and 1.0 ms), verifies the CUBANet-family models start
from bit-identical weights, and emits the training_curves figure.

Writes:
  * training_curves.png — train loss & test accuracy per epoch
  * training_<regime>_<model>.mp4 — per-epoch training videos
  * numbers.json — config + per-regime/per-model best/final + init match

Downstream: nb013b reads these checkpoints to run the eval-Δt sweep;
nb013c reads them for the latency analysis.

Notebook entry: src/docs/src/pages/notebooks/nb013.mdx
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))
sys.path.insert(0, str(REPO / "src" / "pinglab" / "notebooks"))

import _nb013_lib as nb013  # noqa: E402
from _modal import BatchDispatcher, parse_modal_gpu  # noqa: E402
from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from _tier import parse_tier  # noqa: E402

SLUG = "nb013"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG


def main() -> None:
    nb013.theme.apply()
    skip_training = "--skip-training" in sys.argv
    wipe_dir = "--no-wipe-dir" not in sys.argv
    modal_gpu = parse_modal_gpu(sys.argv)
    nb013.TIER = parse_tier(
        sys.argv, choices=nb013.TIER_CONFIG.keys(), default=nb013.DEFAULT_TIER,
    )
    # Redirect nb013's path constants at this entry's ARTIFACTS so the
    # imported helpers write here instead of the umbrella nb013 dir.
    nb013.ARTIFACTS = ARTIFACTS
    nb013.FIGURES = FIGURES

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    print(
        f"notebook_run_id = {notebook_run_id} tier={nb013.TIER}"
        + ("  [skip-training]" if skip_training else "")
    )

    if wipe_dir:
        if skip_training and ARTIFACTS.exists():
            # Preserve train dirs only; nothing else lives under nb013.
            pass
        else:
            for d in (ARTIFACTS, FIGURES):
                if d.exists():
                    print(f"[wipe] {d.relative_to(REPO)}")
                    shutil.rmtree(d)
    FIGURES.mkdir(parents=True, exist_ok=True)
    persist_run_id(SLUG, notebook_run_id)

    init_match = nb013.verify_init_match(nb013.MODELS, nb013.SEED)
    dispatcher = BatchDispatcher(modal_gpu, REPO, nb013.OSCILLOSCOPE)

    regime_train_dirs: dict[float, dict[str, Path]] = {}
    for dt_train in nb013.DT_TRAINS:
        print(f"\n=== regime: train dt = {dt_train} ms ===")
        if skip_training:
            td = {
                m: ARTIFACTS / nb013._regime_key(dt_train) / m / "train"
                for m in nb013.MODELS
            }
            for m, d in td.items():
                if not (d / "weights.pth").exists():
                    raise SystemExit(
                        f"--skip-training requires existing train weights at {d}"
                    )
        else:
            td = {m: nb013.train_model(m, dt_train, dispatcher) for m in nb013.MODELS}
        regime_train_dirs[dt_train] = td
    dispatcher.drain()

    for dt_train, td in regime_train_dirs.items():
        for m, d in td.items():
            if not (d / "metrics.json").exists():
                raise SystemExit(f"training did not produce {d / 'metrics.json'}")
            if not nb013.training_video_path(d).exists():
                raise SystemExit(
                    f"training did not produce {nb013.training_video_path(d)}"
                )

    nb013.plot_training_curves(
        regime_train_dirs, FIGURES / "training_curves.png", notebook_run_id,
    )
    print(f"wrote {(FIGURES / 'training_curves.png').relative_to(REPO)}")

    # Copy training videos under the new figures dir.
    nb013.copy_videos(regime_train_dirs, {}, FIGURES, notebook_run_id)

    numbers_path = FIGURES / "numbers.json"
    duration_s = time.monotonic() - t_start
    # Use nb013.write_numbers with empty sweep dirs — the umbrella's
    # numbers.json schema gracefully handles missing sweep entries.
    summary = nb013.write_numbers(
        regime_train_dirs,
        regime_sweep_dirs={dt: {m: {} for m in nb013.MODELS} for dt in regime_train_dirs},
        out_path=numbers_path,
        notebook_run_id=notebook_run_id,
        duration_s=duration_s,
        init_match=init_match,
    )
    numbers_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {numbers_path.relative_to(REPO)}  duration={summary['duration']}")


if __name__ == "__main__":
    main()
    sys.exit(0)
