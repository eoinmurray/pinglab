"""RunPod-only dispatcher for the exp022 matched_mid_v1 generation.

This is deliberately separate from the published/calibrated exp022 runner.
Dry-run is the default; only ``--live`` creates paid pods.  The intended order:

  1. ``--adopt-stage1`` locally (no training; validates and copies seed 42)
  2. ``--batch canary`` for the four seed-43/44 theta-off replication cells
  3. inspect the three-seed gate, then ``--batch remaining``
  4. ``--collect`` and switch downstream consumers only after the audit

The pod-side path atomically adopts the two already-trained Stage 1 cells from
the shared RunPod volume before doing work, so the full generation never pays
to retrain them.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp022  # noqa: E402
from helpers import runpod  # noqa: E402

RUNNER = "exp022_matched"
STAGE1_NAMES = {"coba__off__seed42", "ping__off__seed42"}
CANARY_NAMES = {cell["name"] for cell in exp022.matched_canary_cells()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--batch", choices=("canary", "remaining", "full"), default="canary",
        help="canary=4 replication cells; remaining=everything after the gate; full=all 87",
    )
    p.add_argument("--live", action="store_true", help="create paid pods (default: dry-run)")
    p.add_argument("--collect", action="store_true", help="collect matched_mid_v1 only")
    p.add_argument("--manifest", action="store_true", help="write/print local generation status")
    p.add_argument("--plumbing", action="store_true", help="tiny remote wiring scale")
    p.add_argument("--pod-run", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--adopt-stage1", action="store_true", help="validate/copy Stage 1 seed 42")
    p.add_argument("--gpu", choices=runpod.GPU_CHOICES, default="5090")
    p.add_argument("--cells-per-pod", type=int, default=9)
    p.add_argument("--only-cells", nargs="*", default=[])
    return p.parse_args(argv)


def selected_cells(args: argparse.Namespace) -> list[dict]:
    if args.batch == "canary":
        cells = exp022.matched_canary_cells()
    elif args.batch == "remaining":
        cells = [
            cell for cell in exp022.MATCHED_CELLS
            if cell["name"] not in STAGE1_NAMES | CANARY_NAMES
        ]
    else:
        cells = list(exp022.MATCHED_CELLS)
    if args.only_cells:
        wanted = set(args.only_cells)
        known = {cell["name"] for cell in exp022.MATCHED_CELLS}
        unknown = wanted - known
        if unknown:
            raise SystemExit(f"unknown matched cell(s): {sorted(unknown)}")
        cells = [cell for cell in cells if cell["name"] in wanted]
    return cells


def buckets(cells: list[dict], *, batch: str, cells_per_pod: int) -> list[dict]:
    # Canary cells each get a GPU: it is a decision gate, so latency matters.
    if batch == "canary":
        return [{"name": f"gate-{c['name']}", "cells": [c["name"]]} for c in cells]
    canonical = [c["name"] for c in cells if c["family"] == "canonical"]
    sweep = [c["name"] for c in cells if c["family"] != "canonical"]
    result = [{"name": f"canon-{name}", "cells": [name]} for name in canonical]
    result += runpod.chunk_buckets(sweep, cells_per_pod, prefix="matched")
    return result


def _cell(name: str) -> dict | None:
    return next((c for c in exp022.MATCHED_CELLS if c["name"] == name), None)


def _train(cell: dict, *, plumbing: bool) -> None:
    if plumbing:
        os.environ["PINGLAB_NB022_PLUMBING"] = "1"
        spec = {k: v for k, v in cell.items() if k != "max_samples"}
    else:
        spec = cell
    samples, epochs = exp022.cell_samples_epochs(spec)
    out = exp022.matched_cell_dir(cell["name"])
    args = exp022.build_matched_train_args(spec, out, samples, epochs)
    print(f"[matched train] {cell['name']} (n={samples}, {epochs} ep) → {out}")
    subprocess.run([sys.executable, str(exp022.SNN_TOOL), *args], cwd=REPO, check=True)


def pod_run() -> None:
    plumbing = os.environ.get("PINGLAB_NB022_PLUMBING") == "1"
    # Stage 1 lives on the same network volume under matched_mid_stage1.
    # Atomic adoption is safe even when all four canary pods arrive together.
    exp022.adopt_matched_stage1()

    def is_done(name: str) -> bool:
        cell = _cell(name)
        return cell is not None and exp022.matched_cell_is_done(
            cell, plumbing=plumbing,
        )

    def run_job(name: str) -> None:
        cell = _cell(name)
        assert cell is not None  # pod_run_loop only passes registered job ids
        _train(cell, plumbing=plumbing)

    runpod.pod_run_loop(
        job_ids=[c["name"] for c in exp022.MATCHED_CELLS],
        is_done=is_done,
        run_job=run_job,
        label="matched-pod",
    )


def dispatch(args: argparse.Namespace) -> None:
    cells = selected_cells(args)
    fleet = buckets(cells, batch=args.batch, cells_per_pod=args.cells_per_pod)
    runpod.dispatch(
        slug=exp022.SLUG,
        runner=RUNNER,
        buckets=fleet,
        gpu=args.gpu,
        live=args.live,
        plumbing=args.plumbing,
        collect=args.collect,
        collect_subdir=f"{runpod.TRAINING_SUBDIR}/{exp022.MATCHED_GENERATION}",
        local_collect_dir=str(exp022.MATCHED_TRAINING_ROOT),
        plumbing_env={"PINGLAB_NB022_PLUMBING": "1"},
    )


def write_manifest() -> dict:
    """Write the matched generation's complete declared/observed contract."""
    rows = []
    for cell in exp022.MATCHED_CELLS:
        valid, errors = exp022.matched_cell_validation(cell)
        directory = exp022.matched_cell_dir(cell["name"])
        config = exp022.load_config(directory) if valid else {}
        metrics = exp022.load_metrics(directory) if valid else {}
        rate_e, rate_i = exp022.final_rates(directory) if valid else (None, None)
        rows.append({
            "name": cell["name"],
            "family": cell["family"],
            "model": cell["model"],
            "seed": cell["seed"],
            "theta_u": cell.get("theta_u"),
            "expected_config": exp022.matched_expected_config(cell),
            "valid": valid,
            "validation_errors": errors,
            "git_sha": config.get("git_sha"),
            "best_acc": metrics.get("best_acc"),
            "best_epoch": metrics.get("best_epoch"),
            "rate_e": rate_e,
            "rate_i": rate_i,
        })
    manifest = {
        "generation": exp022.MATCHED_GENERATION,
        "recipe_family": "matched_mid",
        "training_root": str(exp022.MATCHED_TRAINING_ROOT),
        "declared_cells": len(rows),
        "valid_cells": sum(row["valid"] for row in rows),
        "families": {
            family: sum(row["family"] == family for row in rows)
            for family in ("canonical", "theta_u", "tau_gaba", "dt", "init")
        },
        "cells": rows,
    }
    exp022.MATCHED_TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
    path = exp022.MATCHED_TRAINING_ROOT / "_generation.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"matched generation: {manifest['valid_cells']}/{manifest['declared_cells']} valid")
    print(f"manifest: {path}")
    return manifest


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.pod_run:
        pod_run()
        return
    if args.adopt_stage1:
        adopted = exp022.adopt_matched_stage1()
        print(f"Stage 1 ready: {len(adopted)} adopted, {2 - len(adopted)} already valid")
        if not (args.live or args.collect):
            if args.manifest:
                write_manifest()
            return
    if args.manifest and not (args.live or args.collect):
        write_manifest()
        return
    dispatch(args)
    if args.collect:
        write_manifest()


if __name__ == "__main__":
    main()
