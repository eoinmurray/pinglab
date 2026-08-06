"""Authorized synchronous Modal dispatcher for exp077 Step 5 cells."""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

from helpers import modal_backend
from helpers.modal_exp077_app import app, train_cell

REPO = Path(__file__).resolve().parents[1]
DESTINATION = REPO / "artifacts" / "data" / "exp077"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "full"), required=True)
    parser.add_argument("--probe", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--live", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpu = __import__("os").environ.get("PINGLAB_MODAL_GPU", "L40S")
    print(
        f"{'LIVE' if args.live else 'DRY-RUN'} exp077 stage={args.stage} "
        f"probe={args.probe:g} seed={args.seed} gpu={gpu}"
    )
    if not args.live:
        return
    if args.stage == "full":
        from exp077 import verify_expanded_rate_training_protocol

        verify_expanded_rate_training_protocol()
    modal = modal_backend._require_modal()
    output = getattr(modal, "enable_output", lambda: contextlib.nullcontext())()
    started = time.monotonic()
    with output:
        with app.run():
            result = train_cell.remote(args.stage, args.probe, args.seed)
    payload = bytes(result.pop("artifact_tar_gz"))
    if modal_backend.sha256_bytes(payload) != result["artifact_sha256"]:
        raise RuntimeError("Modal artifact payload hash mismatch")
    modal_backend._extract_tree(payload, DESTINATION / "step5")
    rate = modal_backend.GPU_USD_PER_SECOND.get(gpu)
    cost = None if rate is None else float(result["elapsed_s"]) * rate
    result.update(
        {
            "gpu": gpu,
            "gpu_usd_per_second": rate,
            "estimated_cost_usd": cost,
            "dispatch_elapsed_s": time.monotonic() - started,
            "exact_provider_billing": False,
        }
    )
    ledger = DESTINATION / "step5" / args.stage / f"probe-{args.probe:g}" / f"seed-{args.seed}" / "modal.json"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps(result, indent=2) + "\n")
    if not result["success"]:
        raise RuntimeError(result["error"])
    print(f"collected {ledger.parent}; estimated GPU cost ${cost:.4f}")


if __name__ == "__main__":
    main()
