"""Authorized Modal dispatcher for exp080's frozen held-out evaluation."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

from helpers import modal_backend
from helpers.modal_exp080_eval_app import app, evaluate

REPO = Path(__file__).resolve().parents[1]
DESTINATION = REPO / "artifacts" / "data" / "exp080" / "step6"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    gpu = os.environ.get("PINGLAB_MODAL_GPU", "L40S")
    print(f"{'LIVE' if args.live else 'DRY-RUN'} exp080 frozen evaluation gpu={gpu}")
    if not args.live:
        return
    modal = modal_backend._require_modal()
    output = getattr(modal, "enable_output", lambda: contextlib.nullcontext())()
    started = time.monotonic()
    with output:
        with app.run():
            result = evaluate.remote()
    payload = bytes(result.pop("artifact_tar_gz"))
    if modal_backend.sha256_bytes(payload) != result["artifact_sha256"]:
        raise RuntimeError("Modal evaluation artifact payload hash mismatch")
    modal_backend._extract_tree(payload, DESTINATION)
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
    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION / "modal.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["success"]:
        raise RuntimeError(result["error"])
    print(f"collected held-out evaluation; estimated GPU cost ${cost:.4f}")


if __name__ == "__main__":
    main()
