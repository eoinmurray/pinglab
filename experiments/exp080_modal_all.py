"""Authorized parallel Modal dispatcher for all exp080 full-training cells."""

from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path

from exp080 import PROBE_CONDUCTANCES_US, SEEDS, verify_expanded_rate_training_protocol
from helpers import modal_backend
from helpers.modal_exp080_app import app, train_cell

REPO = Path(__file__).resolve().parents[1]
DESTINATION = REPO / "artifacts" / "data" / "exp080" / "step5"


def main() -> None:
    verify_expanded_rate_training_protocol()
    gpu = os.environ.get("PINGLAB_MODAL_GPU", "L40S")
    cells = [("full", probe, seed) for probe in PROBE_CONDUCTANCES_US for seed in SEEDS]
    print(f"LIVE exp080 parallel full training: {len(cells)} cells on {gpu}")
    modal = modal_backend._require_modal()
    output = getattr(modal, "enable_output", lambda: contextlib.nullcontext())()
    started = time.monotonic()
    results = []
    with output:
        with app.run():
            futures = [train_cell.spawn(*cell) for cell in cells]
            results = [future.get() for future in futures]

    failures: list[str] = []
    rate = modal_backend.GPU_USD_PER_SECOND.get(gpu)
    for result in results:
        payload = bytes(result.pop("artifact_tar_gz"))
        if modal_backend.sha256_bytes(payload) != result["artifact_sha256"]:
            raise RuntimeError("Modal artifact payload hash mismatch")
        modal_backend._extract_tree(payload, DESTINATION)
        cost = None if rate is None else float(result["elapsed_s"]) * rate
        result.update(
            {
                "gpu": gpu,
                "gpu_usd_per_second": rate,
                "estimated_cost_usd": cost,
                "parallel_dispatch_elapsed_s": time.monotonic() - started,
                "exact_provider_billing": False,
            }
        )
        ledger = (
            DESTINATION
            / "full"
            / f"probe-{result['probe_uS']:g}"
            / f"seed-{result['seed']}"
            / "modal.json"
        )
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text(json.dumps(result, indent=2) + "\n")
        if not result["success"]:
            failures.append(
                f"probe={result['probe_uS']:g}, seed={result['seed']}: {result['error']}"
            )
        print(
            f"collected probe={result['probe_uS']:g} seed={result['seed']} "
            f"elapsed={result['elapsed_s']:.1f}s estimated_cost=${cost:.4f}"
        )
    if failures:
        raise RuntimeError("\n".join(failures))
    print(f"all cells collected in {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
