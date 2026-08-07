#!/usr/bin/env python
"""Disposable RunPod capacity smoke test.

Checks that a requested GPU can be allocated in pinglab's RunPod datacenter,
become reachable over SSH, import the project's CUDA-enabled PyTorch build, and
execute real FP32 work.  It never writes experiment artifacts and always
destroys every pod it creates.

Dry-run (free, the default):
    uv run python experiments/helpers/runpod_smoke.py

Live checks (spends money):
    uv run python experiments/helpers/runpod_smoke.py --live
    uv run python experiments/helpers/runpod_smoke.py --live --gpu 4090
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any

# Make ``helpers`` importable when this file is executed from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import runpod  # noqa: E402, I001


DEFAULT_GPUS = tuple(runpod.GPU_CHOICES)
DEFAULT_TIMEOUT_S = 600
DEFAULT_MAX_RUNTIME_S = 900

BENCHMARK = r"""import json
import time
import torch

torch.backends.cuda.matmul.allow_tf32 = False
device = torch.device("cuda")
n = 8192
iterations = 20
a = torch.randn((n, n), device=device, dtype=torch.float32)
b = torch.randn((n, n), device=device, dtype=torch.float32)
for _ in range(3):
    torch.mm(a, b)
torch.cuda.synchronize()
started = time.perf_counter()
for _ in range(iterations):
    result = torch.mm(a, b)
torch.cuda.synchronize()
elapsed = time.perf_counter() - started
props = torch.cuda.get_device_properties(0)
print(json.dumps({
    "name": props.name,
    "vram_gib": round(props.total_memory / 2**30, 2),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "seconds": elapsed,
    "tflops": 2 * n**3 * iterations / elapsed / 1e12,
    "checksum": float(result[0, 0]),
}))
"""


def _pod_prices() -> dict[str, float]:
    """Return actual hourly prices for currently running smoke pods."""
    prices: dict[str, float] = {}
    for pod in runpod.running_pods():
        name = str(pod.get("name", ""))
        if name.startswith("pinglab-smoke-") and pod.get("costPerHr") is not None:
            prices[str(pod["id"])] = float(pod["costPerHr"])
    return prices


def _exercise(pod_id: str, gpu: str, created: float, timeout_s: int) -> dict[str, Any]:
    host, port = runpod.wait_for_ssh(pod_id, timeout=timeout_s)
    ready_s = time.monotonic() - created
    command = (
        f"{runpod.UV} run --project {shlex.quote(runpod.POD_REPO)} "
        f"python -c {shlex.quote(BENCHMARK)}"
    )
    output = runpod.run_on_pod(host, port, command, timeout=timeout_s)
    payload = json.loads(output.strip().splitlines()[-1])
    return {"gpu": gpu, "pod_id": pod_id, "ready_s": ready_s, **payload}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu",
        choices=(*runpod.GPU_CHOICES, "both"),
        default="both",
        help="GPU to check (default: both)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="actually create paid pods; without this flag, print the plan only",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_S,
        help=f"seconds to wait for SSH/benchmark per pod (default: {DEFAULT_TIMEOUT_S})",
    )
    parser.add_argument(
        "--max-runtime",
        type=int,
        default=DEFAULT_MAX_RUNTIME_S,
        help=(
            "pod-side emergency teardown timer in seconds "
            f"(default: {DEFAULT_MAX_RUNTIME_S})"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.timeout <= 0 or args.max_runtime <= 0:
        raise SystemExit("--timeout and --max-runtime must be positive")

    gpus = list(DEFAULT_GPUS if args.gpu == "both" else (args.gpu,))
    ceiling = sum(runpod.COST_CEILING[g] * args.max_runtime / 3600 for g in gpus)
    print(
        f"{'LIVE' if args.live else 'DRY-RUN'} RunPod smoke: "
        f"gpus={','.join(gpus)} dc={runpod.DATACENTER}",
        flush=True,
    )
    print(
        f"hard runtime backstop={args.max_runtime}s; "
        f"price-ceiling exposure=${ceiling:.2f}",
        flush=True,
    )
    if not args.live:
        print("nothing created; add --live to spend")
        return 0

    created: list[tuple[str, str, float]] = []
    destroyed: set[str] = set()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    started = time.monotonic()
    try:
        stamp = int(time.time())
        for gpu in gpus:
            pod_id = runpod.create_pod(
                f"pinglab-smoke-{gpu}-{stamp}",
                gpu,
                datacenter=runpod.DATACENTER,
                volume_id=runpod.VOLUME_ID,
                env={"MAX_RUNTIME": str(args.max_runtime)},
            )
            created.append((pod_id, gpu, time.monotonic()))
            print(f"[{gpu}] created {pod_id}", flush=True)

        prices = _pod_prices()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(created)) as pool:
            futures = {
                pool.submit(_exercise, pod_id, gpu, when, args.timeout): (pod_id, gpu)
                for pod_id, gpu, when in created
            }
            for future in concurrent.futures.as_completed(futures):
                pod_id, gpu = futures[future]
                try:
                    result = future.result()
                    result["cost_per_hour"] = prices.get(pod_id)
                    results.append(result)
                    print(
                        f"[{gpu}] PASS ready={result['ready_s']:.1f}s "
                        f"fp32={result['tflops']:.1f} TFLOP/s",
                        flush=True,
                    )
                except Exception as exc:  # teardown must still run for either pod
                    failures.append({"gpu": gpu, "pod_id": pod_id, "error": str(exc)})
                    print(f"[{gpu}] FAIL {exc}", file=sys.stderr)
                finally:
                    # Do not keep a successful fast pod billing while a sibling
                    # is still cold-starting or waiting to time out.
                    print(f"[{gpu}] destroying {pod_id}", flush=True)
                    runpod.destroy_pod(pod_id)
                    destroyed.add(pod_id)
    finally:
        for pod_id, gpu, _ in created:
            if pod_id not in destroyed:
                print(f"[{gpu}] destroying {pod_id}", flush=True)
                runpod.destroy_pod(pod_id)

    summary = {
        "ok": not failures and len(results) == len(gpus),
        "datacenter": runpod.DATACENTER,
        "elapsed_s": time.monotonic() - started,
        "results": sorted(results, key=lambda item: item["gpu"]),
        "failures": failures,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
