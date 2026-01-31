import os
import sys
import time
from typing import Any

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None
try:
    import resource  # type: ignore
except Exception:  # pragma: no cover
    resource = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from api.app import RunRequest, run_simulation


def _mem_mb() -> float | None:
    if psutil is None:
        if resource is None:
            return None
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS returns bytes, linux returns KB; handle both.
        if usage > 10**7:
            return usage / (1024 * 1024)
        return usage / 1024
    proc = psutil.Process()
    return proc.memory_info().rss / (1024 * 1024)


def _format_mb(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f} MB"


def run_case(name: str, payload: dict[str, Any]) -> None:
    req = RunRequest.model_validate(payload)
    mem_before = _mem_mb()
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    response = run_simulation(req)
    cpu1 = time.process_time()
    wall_ms = (time.perf_counter() - t0) * 1000
    cpu_ms = (cpu1 - cpu0) * 1000
    mem_after = _mem_mb()
    max_rss = None
    if psutil is not None:
        max_rss = psutil.Process().memory_info().rss / (1024 * 1024)

    print(f"\n{name}")
    print(f"- wall: {wall_ms:.1f} ms")
    print(f"- cpu:  {cpu_ms:.1f} ms")
    print(f"- mem before: {_format_mb(mem_before)}")
    print(f"- mem after:  {_format_mb(mem_after)}")
    if max_rss is not None:
        print(f"- rss now:   {_format_mb(max_rss)}")
    print(f"- runtime_ms (from app): {response.runtime_ms:.1f} ms")
    print(f"- spikes: {len(response.spikes.times)} (truncated={response.spikes_truncated})")


def main() -> None:
    cases = [
        ("default", {}),
        (
            "ui-default-ish",
            {
                "config": {
                    "N_E": 400,
                    "N_I": 100,
                    "T": 500,
                }
            },
        ),
        (
            "larger",
            {
                "config": {
                    "N_E": 800,
                    "N_I": 200,
                    "T": 1000,
                }
            },
        ),
    ]
    for name, payload in cases:
        run_case(name, payload)


if __name__ == "__main__":
    main()
