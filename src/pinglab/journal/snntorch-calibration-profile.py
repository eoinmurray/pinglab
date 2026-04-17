"""Profile the real train() loop and attribute .item() cost by caller.

Reproduces the 100x per-call slowdown on the in-repo snntorch path
relative to the snntorch-library wrapper. Run as:

    uv run python src/pinglab/journal/snntorch-calibration-profile.py

Evidence collected on 2026-04-17 (CPU, 1 epoch, 128 samples, bs 64,
dt 0.25 ms, T 600 ms, 1024 hidden):

    snntorch         12.0 s total, 116 item() calls from train loop,
                     9.691 s cumulative in item()  (~83 ms/call)
    snntorch-library  4.6 s total, 124 item() calls from train loop,
                     0.100 s cumulative in item()  (~0.8 ms/call)

Forward and forward+backward in isolation are comparable between the
two paths; the gap appears only once a .item() reads a scalar out of
the autograd graph that SNNTorchNet builds up via its custom
SurrogateSpike.Function + per-step n_spk_tensors accumulator.
"""
from __future__ import annotations

import cProfile
import pstats
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

import models as M  # noqa: E402

# Match the journal config.
M.dt = 0.25
M.T_ms = 600.0
M.T_steps = int(M.T_ms / M.dt)
M.N_IN = 784
M.N_HID = 1024
M.N_OUT = 10
M.HIDDEN_SIZES = [1024]

import oscilloscope as osc  # noqa: E402


def run(model_name: str, out_root: Path) -> None:
    osc.train(
        model_name=model_name,
        lr=0.01,
        epochs=1,
        dt=M.dt,
        observe=False,
        max_samples=128,
        dataset="mnist",
        out_dir=out_root / model_name,
        kaiming_init=True,
        hidden_sizes=[M.N_HID],
    )


def main() -> None:
    out_root = Path("/tmp/snntorch-calibration-profile")
    for name in ("snntorch", "snntorch-library"):
        print(f"\n{'=' * 72}\n  {name}\n{'=' * 72}")
        prof = cProfile.Profile()
        t0 = time.perf_counter()
        prof.enable()
        run(name, out_root)
        prof.disable()
        print(f"total: {time.perf_counter() - t0:.1f}s")
        stats = pstats.Stats(prof).sort_stats("tottime")
        print("\n--- callers of .item() ---")
        stats.print_callers(0.05, "item")


if __name__ == "__main__":
    main()
