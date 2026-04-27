"""Notebook runner for entry 000 — perf baseline.

Profiling reference workload. Recipe-identical clone of nb006 (standard-snn
LIF classifier on MNIST at dt=0.1 ms): same model, same hyperparameters,
same success criteria. Exists as a fixed reference point for performance
work — measurements taken here over time should reflect optimisation /
regression in the shared training stack, not recipe drift. Anchors the
"one architecture that runs efficiently on local MPS and Modal" goal.

Notebook entry: src/docs/src/pages/notebooks/nb000.mdx
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _per_model import run, osc_base_args  # noqa: E402

SLUG = "nb000"
MODEL = "standard-snn"


def build_osc_args(tier: str, out_dir: Path) -> list[str]:
    # Recipe is intentionally a verbatim copy of nb006 — see nb006.py for
    # why each customisation is needed. nb000 must not drift from nb006;
    # if the science recipe changes, change it there first and mirror.
    return osc_base_args(out_dir, tier, build_as=MODEL) + [
        "--kaiming-init",
        "--readout", "li",
        "--surrogate-slope", "1",
        "--lr", "0.01",
        "--batch-size", "256",
    ]


def perf_criteria(figures: Path, run_dir: Path, tier: str) -> list[dict]:
    """Perf-baseline gates: confirm the workload ran end-to-end and
    populated the perf block. Deliberately permissive on the science
    side — nb000 inherits nb006's recipe at every tier (incl. ones the
    science is not calibrated for), so failing on rate-band overshoot
    or low accuracy at small/medium would just be noise. Science gates
    live in nb006."""
    crits: list[dict] = []
    figs_root = figures.parents[2]

    def artifact(name: str, label: str) -> None:
        path = figures / name
        ok = path.exists() and path.stat().st_size > 0
        href = "/" + str(path.relative_to(figs_root)) if ok else None
        crits.append({
            "label": label,
            "passed": bool(ok),
            "detail": f"{path.name} ({path.stat().st_size} bytes)" if ok
                      else f"missing {path.name}",
            "detail_href": href,
        })

    artifact("training_curves.png", "training curves rendered")
    artifact("firing_rates.png", "firing-rate trace rendered")
    artifact("training.mp4", "training video rendered")

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        crits.append({"label": "training metrics present", "passed": False,
                      "detail": f"missing {metrics_path.name}"})
        return crits
    metrics = json.loads(metrics_path.read_text())
    last = metrics["epochs"][-1]
    rate = float(last.get("rate_e") or 0.0)
    final = float(last["acc"])
    perf = metrics.get("perf") or {}
    sps = perf.get("samples_per_sec_warm")

    crits.append({
        "label": "hidden layer spiked (rate_e > 0)",
        "passed": rate > 0.0,
        "detail": f"rate_e={rate:.2f} Hz",
    })
    crits.append({
        "label": "forward/backward did something (acc > 1%)",
        "passed": final > 1.0,
        "detail": f"final={final:.2f}%",
    })
    crits.append({
        "label": "perf block populated",
        "passed": isinstance(sps, (int, float)) and sps > 0,
        "detail": (f"samples_per_sec_warm={sps:.1f}"
                   if isinstance(sps, (int, float))
                   else "samples_per_sec_warm missing"),
    })
    return crits


if __name__ == "__main__":
    run(SLUG, MODEL, build_osc_args,
        criteria_fn=perf_criteria, track_baselines=True)
