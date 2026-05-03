"""Notebook runner for entry 000 — perf baseline.

Profiling reference workload. Trains two models per run so the perf
matrix covers both training backbones used elsewhere in the lab:

  - standard-snn (CUBANet path, recipe mirrors nb007)
  - coba         (COBANet path with ei_strength=0, recipe mirrors nb011)

Both recipes are verbatim copies of their science-owner notebooks; if
the science changes, change it there first and mirror here. The point
of nb000 is that recipes are *fixed* so wall-time / throughput numbers
reflect changes in the training stack itself (kernel launches, sync
points, compile coverage, memory layout), not recipe drift.

Anchors the "one architecture that runs efficiently on local MPS and
Modal" goal.

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
    # Recipe is intentionally a verbatim copy of nb007 — see nb007.py for
    # why each customisation is needed. nb000 must not drift from nb007;
    # if the science recipe changes, change it there first and mirror.
    return osc_base_args(out_dir, tier, build_as=MODEL) + [
        "--kaiming-init",
        "--readout",
        "li",
        "--surrogate-slope",
        "1",
        "--lr",
        "0.01",
        "--batch-size",
        "256",
    ]


def build_coba_osc_args(tier: str, out_dir: Path) -> list[str]:
    """Mirrors nb011's coba recipe verbatim. coba is dispatched via COBANet
    with ei_strength=0, so this exercises the COBANet compile path next to
    standard-snn's CUBANet path — both backbones tested per nb000 run."""
    return osc_base_args(out_dir, tier, build_as="ping") + [
        "--ei-strength",
        "0",
        "--v-grad-dampen",
        "1000",
        "--w-in",
        "0.3",
        "--w-in-sparsity",
        "0.95",
        "--lr",
        "0.0001",
    ]


def perf_criteria(figures: Path, run_dir: Path, tier: str) -> list[dict]:
    """Perf-baseline gates: confirm the workload ran end-to-end and
    populated the perf block. Deliberately permissive on the science
    side — nb000 inherits nb007's recipe at every tier (incl. ones the
    science is not calibrated for), so failing on rate-band overshoot
    or low accuracy at small/medium would just be noise. Science gates
    live in nb007."""
    crits: list[dict] = []
    figs_root = figures.parents[2]

    def artifact(name: str, label: str) -> None:
        path = figures / name
        ok = path.exists() and path.stat().st_size > 0
        href = "/" + str(path.relative_to(figs_root)) if ok else None
        crits.append(
            {
                "label": label,
                "passed": bool(ok),
                "detail": f"{path.name} ({path.stat().st_size} bytes)"
                if ok
                else f"missing {path.name}",
                "detail_href": href,
            }
        )

    artifact("training_curves.png", "training curves rendered")
    artifact("firing_rates.png", "firing-rate trace rendered")
    artifact("training.mp4", "training video rendered")

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        crits.append(
            {
                "label": "training metrics present",
                "passed": False,
                "detail": f"missing {metrics_path.name}",
            }
        )
        return crits
    metrics = json.loads(metrics_path.read_text())
    last = metrics["epochs"][-1]
    rate = float(last.get("rate_e") or 0.0)
    final = float(last["acc"])
    perf = metrics.get("perf") or {}
    sps = perf.get("samples_per_sec_warm")

    crits.append(
        {
            "label": "hidden layer spiked (rate_e > 0)",
            "passed": rate > 0.0,
            "detail": f"rate_e={rate:.2f} Hz",
        }
    )
    crits.append(
        {
            "label": "forward/backward did something (acc > 1%)",
            "passed": final > 1.0,
            "detail": f"final={final:.2f}%",
        }
    )
    crits.append(
        {
            "label": "perf block populated",
            "passed": isinstance(sps, (int, float)) and sps > 0,
            "detail": (
                f"samples_per_sec_warm={sps:.1f}"
                if isinstance(sps, (int, float))
                else "samples_per_sec_warm missing"
            ),
        }
    )

    # Extras-aware gate: confirm the coba secondary (COBANet path) also
    # produced metrics. run_dir is artifacts/nb000/train/; the secondary
    # trainer writes to its sibling artifacts/nb000/train_coba/.
    coba_metrics = run_dir.parent / "train_coba" / "metrics.json"
    coba_ok = coba_metrics.exists()
    coba_sps = None
    if coba_ok:
        coba_sps = (
            json.loads(coba_metrics.read_text())
            .get("perf", {})
            .get("samples_per_sec_warm")
        )
    crits.append(
        {
            "label": "coba (COBANet path) perf populated",
            "passed": isinstance(coba_sps, (int, float)) and coba_sps > 0,
            "detail": (
                f"samples_per_sec_warm={coba_sps:.1f}"
                if isinstance(coba_sps, (int, float))
                else f"missing {coba_metrics.name}"
            ),
        }
    )
    return crits


if __name__ == "__main__":
    run(
        SLUG,
        MODEL,
        build_osc_args,
        criteria_fn=perf_criteria,
        track_baselines=True,
        extra_train_models=[("coba", build_coba_osc_args)],
    )
