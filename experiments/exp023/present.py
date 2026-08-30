"""Render completed exp023 measurements and pinned rasters; never simulate."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp023 import inputs, plots, recipe
from experiments.helpers import theme
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import stage_run


def present(identity: str, *, run_id: str | None = None) -> str:
    analysis = inputs.source(REPO, identity, "analyse")
    if set(analysis.record["inputs"]) != {"compute"}:
        raise PingstoreError("exp023 analysis must pin exactly its computation")
    ref = analysis.record["inputs"]["compute"]
    compute = inputs.source(REPO, ref["run_id"], "compute", reference=ref)
    cfg = inputs.configuration(compute)
    results = load_json(analysis.export / "results.json")
    if (
        results.get("schema") != "exp023.analysis/v1"
        or results.get("config") != cfg
        or results.get("measurement") != analysis.record["execution"]["configuration"]
    ):
        raise PingstoreError("unsupported or inconsistent exp023 analysis payload")
    with np.load(analysis.export / "spectra.npz", allow_pickle=False) as data:
        spectra = {
            cell: {
                key: np.array(data[f"{cell}__{key}"])
                for key in ("frequency_hz", "density")
            }
            for cell in cfg["cells"]
        }
    with np.load(analysis.export / "traces.npz", allow_pickle=False) as data:
        traces = {
            cell: {
                key.removeprefix(cell + "__"): np.array(data[key])
                for key in data.files
                if key.startswith(cell + "__")
            }
            for cell in cfg["cells"]
        }
    snaps = {}
    for cell in cfg["cells"]:
        with np.load(
            compute.file("scope", cell, "snapshot.npz"), allow_pickle=False
        ) as data:
            snaps[cell] = {key: np.array(data[key]) for key in ("spk_e", "spk_i", "dt")}
    started = time.monotonic()
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs={"analysis": analysis, "compute": compute},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        theme.set_paper_mode(True)
        plots.plot_architecture(run.export / "architecture")
        for cell in cfg["cells"]:
            plots.plot_traces(
                traces[cell],
                results["raster"][cell],
                cfg["biophysics"],
                run.export / f"traces__{cell}",
                cell.upper(),
            )
        titles = {
            "coba": "A   COBA — recurrent loop off",
            "ping": "B   PING — recurrent loop active",
        }
        for name, include_arch in (
            ("raster_compound", False),
            ("overview_compound", True),
        ):
            plots.plot_raster_compound(
                snaps,
                results["fi_curves"],
                run.export / name,
                titles,
                spectra,
                results["f_gamma_hz"],
                results["measurement"]["frequency_band_hz"],
                include_arch=include_arch,
            )
        run.record["presentation_lineage"] = {
            "measurements": analysis.reference,
            "rasters": compute.reference,
            "operation": "render retained measurements and spikes; no simulation or remeasurement",
        }
        write_json_atomic(
            run.export / "numbers.json",
            {
                **results,
                "run_id": run.run_id,
                "duration_s": time.monotonic() - started,
                "git_sha": run.record["provenance"]["git_commit"],
            },
        )
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="completed exp023 analyse run ID"
    )
    parser.add_argument("--run-id", help="unused v3 identity reserved before dispatch")
    args = parser.parse_args()
    present(args.source, run_id=args.run_id)


if __name__ == "__main__":
    main()
