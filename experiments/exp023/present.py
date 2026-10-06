"""Render completed exp023 measurements and pinned rasters; never simulate."""

from __future__ import annotations

import argparse
import sys
import time
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp023 import inputs, plots, recipe
from experiments.helpers import theme
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import stage_run


def reported_configuration(compute, source_cfg: dict) -> tuple[dict, dict | None]:
    """Correct only the audited retained computation's refractory declaration."""
    if source_cfg["schema"] != "exp023.recipe/v1":
        return source_cfg, None
    if (
        compute.record["run_id"] != "exp023-r011-compute"
        or compute.record["payload_digest"]
        != "sha256:c08f1e26a231130156e54d294da5950f5191411775d79e012d21b5ba8e710671"
        or source_cfg["biophysics"]["refractory_E_ms"] != 3.0
        or source_cfg["biophysics"]["refractory_I_ms"] != 1.5
    ):
        raise PingstoreError("unaudited legacy refractory declaration")
    cfg = deepcopy(source_cfg)
    cfg["schema"] = "exp023.reported-configuration/v1"
    cfg["biophysics"].update(refractory_E_ms=1.2, refractory_I_ms=0.6)
    correction = {
        "schema": "exp023.refractory-metadata-correction/v1",
        "declared_ms": {"E": 3.0, "I": 1.5},
        "executed_ms": {"E": 1.2, "I": 0.6},
        "executed_steps": {"E": 12, "I": 6},
        "dt_ms": 0.1,
        "measurements_changed": False,
        "basis": "Audited simulator used 12/6 step counters; retained voltage reset holds corroborate the executed periods. Reporting correction only.",
        "source_audit_limit": "The recorded Git base is not a complete frozen dirty checkout; retained traces corroborate the counter audit.",
        "source_was_dirty": True,
        "source_recipe_schema": source_cfg["schema"],
    }
    return cfg, correction


def present(identity: str, *, run_id: str | None = None) -> str:
    analysis = inputs.source(REPO, identity, "analyse")
    if set(analysis.record["inputs"]) != {"compute"}:
        raise PingstoreError("exp023 analysis must pin exactly its computation")
    ref = analysis.record["inputs"]["compute"]
    compute = inputs.source(REPO, ref["run_id"], "compute", reference=ref)
    source_cfg = inputs.configuration(compute)
    cfg, correction = reported_configuration(compute, source_cfg)
    results = load_json(analysis.export / "results.json")
    if (
        results.get("schema") != "exp023.analysis/v1"
        or results.get("config") != source_cfg
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
            compute.file("scope", cell, "recording.npz"), allow_pickle=False
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
        if correction is not None:
            run.record["metadata_correction"] = correction
            run.record["execution"]["configuration"] = {
                "schema": "exp023.presentation-metadata/v1",
                "source_recipe": source_cfg,
                "reported_biophysics": cfg["biophysics"],
            }
            with (run.directory / "README.md").open("a") as history:
                history.write(
                    "\nRe-rendered from compute and analysis only. Corrected reported E/I refractory periods to 1.2/0.6 ms; retained measurements are unchanged. No simulation or analysis ran.\n"
                )
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
            "coba": "COBA — recurrent loop off",
            "ping": "PING — recurrent loop active",
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
                "config": cfg,
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
    parser.add_argument("--run-id", help="unused v4 identity reserved before dispatch")
    args = parser.parse_args()
    present(args.source, run_id=args.run_id)


if __name__ == "__main__":
    main()
