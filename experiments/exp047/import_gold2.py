"""Import an explicitly approved Gold-2 selection; no simulation or publication."""

import argparse
import gzip
import hashlib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp047 import evidence, historical, measurements, recipe
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.native import execution_origin
from pingstore.stages import stage_run


def copy_selected(archive, run, row):
    source = historical.safe_path(archive, row["path"])
    data = source.read_bytes()
    if (
        len(data) != row["size_bytes"]
        or hashlib.sha256(data).hexdigest() != row["sha256"]
    ):
        raise PingstoreError("Gold-2 source changed during import")
    encoded = gzip.compress(data, mtime=0) if row["encoding"] == "gzip" else data
    target = run.directory / row["target"]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(encoded)
    retained = target.read_bytes()
    recovered = gzip.decompress(retained) if row["encoding"] == "gzip" else retained
    if recovered != data or len(retained) != row["retained_bytes"]:
        raise PingstoreError("imported evidence differs from approved source")
    return {**row, "target_sha256": hashlib.sha256(retained).hexdigest()}


def import_subset(archive, plan, live_metadata):
    archive = Path(archive).absolute()
    if historical.make_plan(archive, live_metadata) != plan:
        raise PingstoreError("import plan differs; a new selection requires approval")
    if execution_origin() != "local":
        raise PingstoreError("this historical import must execute locally")
    with stage_run(
        REPO,
        recipe.SLUG,
        "compute",
        inputs={},
        configuration=plan["recipe"],
        operation="historical-import",
    ) as run:
        mappings = [copy_selected(archive, run, row) for row in plan["files"]]
        write_json_atomic(run.provenance / "import-plan.json", plan)
        write_json_atomic(
            run.provenance / "file-mapping.json",
            {
                "schema": "exp047.gold2-file-mapping/v1",
                "files": mappings,
            },
        )
        shutil.copyfile(__file__, run.provenance / "import_gold2.py")
        shutil.copyfile(historical.__file__, run.provenance / "historical.py")
        cfg = plan["recipe"]
        write_json_atomic(
            run.export / "evidence.json",
            {
                "schema": "exp047.compute/v1",
                "recipe": cfg,
                "jobs": recipe.jobs(cfg),
            },
        )
        rows = evidence.rows(run.export, run.provenance, cfg)
        actual = measurements.analyse_rows(rows, cfg)
        original = load_json(
            run.provenance / "archive" / historical.DERIVED / "numbers.json"
        )
        if any(
            actual[key] != original[key]
            for key in ("config", "definition", "raw", "summary")
        ):
            raise PingstoreError("retained historical numerical replay differs")
        run.record["historical_import"] = {
            "archive_uri": historical.URI,
            "producer": plan["producer"],
            "archive_metadata": plan["metadata"],
            "plan": "provenance/import-plan.json",
            "mapping": "provenance/file-mapping.json",
            "original_records": "provenance/archive",
            "source_bytes": plan["source_bytes"],
            "retained_source_bytes": sum(row["retained_bytes"] for row in mappings),
            "scientific_metric_bytes": plan["scientific_metric_bytes"],
            "simulation_executed": False,
            "upstream_banks": [],
            "preservation": "All 42 original metrics and 168 simulation sidecars retained unchanged. Full archive inventory compressed losslessly. No subsampling; Gold-2 remains unchanged.",
        }
        (run.directory / "README.md").write_text(
            "# exp047: local historical Gold-2 import\n\n"
            "This operation imported retained evidence; it did not simulate or train.\n"
            f"Original producer: {historical.CAMPAIGN}, commit {historical.COMMIT},\n"
            f"Slurm job {historical.JOB}. The importing host and time are recorded separately\n"
            "in run.json. Original naive probe timestamps retain their original meaning.\n\n"
            "All 42 unique simulation metrics and their 168 configuration/command/log\n"
            "sidecars are unchanged. These support 54 reported rows because matching\n"
            "conditions share the same simulations. No trials or seeds were subsampled.\n"
            "There is no upstream model bank. Raw spikes and realised weights were not\n"
            "retained historically, so these rates cannot be remeasured from spikes.\n\n"
            "Original HPC records, lineage, numbers and archive manifest are retained\n"
            "under provenance/archive. The entire original inventory is losslessly\n"
            "gzipped there. import-plan.json and file-mapping.json record all source and\n"
            "target hashes and sizes. The archive and every older run remain unchanged.\n"
            "Independent analysis and presentation follow; publication is not automatic.\n"
        )
        if historical.make_plan(archive, live_metadata) != plan:
            raise PingstoreError("Gold-2 evidence changed during import")
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--plan", type=Path, required=True, help="explicit approved selection plan"
    )
    parser.add_argument(
        "--live-metadata",
        type=Path,
        required=True,
        help="freshly retrieved R2 run.json and inventory.json",
    )
    args = parser.parse_args()
    try:
        import_subset(args.archive, load_json(args.plan), args.live_metadata)
    except (PingstoreError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"exp047 Gold-2 import: {exc}\n")


if __name__ == "__main__":
    main()
