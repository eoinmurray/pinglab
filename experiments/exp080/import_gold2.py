"""Import the approved Gold-2 selection; never train, simulate or publish."""

import argparse
import gzip
import hashlib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp080 import evidence, historical, measurements, recipe
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.native import execution_origin
from pingstore.stages import stage_run


def copy_selected(archive, run, row):
    source = historical.safe_path(archive, row["path"])
    original = source.read_bytes()
    if (
        len(original) != row["size_bytes"]
        or hashlib.sha256(original).hexdigest() != row["sha256"]
    ):
        raise PingstoreError("Gold-2 source changed during import")
    encoded = (
        gzip.compress(original, mtime=0) if row["encoding"] == "gzip" else original
    )
    target = run.directory / row["target"]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(encoded)
    retained = target.read_bytes()
    recovered = gzip.decompress(retained) if row["encoding"] == "gzip" else retained
    if recovered != original or len(retained) != row["retained_bytes"]:
        raise PingstoreError("retained evidence differs from approved source")
    return {**row, "target_sha256": hashlib.sha256(retained).hexdigest()}


def import_subset(archive, plan, live_metadata, *, verification=None):
    archive = Path(archive).absolute()
    if historical.make_plan(archive, live_metadata) != plan:
        raise PingstoreError(
            "import selection differs; changed plans need fresh approval"
        )
    if execution_origin() != "local":
        raise PingstoreError("this historical import must execute locally")
    verification_bytes = None
    if verification is not None:
        verification = Path(verification)
        record = load_json(verification)
        checks = {row["file"]: row for row in record["checks"]}
        if set(checks) != set(plan["metadata"]) or any(
            row.get("identical") is not True
            or row.get("sha256") != plan["metadata"][name]["sha256"]
            or row.get("cached_sha256") != row.get("sha256")
            for name, row in checks.items()
        ):
            raise PingstoreError("R2 verification does not match approved metadata")
        verification_bytes = verification.read_bytes()
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
                "schema": "exp080.gold2-file-mapping/v1",
                "files": mappings,
            },
        )
        if verification_bytes is not None:
            (run.provenance / "r2-verification.json").write_bytes(verification_bytes)
        shutil.copyfile(__file__, run.provenance / "import_gold2.py")
        shutil.copyfile(historical.__file__, run.provenance / "historical.py")
        original = load_json(
            run.provenance / "archive" / historical.DERIVED / "numbers.json"
        )
        write_json_atomic(
            run.export / "evidence.json", historical.compute_document(original)
        )
        _, correctness = evidence.validate(run.export, plan["recipe"], historical=True)
        if measurements.analyze(correctness, plan["recipe"]) != original["decision"]:
            raise PingstoreError(
                "imported scientific results differ from original evidence"
            )
        run.record["historical_import"] = {
            "archive_uri": historical.URI,
            "producer": plan["producer"],
            "archive_metadata": plan["metadata"],
            "plan": "provenance/import-plan.json",
            "mapping": "provenance/file-mapping.json",
            "original_records": "provenance/archive",
            "source_bytes": plan["source_bytes"],
            "retained_source_bytes": sum(row["retained_bytes"] for row in mappings),
            "scientific_export_bytes": plan["scientific_export_bytes"],
            "simulation_executed": False,
            "training_executed": False,
            "upstream_banks": [],
            "preservation": "All three validation-selected decoder checkpoints, full histories, all 120000 correctness values and original feature PNG retained unchanged. Full archive inventory compressed losslessly. No subsampling; Gold-2 is unchanged.",
        }
        (run.directory / "README.md").write_text(
            "# exp080: local historical Gold-2 import\n\n"
            "This operation imported retained evidence; it did not train or simulate.\n"
            f"Original scientific producer: {historical.CAMPAIGN}, commit {historical.COMMIT},\n"
            f"Slurm job {historical.JOB}. Original producer times and host are retained\n"
            "separately from the importing operation in run.json.\n\n"
            "All three validation-selected decoders, full training histories and held-out\n"
            "correctness/labels/rates/seeds are byte-identical to Gold-2. The original\n"
            "feature illustration is carried unchanged: raw illustrative features were\n"
            "not retained historically. No raw MNIST data or ephemeral training features\n"
            "were archived; original dataset hashes remain in the evidence.\n\n"
            "The full Gold-2 inventory is losslessly gzipped under provenance/archive.\n"
            "Original manifests, lineage, commands, Slurm logs and results are retained\n"
            "there. import-plan.json and file-mapping.json pin every selected source\n"
            "and target byte. No upstream bank is needed, copied or referenced.\n\n"
            "Analysis and presentation are separate commands. Nothing was materialized\n"
            "or published, and all pre-existing completed runs and Gold-2 are unchanged.\n"
        )
        if historical.make_plan(archive, live_metadata) != plan:
            raise PingstoreError("Gold-2 evidence changed during import")
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--plan", type=Path, required=True, help="approved selection plan"
    )
    parser.add_argument("--live-metadata", type=Path, required=True)
    parser.add_argument(
        "--verification", type=Path, help="R2 retrieval commands and checksum evidence"
    )
    args = parser.parse_args()
    try:
        import_subset(
            args.archive,
            load_json(args.plan),
            args.live_metadata,
            verification=args.verification,
        )
    except (PingstoreError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"exp080 Gold-2 import: {exc}\n")


if __name__ == "__main__":
    main()
