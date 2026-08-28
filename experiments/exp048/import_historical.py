"""Import the approved non-Gold-2 archive as summary evidence, never as compute."""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp048 import historical, inputs


def fetch(source, directory):
    if source.rstrip("/") != historical.SOURCE:
        raise ValueError("only the explicitly approved historical source is supported")
    for name in (
        *historical.SOURCE_HASHES,
        *(f"payload/{n}" for n in sorted(historical.PAYLOAD_NAMES)),
    ):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as output:
            subprocess.run(
                ["rclone", "cat", f"{historical.SOURCE}/{name}"],
                stdout=output,
                check=True,
                timeout=60,
            )


def import_run(source, *, run_id=None):
    with tempfile.TemporaryDirectory(prefix="exp048-approved-import-") as tmp:
        directory = Path(tmp)
        fetch(source, directory)
        files = historical.archive_files(directory)
        history = historical.provenance(directory, files)
        with inputs.execution(
            REPO,
            "analyse",
            sources={},
            run_id=run_id,
            configuration=historical.IMPORT,
            operation="historical-import",
        ) as run:
            shutil.copytree(directory, run.provenance / "archive")
            shutil.copyfile(
                directory / "payload/numbers.json", run.export / "numbers.json"
            )
            run.record["historical"] = history
            run.record["source_file_mapping"] = {
                name: f"provenance/archive/{name}" for name in files
            }
            (run.directory / "README.md").write_text(
                "# exp048 historical summary import — not Gold-2\n\n"
                "Imported locally from the explicitly approved baseline-20260826 archive. "
                "Original producer: local; manifest date 2026-07-24. This operation is not "
                "a simulation, training job or raw-data replay.\n\n"
                "All 13 original payload files and both archive records are retained byte-for-byte "
                "under provenance/archive; export/numbers.json is the unchanged numerical source. "
                "The eight original figures are evidence, not outputs of a new presentation stage.\n\n"
                "Unresolved: archive r003 versus numerical r001; low-rate attribution to exp065 "
                "initial computation; missing exact checkpoint and simulator lineage. No Gold-2 "
                "bank is pinned. Raw streams and rasters were not found, so decoder replay and "
                "raster regeneration are unavailable. Analysis can reaggregate seed summaries; "
                "presentation may carry historical raster figures explicitly.\n"
            )
            if historical.archive_files(run.provenance / "archive") != files:
                raise ValueError("copied historical evidence differs")
        return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    import_run(args.source, run_id=args.run_id)


if __name__ == "__main__":
    main()
