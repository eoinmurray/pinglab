"""Import a validated saved digit trial; this entrypoint does not simulate."""

import argparse
import shutil
import sys
import tarfile
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
from experiments.exp164 import recipe
from pingstore.library import (
    file_sha256,
    source_run,
    stage_run,
    validate_operational_run_directory,
)


def preserve_history(source, destination):
    """Archive unique working history after authenticating duplicate run copies."""
    source = Path(source).resolve()
    if not source.is_dir():
        raise ValueError("history directory is missing")
    for duplicate in sorted((source / "collected").glob("*")):
        if not duplicate.is_dir():
            raise ValueError("unexpected recovered file")
        record = validate_operational_run_directory(duplicate)
        canonical = REPO / ".pingstore/runs" / duplicate.name
        actual = validate_operational_run_directory(canonical)
        if record != actual:
            raise ValueError("recovered run differs from canonical evidence")
    for archive in sorted(source.glob("*.tar.gz")):
        with tarfile.open(archive) as tar:
            for item in tar.getmembers():
                if not item.isfile():
                    continue
                path = Path(item.name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("invalid historical archive path")
                canonical = REPO / ".pingstore/runs" / path
                if tar.extractfile(item).read() != canonical.read_bytes():
                    raise ValueError("archived run differs from canonical evidence")
    files = [
        p
        for p in sorted(source.rglob("*"))
        if p.is_file()
        and "__pycache__" not in p.parts
        and "collected" not in p.relative_to(source).parts
        and not p.name.endswith(".tar.gz")
    ]
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(source))
    with ZipFile(destination) as z:
        for p in files:
            if z.read(str(p.relative_to(source))) != p.read_bytes():
                raise ValueError("history archive failed byte verification")
    return dict(
        sha256=file_sha256(destination),
        files=len(files),
        byte_verified=True,
        duplicate_run_copies_verified=True,
        purpose="historical working source/log attachment; not execution input",
    )


def compute(*, source, preserve_scratch=None, run_id=None):
    started = perf_counter()
    original = source_run(
        REPO / ".pingstore",
        source,
        experiment="exp163",
        stage="compute",
        phase="scratch-digit-routing",
    )
    cfg = original.record["execution"]["configuration"]
    recipe.configuration(source_configuration=cfg)
    with stage_run(
        REPO,
        recipe.SLUG,
        "compute",
        inputs={"original_trial": original},
        configuration=cfg,
        operation="import-saved-trial",
        run_id=run_id,
    ) as run:
        for name in ["spikes.npz", "inputs.npz", "images.npz", "circuit.svg"]:
            shutil.copyfile(original.file(name), run.export / name)
            if file_sha256(original.file(name)) != file_sha256(run.export / name):
                raise ValueError("imported bytes differ")
        for name in ["network.bundle", "execution/start"]:
            dst = original.restore_bundle(name, run.export / name)
            mapping = original.record["scientific_artifacts"]["native_paths"]
            for native, retained in mapping.items():
                if Path(native).is_relative_to(name):
                    if file_sha256(original.outputs / retained) != file_sha256(
                        dst / Path(native).relative_to(name)
                    ):
                        raise ValueError("imported native evidence differs")
        run.record["scientific_artifacts"] = {
            "snapshot_stages": {"execution": ["start"]}
        }
        run.record["execution"].update(
            no_simulation=True,
            snnlab_version=original.record["execution"]["snnlab_version"],
            original_simulation_seconds=original.record["execution"][
                "simulation_seconds"
            ],
        )
        if preserve_scratch:
            run.record["execution"]["migration_archive"] = preserve_history(
                preserve_scratch, run.export / "history.zip"
            )
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--preserve-scratch", type=Path)
    p.add_argument("--run-id")
    compute(**vars(p.parse_args()))
