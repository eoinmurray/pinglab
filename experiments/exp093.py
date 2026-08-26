"""EXP093: compare legacy and current manuscript figures from gold-star runs."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers.cli import parse_meta  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp093"
STATUS = "draft"
LEGACY_ARCHIVE_ID = "exp022-cc36be1"
CURRENT_ARCHIVE_ID = "ggs-production-composite-20260821-6d9c38eb"
DEFAULT_STORE = "r2:pinglab/campaigns"
LEGACY_BUNDLE = "derived/artifacts-data.tar.gz"

# Order follows the manuscript figure sequence in writings/exp092.typ. Only figures
# present in both immutable campaign archives belong in this before/after review.
MANUSCRIPT_FIGURES = (
    ("exp025", "results_compound.png", "Trained PING versus COBA"),
    ("exp038", "loop_transfer_compound.png", "Inference-time loop activation"),
    ("exp049", "training_curves.svg", "Released-loop training"),
    ("exp041", "rate_vs_fgamma.svg", "Rate versus gamma frequency"),
    ("exp046", "spikes_per_cycle_distribution.svg", "Spikes per gamma cycle"),
    ("exp037", "perturbation_curves.svg", "Spike perturbation robustness"),
    ("exp042", "rhythm_compound.png", "Inhibitory timing perturbations"),
    ("exp044", "dt_sweep.svg", "Integration-timestep invariance"),
)

SCALE = {
    "status": STATUS,
    "completed_methods": [1, 2, 3],
    "legacy_archive_id": LEGACY_ARCHIVE_ID,
    "current_archive_id": CURRENT_ARCHIVE_ID,
    "manuscript_figures": len(MANUSCRIPT_FIGURES),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rclone_json(path: str) -> dict[str, Any]:
    result = subprocess.run(
        ["rclone", "cat", path],
        check=True,
        capture_output=True,
        text=True,
    )
    if not result.stdout.strip():
        raise RuntimeError(f"archive object is missing or empty: {path}")
    return json.loads(result.stdout)


def _rclone_copy(source: str, destination: Path) -> None:
    subprocess.run(
        ["rclone", "copyto", source, str(destination)],
        check=True,
        capture_output=True,
        text=True,
    )


def _validated_run(payload: dict[str, Any], archive_id: str) -> dict[str, Any]:
    required = {"run_id", "archive", "source"}
    if not required <= payload.keys() or not isinstance(payload["archive"], dict):
        raise RuntimeError(f"invalid run manifest for archive {archive_id}")
    archive = payload["archive"]
    if archive.get("archive_id") != archive_id or not archive.get("uri"):
        raise RuntimeError(f"archive identity mismatch for {archive_id}")
    return payload


def _validated_inventory(
    payload: dict[str, Any], run_id: str, archive_id: str
) -> dict[str, Any]:
    if payload.get("run_id") != run_id:
        raise RuntimeError(f"run and inventory identities differ for {archive_id}")
    if not isinstance(payload.get("files"), list):
        raise RuntimeError(f"invalid inventory file list for {archive_id}")
    digest = payload.get("payload_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(f"invalid inventory digest for {archive_id}")
    return payload


def _archive_metadata(
    store: str, archive_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = f"{store.rstrip('/')}/{archive_id}"
    run = _validated_run(_rclone_json(f"{root}/run.json"), archive_id)
    inventory = _validated_inventory(
        _rclone_json(f"{root}/inventory.json"), run["run_id"], archive_id
    )
    return run, inventory


def _inventory_files(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["path"]: row for row in inventory["files"]}


def _safe_member(bundle: tarfile.TarFile, name: str) -> tarfile.TarInfo:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe legacy bundle member: {name}")
    try:
        member = bundle.getmember(name)
    except KeyError as error:
        raise RuntimeError(f"legacy manuscript figure is missing: {name}") from error
    if not member.isfile():
        raise RuntimeError(f"legacy manuscript figure is not a file: {name}")
    return member


def build_manuscript_comparison(
    *,
    output_root: Path,
    current_artifact_root: Path,
    store: str = DEFAULT_STORE,
) -> dict[str, Any]:
    """Materialise compact legacy figures and a verified comparison manifest."""

    legacy_store = f"{store.rstrip('/')}/legacy"
    legacy_run, legacy_inventory = _archive_metadata(legacy_store, LEGACY_ARCHIVE_ID)
    current_run, current_inventory = _archive_metadata(store, CURRENT_ARCHIVE_ID)
    legacy_files = _inventory_files(legacy_inventory)
    current_files = _inventory_files(current_inventory)

    bundle_row = legacy_files.get(LEGACY_BUNDLE)
    if bundle_row is None:
        raise RuntimeError(f"legacy inventory does not contain {LEGACY_BUNDLE}")

    output_root = output_root.resolve()
    current_artifact_root = current_artifact_root.resolve()
    staging = output_root.with_name(f".{output_root.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    figures: list[dict[str, Any]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="pinglab-exp093-") as temporary:
            bundle_path = Path(temporary) / "artifacts-data.tar.gz"
            _rclone_copy(
                f"{legacy_store}/{LEGACY_ARCHIVE_ID}/{LEGACY_BUNDLE}", bundle_path
            )
            if bundle_path.stat().st_size != bundle_row["size_bytes"]:
                raise RuntimeError(
                    "legacy derived bundle size does not match inventory"
                )
            if _sha256(bundle_path) != bundle_row["sha256"]:
                raise RuntimeError(
                    "legacy derived bundle hash does not match inventory"
                )

            with tarfile.open(bundle_path, "r:gz") as bundle:
                for experiment, filename, title in MANUSCRIPT_FIGURES:
                    relative = f"{experiment}/{filename}"
                    legacy_member_name = f".artifacts/{relative}"
                    current_inventory_path = f"derived/.artifacts/{relative}"
                    current_row = current_files.get(current_inventory_path)
                    if current_row is None:
                        raise RuntimeError(
                            f"current archive inventory is missing {current_inventory_path}"
                        )
                    current_path = current_artifact_root / relative
                    if not current_path.is_file():
                        raise RuntimeError(
                            f"current publication figure is missing: {current_path}"
                        )
                    if current_path.stat().st_size != current_row["size_bytes"]:
                        raise RuntimeError(
                            f"current figure size does not match archive: {relative}"
                        )
                    current_sha = _sha256(current_path)
                    if current_sha != current_row["sha256"]:
                        raise RuntimeError(
                            f"current figure hash does not match archive: {relative}"
                        )

                    member = _safe_member(bundle, legacy_member_name)
                    legacy_path = staging / "legacy" / relative
                    legacy_path.parent.mkdir(parents=True, exist_ok=True)
                    source = bundle.extractfile(member)
                    if source is None:
                        raise RuntimeError(
                            f"could not read legacy figure: {legacy_member_name}"
                        )
                    with source, legacy_path.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
                    legacy_sha = _sha256(legacy_path)
                    figures.append(
                        {
                            "experiment": experiment,
                            "filename": filename,
                            "title": title,
                            "status": "unchanged"
                            if legacy_sha == current_sha
                            else "changed",
                            "legacy_path": f"/.artifacts/exp093/legacy/{relative}",
                            "current_path": f"/.artifacts/{relative}",
                            "legacy_sha256": legacy_sha,
                            "current_sha256": current_sha,
                        }
                    )

        payload = {
            "schema": "pinglab.manuscript-figure-comparison/v1",
            "legacy": {
                "run_id": legacy_run["run_id"],
                "archive_id": LEGACY_ARCHIVE_ID,
                "uri": legacy_run["archive"]["uri"],
                "payload_digest": legacy_inventory["payload_digest"],
                "derived_bundle_sha256": bundle_row["sha256"],
            },
            "current": {
                "run_id": current_run["run_id"],
                "archive_id": CURRENT_ARCHIVE_ID,
                "uri": current_run["archive"]["uri"],
                "git_commit": current_run["source"]["git_commit"],
                "payload_digest": current_inventory["payload_digest"],
            },
            "figures": figures,
        }
        (staging / "numbers.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        if output_root.exists():
            shutil.rmtree(output_root)
        staging.replace(output_root)
        return payload
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    """Run the fixed archive-comparison recipe and publish its compact result."""

    parse_meta(sys.argv)
    started = time.monotonic()
    run_id = next_run_id(SLUG)
    with published_run(SLUG, run_id, scale=SCALE, make_artifacts=False) as (
        _artifacts,
        staging,
    ):
        payload = build_manuscript_comparison(
            output_root=staging,
            current_artifact_root=REPO / "artifacts" / "data",
        )
        write_numbers(
            staging,
            run_id=run_id,
            duration_s=time.monotonic() - started,
            payload=payload,
        )


if __name__ == "__main__":
    main()
