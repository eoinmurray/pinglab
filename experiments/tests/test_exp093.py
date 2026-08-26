from __future__ import annotations

import io
import tarfile
from pathlib import Path

from experiments import exp093


def test_manuscript_comparison_materialises_verified_pairs(
    tmp_path: Path, monkeypatch
) -> None:
    legacy_files: dict[str, bytes] = {}
    current_root = tmp_path / "current"
    for experiment, filename, _title in exp093.MANUSCRIPT_FIGURES:
        relative = f"{experiment}/{filename}"
        legacy_files[f".artifacts/{relative}"] = f"legacy-{relative}".encode()
        current_path = current_root / relative
        current_path.parent.mkdir(parents=True, exist_ok=True)
        current_path.write_bytes(f"current-{relative}".encode())

    bundle_path = tmp_path / "legacy.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as bundle:
        for name, contents in legacy_files.items():
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            bundle.addfile(info, io.BytesIO(contents))

    def row(path: Path, inventory_path: str) -> dict[str, object]:
        return {
            "path": inventory_path,
            "size_bytes": path.stat().st_size,
            "sha256": exp093._sha256(path),
            "role": "derived",
        }

    legacy_run = {
        "run_id": "legacy-exp022-cc36be1",
        "archive": {
            "archive_id": exp093.LEGACY_ARCHIVE_ID,
            "uri": "r2://test/legacy/exp022-cc36be1",
        },
    }
    current_run = {
        "run_id": exp093.CURRENT_ARCHIVE_ID,
        "archive": {
            "archive_id": exp093.CURRENT_ARCHIVE_ID,
            "uri": f"r2://test/{exp093.CURRENT_ARCHIVE_ID}",
        },
        "source": {"git_commit": "a" * 40},
    }
    legacy_inventory = {
        "run_id": legacy_run["run_id"],
        "payload_digest": "b" * 64,
        "files": [row(bundle_path, exp093.LEGACY_BUNDLE)],
    }
    current_inventory = {
        "run_id": current_run["run_id"],
        "payload_digest": "c" * 64,
        "files": [
            row(
                current_root / experiment / filename,
                f"derived/.artifacts/{experiment}/{filename}",
            )
            for experiment, filename, _title in exp093.MANUSCRIPT_FIGURES
        ],
    }

    def fake_metadata(store: str, archive_id: str):
        if store.endswith("/legacy"):
            return legacy_run, legacy_inventory
        return current_run, current_inventory

    monkeypatch.setattr(exp093, "_archive_metadata", fake_metadata)
    monkeypatch.setattr(
        exp093,
        "_rclone_copy",
        lambda _source, destination: destination.write_bytes(bundle_path.read_bytes()),
    )

    output = tmp_path / "exp093"
    payload = exp093.build_manuscript_comparison(
        output_root=output,
        current_artifact_root=current_root,
        store="r2:test/campaigns",
    )

    assert len(payload["figures"]) == len(exp093.MANUSCRIPT_FIGURES)
    assert {figure["status"] for figure in payload["figures"]} == {"changed"}
    assert (output / "numbers.json").is_file()
    assert (output / "legacy" / "exp025" / "results_compound.png").is_file()
