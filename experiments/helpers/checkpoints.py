"""Read explicitly pinned checkpoint data without importing its producer."""

from pathlib import Path

from pingstore.contracts import file_sha256, load_json


def pinned_checkpoint(
    training_unit: Path, *, role: str, filename: str, sha256: str
) -> Path:
    """Validate the role registration and exact file identity supplied by a recipe."""
    record = load_json(training_unit / "metrics.json").get("checkpoints", {}).get(role)
    if not isinstance(record, dict) or record.get("filename") != filename:
        raise ValueError(f"Checkpoint role {role!r} does not name {filename!r}")
    if Path(filename).name != filename:
        raise ValueError("Checkpoint filename must be a single path component")
    path = training_unit / filename
    if record.get("sha256") != sha256 or file_sha256(path) != sha256:
        raise ValueError(f"Checkpoint identity mismatch: {path}")
    return path
