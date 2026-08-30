"""Build a read-only file-level audit of Pingstore evidence directories."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / ".pingstore" / "runs"
HERE = Path(__file__).resolve().parent

# These are the evidence files read semantically by completed downstream stages
# or by Pinglab's presentation projection. The rules were established by tracing
# the production readers for every completed descendant run.
DOWNSTREAM_RULES = {
    "exp022-r001-compute": {"imported-run.json"},
    "exp037-r001-compute": {"simulations/**/config.json"},
    "exp038-r001-compute": {"simulations/**/config.json"},
    "exp047-r001-compute": {"simulations/**/config.json"},
    "exp048-r001-analyse": {"**"},
    "exp049-r001-compute": {"simulations/**/config.json"},
    "exp076-r001-compute": {"commands.json"},
    "exp082-r001-compute": {
        "import.json",
        "archive/derived/artifacts/data/exp082/numbers.json",
    },
}


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def matches(relative: str, rules: set[str]) -> bool:
    if "**" in rules:
        return True
    for rule in rules:
        if "/**/" in rule:
            prefix, suffix = rule.split("/**/", 1)
            if relative.startswith(prefix + "/") and relative.endswith("/" + suffix):
                return True
        elif relative == rule:
            return True
    return False


def write_manifest(name: str, rows: list[tuple[str, int]]) -> None:
    lines = [f"{path}\t{size}" for path, size in sorted(rows)]
    (HERE / name).write_text("\n".join(lines) + ("\n" if lines else ""))


def main() -> None:
    downstream = []
    metadata_only = []
    unreferenced = []
    for evidence in sorted(RUNS.glob("*/export/evidence")):
        run = evidence.parent.parent
        record = json.loads((run / "run.json").read_text())
        references = {
            value.removeprefix("export/evidence/").rstrip("/")
            for value in strings(record)
            if value.startswith("export/evidence/")
        }
        for path in sorted(evidence.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(evidence).as_posix()
            row = (path.relative_to(REPO).as_posix(), path.stat().st_size)
            if matches(relative, DOWNSTREAM_RULES.get(run.name, set())):
                downstream.append(row)
            elif any(
                relative == reference or relative.startswith(reference + "/")
                for reference in references
            ):
                metadata_only.append(row)
            else:
                unreferenced.append(row)
    write_manifest("downstream-used.tsv", downstream)
    write_manifest("run-metadata-only.tsv", metadata_only)
    write_manifest("deletion-candidates.tsv", unreferenced)
    write_manifest("downstream-unused.tsv", metadata_only + unreferenced)
    summary = {
        "downstream_used": {
            "files": len(downstream),
            "bytes": sum(size for _, size in downstream),
        },
        "run_metadata_only": {
            "files": len(metadata_only),
            "bytes": sum(size for _, size in metadata_only),
        },
        "deletion_candidates": {
            "files": len(unreferenced),
            "bytes": sum(size for _, size in unreferenced),
        },
        "downstream_unused_total": {
            "files": len(metadata_only) + len(unreferenced),
            "bytes": sum(size for _, size in metadata_only + unreferenced),
        },
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
