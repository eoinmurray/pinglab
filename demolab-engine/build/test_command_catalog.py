"""AGENTS.md's command tables must match the actual runbook / guide files.

The trigger grammar — `HELP` lists them, a `NAME` in CAPS routes to one — only works if the tables
in AGENTS.md stay in step with `demolab-engine/runbooks/*.md` and `guides/*.md`. This asserts they
do, so a new runbook can't ship without a table row (and `HELP` can't advertise a file that's gone).
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENGINE = REPO / "demolab-engine"
AGENTS = (REPO / "AGENTS.md").read_text()


def _files(subdir: str) -> set[str]:
    return {p.stem for p in (ENGINE / subdir).glob("*.md")}


def _table(subdir: str) -> set[str]:
    # Names linked as `[`NAME`](demolab-engine/<subdir>/NAME.md)` in AGENTS.md.
    return set(re.findall(rf"{subdir}/([A-Z][A-Z0-9-]+)\.md", AGENTS))


def test_runbook_table_matches_files():
    files, table = _files("runbooks"), _table("runbooks")
    assert files == table, f"AGENTS runbook table vs files drifted:\n  files-only: {sorted(files - table)}\n  table-only: {sorted(table - files)}"


def test_guide_table_matches_files():
    files, table = _files("guides"), _table("guides")
    assert files == table, f"AGENTS guide table vs files drifted:\n  files-only: {sorted(files - table)}\n  table-only: {sorted(table - files)}"
