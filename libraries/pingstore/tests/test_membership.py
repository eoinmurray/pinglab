from __future__ import annotations

from pathlib import Path

import pytest
from pingstore import core
from pingstore.library import PingstoreError, membership, memberships


def _report(repo: Path, experiment: str, collection: str) -> None:
    path = repo / "experiments" / experiment / "report" / "report.mdx"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nid: {experiment}\ncollection: {collection}\n---\n")


def test_memberships_come_from_report_metadata(tmp_path: Path) -> None:
    _report(tmp_path, "exp001", "demo")
    _report(tmp_path, "exp002", "gamma-gated-sparsity")
    assert memberships(tmp_path) == {
        "exp001": "demo",
        "exp002": "gamma-gated-sparsity",
    }
    assert membership(tmp_path, "exp002") == "gamma-gated-sparsity"


def test_missing_active_membership_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(PingstoreError, match="must declare collection metadata"):
        membership(tmp_path, "exp001")


def test_migrated_note_preserves_presentation_ownership_and_inputs(
    tmp_path: Path,
) -> None:
    note = tmp_path / "notes" / "manuscript"
    note.mkdir(parents=True)
    (note / "index.mdx").write_text(
        "---\ncollection: [gamma-gated-sparsity, notes]\n"
        "presentation: exp110\ninputs: [exp110, exp025]\n---\n"
    )
    (note / "appendix.mdx").write_text("## Appendix A\n")
    (note / "references.mdx").write_text("1. Reference.\n")
    figures = note / "figures"
    figures.mkdir()
    (figures / "README.md").write_text("Diagram instructions.\n")
    assert core.articles(tmp_path)["notes/manuscript"]["inputs"] == ["exp110", "exp025"]
    assert memberships(tmp_path) == {"exp110": "gamma-gated-sparsity"}


def test_note_cannot_duplicate_experiment_presentation_ownership(
    tmp_path: Path,
) -> None:
    _report(tmp_path, "exp001", "demo")
    note = tmp_path / "notes" / "article.mdx"
    note.parent.mkdir()
    note.write_text(
        "---\ncollection: notes\npresentation: exp001\ninputs: [exp001]\n---\n"
    )
    with pytest.raises(PingstoreError, match="duplicate presentation owner"):
        memberships(tmp_path)


@pytest.mark.parametrize("presentation", ["exp110", "article"])
def test_note_presentation_must_be_a_declared_experiment_input(
    tmp_path: Path, presentation: str
) -> None:
    note = tmp_path / "notes" / "article.mdx"
    note.parent.mkdir()
    note.write_text(
        f"---\ncollection: notes\npresentation: {presentation}\ninputs: []\n---\n"
    )
    with pytest.raises(
        PingstoreError, match="note presentation must name a declared input"
    ):
        core.articles(tmp_path)
