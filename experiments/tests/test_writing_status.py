"""Validate authored availability badges without requiring a local run store."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from demolab_cli import _paths

ROOT = Path(__file__).resolve().parents[2]
ARTICLES = sorted((ROOT / "writings").glob("exp[0-9][0-9][0-9].typ"))
STATUSES = {
    "[≡ TXT]",
    "[▦ DATA]",
}


def test_status_vocabulary_matches_writing_guide():
    guide = (ROOT / "writings/README.md").read_text()
    section = guide.split("### 3.5. Local-data availability\n", 1)[1].split(
        "\n## 4.", 1
    )[0]
    labels = {
        line.split("`", 2)[1]
        for line in section.splitlines()
        if line.startswith("| `")
    }
    assert labels == STATUSES
    assert ARTICLES, "No experiment articles found"


@pytest.mark.parametrize("article", ARTICLES, ids=lambda path: path.stem)
def test_article_declares_supported_status(article):
    expression = '{ import "/writings/' + article.name + '": meta; meta }'
    result = subprocess.run(
        [_paths.find_typst(ROOT), "eval", "--root", str(ROOT), expression],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Some Typst eval versions report errors on stderr with a zero exit status.
    assert result.returncode == 0 and result.stdout.strip(), result.stderr
    metadata = json.loads(result.stdout)
    assert metadata.get("status") in STATUSES, article
