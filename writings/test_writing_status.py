"""Validate authored availability badges without requiring a local run store."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
from demolab_cli import _paths

ROOT = Path(__file__).resolve().parents[1]
ARTICLES = sorted((ROOT / "writings").glob("exp[0-9][0-9][0-9].typ"))
GUIDE = (ROOT / "writings/README.md").read_text()
GUIDE_VERSION = re.search(r"^Version: \*\*(\d+\.\d+\.\d+)\*\*$", GUIDE, re.MULTILINE)
assert GUIDE_VERSION is not None
STATUSES = {
    f"[≡ TXT | v{GUIDE_VERSION.group(1)}]",
    f"[▦ DATA | v{GUIDE_VERSION.group(1)}]",
}
KNOWN_GUIDE_VERSIONS = set(
    re.findall(r"^- \*\*(\d+\.\d+\.\d+)\*\*", GUIDE, re.MULTILINE)
)
STATUS_PATTERN = re.compile(r"^\[(?:≡ TXT|▦ DATA) \| v(\d+\.\d+\.\d+)\]$")


def test_status_vocabulary_matches_writing_guide():
    section = GUIDE.split("### 3.5. Local-data availability\n", 1)[1].split(
        "\n## 4.", 1
    )[0]
    labels = {
        line.split("`", 2)[1].replace(r"\|", "|")
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
    status = metadata.get("status", "")
    match = STATUS_PATTERN.fullmatch(status)
    assert match is not None, article
    assert match.group(1) in KNOWN_GUIDE_VERSIONS, article
