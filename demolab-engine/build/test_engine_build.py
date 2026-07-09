"""Engine smoke test — builds the shipped scaffold fixtures end-to-end.

The demo under `demolab-engine/scaffold/demo/` doubles as the engine's integration test:
we assemble a throwaway repo (skeleton + demo) in a tmp dir, point `build.py` at it via
`DEMOLAB_ROOT`, and assert the compiler emits a real site. The empty case (skeleton only)
proves a freshly-scaffolded repo builds a friendly empty-state homepage rather than erroring.

Needs the `typst` CLI on PATH (same as `task build`); skipped if it's missing.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ENGINE = REPO / "demolab-engine"
SCAFFOLD = ENGINE / "scaffold"

pytestmark = pytest.mark.skipif(
    shutil.which("typst") is None, reason="typst CLI not installed"
)


def _assemble(root: Path, *, demo: bool) -> None:
    """Lay down a full working tree at `root`: engine build/ + skeleton (+ demo overlay).

    The engine's build/ is copied (not symlinked) into the fixture: writings import
    `/demolab-engine/build/lib.typ` root-relative, and typst requires the source to live
    under --root, so a symlink pointing outside the tmp dir is rejected."""
    shutil.copytree(ENGINE / "build", root / "demolab-engine" / "build")
    shutil.copy(ENGINE / "VERSION", root / "demolab-engine" / "VERSION")  # read by web-styles' generator meta
    shutil.copytree(SCAFFOLD / "skeleton", root, dirs_exist_ok=True)
    if demo:
        shutil.copytree(SCAFFOLD / "demo", root, dirs_exist_ok=True)


def _build(root: Path) -> None:
    subprocess.run(
        [sys.executable, str(ENGINE / "build" / "build.py")],
        env={**os.environ, "DEMOLAB_ROOT": str(root)},
        check=True,
    )


def test_demo_fixture_builds_full_site(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _assemble(root, demo=True)
    _build(root)

    site = root / "artifacts" / "site"
    assert (site / "index.html").exists()
    assert (site / "all.html").exists()
    assert (site / "pdfs" / "book.pdf").exists()
    assert (site / "exp000.html").exists(), "per-entry page emitted"
    assert (site / "pdfs" / "exp000.pdf").exists(), "per-entry PDF emitted"

    index = (site / "index.html").read_text()
    # ".empty-state" also appears in the inlined stylesheet, so key on the rendered copy.
    assert "Your lab is ready" not in index, "demo has content, so no empty state"
    assert "Installation" in index, "demo welcome block renders on the homepage"
    assert "Open source, MIT licensed" in index, "demo welcome footer renders"
    assert '<ul class="coll-list"' not in index, "demo landing hides the collection directory"
    assert '<p class="page-foot"' not in index, "demo landing hides the page foot"
    entry = (site / "exp000.html").read_text()
    assert "<img" in entry or "<svg" in entry, "a figure made it into the entry page"


def test_emitted_html_has_no_root_absolute_urls(tmp_path: Path) -> None:
    """Every emitted href/src must be relative — no leading-slash root-absolute URLs.

    This is a hosting contract, not cosmetics: the PR-preview deploy (demolab-engine/deploy/
    preview.yml) serves each site from a subpath (pr-preview/pr-N/), so a root-absolute
    `href="/foo.html"` or `src="/artifacts/…"` would resolve against the domain root and 404 in
    every preview. lib.typ emits only relative links today; this test stops a future engine
    change from silently breaking previews. (Protocol-relative `//host` and absolute
    `https://…` are fine; a lone leading `/` is the violation.)"""
    import re

    root = tmp_path / "repo"
    root.mkdir()
    _assemble(root, demo=True)
    _build(root)

    site = root / "artifacts" / "site"
    bad = re.compile(r'(?:href|src)="/(?!/)')  # a single leading slash, not // (protocol-relative)
    offenders = []
    for html in site.rglob("*.html"):
        for i, line in enumerate(html.read_text().splitlines(), 1):
            if bad.search(line):
                offenders.append(f"{html.relative_to(site)}:{i}")
    assert not offenders, (
        "root-absolute URLs break subpath-served PR previews — make them relative:\n  "
        + "\n  ".join(offenders)
    )


def test_broken_entry_is_stubbed_not_fatal(tmp_path: Path) -> None:
    """One entry that references a missing figure must not take down the whole site: it's flagged,
    rendered as a stub at its own URL, and dropped from the listings, while everything else builds."""
    root = tmp_path / "repo"
    root.mkdir()
    _assemble(root, demo=True)
    (root / "writings" / "exp099.typ").write_text(
        '#let meta = (title: "Broken on purpose", date: "2026-07-06", collection: "neuron-models")\n'
        '#let body = [#image("/artifacts/data/exp099/missing.svg")]\n'
    )
    _build(root)  # check=True — the build must still succeed

    site = root / "artifacts" / "site"
    assert (site / "exp000.html").exists(), "good entries still built"
    assert (site / "index.html").exists(), "homepage still built"
    stub = site / "exp099.html"
    assert stub.exists(), "the broken entry got a stub page at its URL"
    assert "failed to build" in stub.read_text(), "the stub explains what happened"
    assert "exp099.html" not in (site / "index.html").read_text(), "the stub stays out of the listings"


def test_empty_tree_builds_empty_state(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _assemble(root, demo=False)
    _build(root)

    site = root / "artifacts" / "site"
    assert (site / "index.html").exists(), "homepage always exists"
    assert not (site / "all.html").exists(), "no all-entries page without content"
    assert not (site / "pdfs" / "book.pdf").exists(), "no book without content"
    assert "Your lab is ready" in (site / "index.html").read_text()
