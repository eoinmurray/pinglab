"""Tests for overlay.py — the portable rsync replacement behind `task scaffold` and
`task add-demo-content`. Pure filesystem, so the Windows CI leg exercises it for real."""
from pathlib import Path

import overlay


def _tree(base: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


def test_overlay_copies_nested_tree(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _tree(src, {"a.txt": "A", "sub/deep/b.txt": "B"})
    dst.mkdir()
    assert overlay.overlay(src, dst) == 2
    assert (dst / "a.txt").read_text() == "A"
    assert (dst / "sub" / "deep" / "b.txt").read_text() == "B"


def test_keep_existing_never_clobbers(tmp_path):
    # A re-run of `task scaffold` must not overwrite user files (rsync --ignore-existing).
    src, dst = tmp_path / "src", tmp_path / "dst"
    _tree(src, {"a.txt": "scaffold"})
    _tree(dst, {"a.txt": "mine"})
    overlay.overlay(src, dst, keep_existing=True)
    assert (dst / "a.txt").read_text() == "mine"
    # Without the flag the overlay wins (`task add-demo-content` refreshes demo files).
    overlay.overlay(src, dst)
    assert (dst / "a.txt").read_text() == "scaffold"


def test_exclude_skips_top_level_dir(tmp_path):
    # The demo's prebuilt site/ must never land in the user's tree.
    src, dst = tmp_path / "src", tmp_path / "dst"
    _tree(src, {"keep.txt": "K", "site/index.html": "X"})
    dst.mkdir()
    overlay.overlay(src, dst, exclude=("site",))
    assert (dst / "keep.txt").exists()
    assert not (dst / "site").exists()
