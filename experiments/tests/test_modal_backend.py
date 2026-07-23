from __future__ import annotations

import io
import tarfile

import pytest
from experiments.helpers import modal_backend
from experiments.helpers.cli import parse_meta


def test_modal_meta_flag_parses_for_dispatch_runner():
    meta = parse_meta(
        ["exp073.py", "--modal", "--live", "--only-cells", "ping"],
        allow_dispatch=True,
    )
    assert meta.modal is True
    assert meta.live is True
    assert meta.only_cells == ["ping"]


def test_exp073_modal_dry_run_does_not_import_modal(tmp_path, capsys):
    modal_backend.dispatch_exp073(
        cells=["ping"],
        attempt="plastic_wee",
        stage="short",
        ping_only=True,
        live=False,
        local_collect_dir=tmp_path / "scratch",
        ledger_path=tmp_path / "ledger.json",
        timeout_s=60,
    )
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert "backend=modal" in out
    assert not (tmp_path / "ledger.json").exists()


def test_modal_artifact_extract_rejects_path_traversal(tmp_path):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        info = tarfile.TarInfo("../escape.txt")
        content = b"nope"
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    with pytest.raises(RuntimeError, match="unsafe Modal artifact path"):
        modal_backend._extract_tree(payload.getvalue(), tmp_path / "dest")
