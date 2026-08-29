from __future__ import annotations

import io
import tarfile

import pytest
from experiments.helpers import modal_backend
from experiments.helpers.cli import parse_meta


def test_modal_meta_flag_parses_for_dispatch_runner():
    meta = parse_meta(
        ["exp999.py", "--modal", "--live", "--only-cells", "ping"],
        allow_dispatch=True,
    )
    assert meta.modal is True
    assert meta.live is True
    assert meta.only_cells == ["ping"]


def test_scheduler_cell_meta_flags_parse_for_dispatch_runner():
    meta = parse_meta(
        ["compute.py", "--train-cell", "ping__off__seed42"],
        allow_dispatch=True,
    )
    assert meta.train_cell == "ping__off__seed42"

    listing = parse_meta(
        ["compute.py", "--list-cells", "variable_rate"],
        allow_dispatch=True,
    )
    assert listing.list_cells == "variable_rate"


def test_generic_modal_dry_run_does_not_import_modal(tmp_path, capsys):
    modal_backend.dispatch(
        slug="exp999",
        runner="exp999",
        job_ids=["first", "second"],
        live=False,
        local_collect_dir=tmp_path / "scratch",
        ledger_path=tmp_path / "ledger.json",
        timeout_s=60,
        extra_env={"EXP999_STAGE": "short"},
    )
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert "backend=modal" in out
    assert "runner=exp999" in out
    assert "jobs: first second" in out
    assert not (tmp_path / "ledger.json").exists()


def test_generic_modal_dispatch_validates_timeout(tmp_path):
    with pytest.raises(ValueError, match="timeout_s"):
        modal_backend.dispatch(
            slug="exp999",
            runner="exp999",
            job_ids=["first"],
            live=False,
            local_collect_dir=tmp_path / "scratch",
            ledger_path=tmp_path / "ledger.json",
            timeout_s=modal_backend.MAX_RUNTIME_S + 1,
        )


def test_generic_modal_dispatch_rejects_duplicate_jobs(tmp_path):
    with pytest.raises(ValueError, match="non-empty and unique"):
        modal_backend.dispatch(
            slug="exp999",
            runner="exp999",
            job_ids=["same", "same"],
            live=False,
            local_collect_dir=tmp_path / "scratch",
            ledger_path=tmp_path / "ledger.json",
            timeout_s=60,
        )


def test_modal_artifact_extract_rejects_path_traversal(tmp_path):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        info = tarfile.TarInfo("../escape.txt")
        content = b"nope"
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    with pytest.raises(RuntimeError, match="unsafe Modal artifact path"):
        modal_backend._extract_tree(payload.getvalue(), tmp_path / "dest")


def test_modal_auth_error_is_recognized_by_name_and_module():
    AuthError = type("AuthError", (Exception,), {"__module__": "modal.exception"})
    assert modal_backend._is_modal_auth_error(AuthError("missing token"))
    assert not modal_backend._is_modal_auth_error(RuntimeError("missing token"))
