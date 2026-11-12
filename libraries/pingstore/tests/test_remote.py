"""Exercise real rclone call ordering with a filesystem-backed remote substitute."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from pingstore import remote
from pingstore.core import Store, write_json_atomic
from pingstore.remote import Remote


@pytest.fixture
def cloud(tmp_path, monkeypatch):
    root = tmp_path / "remote"
    root.mkdir()
    calls = []

    def path(value):
        return (
            root / str(value).split(":", 1)[1]
            if str(value).startswith("r2:")
            else Path(value)
        )

    def rclone(command, *args, capture=False):
        calls.append((command, tuple(map(str, args))))
        if command == "lsjson":
            p = path(args[0])
            return json.dumps(
                [
                    dict(
                        Name=f.name,
                        Size=f.stat().st_size,
                        ModTime=str(f.stat().st_mtime_ns),
                    )
                    for f in sorted(p.iterdir())
                    if f.is_file()
                ]
                if p.exists()
                else []
            )
        if command == "cat":
            return path(args[0]).read_text()
        if command == "copyto":
            target = path(args[1])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path(args[0]), target)
        elif command == "copy":
            shutil.copytree(path(args[0]), path(args[1]), dirs_exist_ok=True)
        elif command == "deletefile":
            path(args[0]).unlink()
        elif command == "purge":
            shutil.rmtree(path(args[0]))
        else:
            pytest.fail(f"unexpected rclone operation: {command}")
        return None

    monkeypatch.setattr(remote, "rclone", rclone)
    return root, calls, rclone


def source(tmp_path):
    store = Store(tmp_path / "local")
    with store.create("exp001", identity="one", stage="present") as run:
        (run.directory / "payload.bin").write_bytes(b"payload")
    return store


def test_backup_skips_completed_payloads_and_never_downloads(
    tmp_path, cloud, monkeypatch
):
    _, calls, _ = cloud
    store = source(tmp_path)
    backup = Remote("r2:archive")
    assert backup.backup(store)["uploaded"] == ["one"]
    transfer = next(i for i, c in enumerate(calls) if c[0] == "copy")
    receipt = next(
        i
        for i, c in enumerate(calls)
        if c[0] == "copyto" and c[1][1].endswith("receipts/one.json")
    )
    assert transfer < receipt
    calls.clear()
    monkeypatch.setattr(
        remote, "inventory", lambda *a: pytest.fail("unchanged payload scan")
    )
    assert backup.backup(store) == {"uploaded": [], "skipped": ["one"]}
    assert not any(c[0] in {"copy", "check"} for c in calls)
    assert not any("--download" in c[1] for c in calls)
    assert all("/runs/" not in c[1][0] for c in calls if c[0] == "cat")


def test_interrupted_backup_has_no_receipt_and_retries(tmp_path, cloud, monkeypatch):
    root, _, original = cloud
    store = source(tmp_path)

    def failing(command, *args, **kwargs):
        if command == "copy":
            raise subprocess.CalledProcessError(1, "rclone")
        return original(command, *args, **kwargs)

    monkeypatch.setattr(remote, "rclone", failing)
    with pytest.raises(subprocess.CalledProcessError):
        Remote("r2:archive").backup(store)
    assert not (root / "archive/receipts/one.json").exists()
    monkeypatch.setattr(remote, "rclone", original)
    assert Remote("r2:archive").backup(store)["uploaded"] == ["one"]


def test_restore_and_explicit_remote_deletion(tmp_path, cloud):
    root, _, _ = cloud
    original = source(tmp_path)
    backup = Remote("r2:archive")
    backup.backup(original)
    restored = Store(tmp_path / "restored")
    assert backup.restore(restored, ["one"]) == ["one"]
    assert (restored.get("one").directory / "payload.bin").read_bytes() == b"payload"
    backup.delete(restored, ["one"])
    assert original.get("one").record and restored.get("one").record
    assert not (root / "archive/runs/one").exists()


def test_missing_index_rebuilds_from_receipts_and_conflicting_id_fails(tmp_path, cloud):
    root, _, _ = cloud
    store = source(tmp_path)
    backup = Remote("r2:archive")
    backup.backup(store)
    (root / "archive/index.json").unlink()
    assert backup.backup(store)["skipped"] == ["one"]
    value = store.get("one").record
    value["description"] = "changed"
    write_json_atomic(store.get("one").directory / "run.json", value)
    with pytest.raises(ValueError, match="different metadata"):
        backup.backup(store)


@pytest.mark.skipif(shutil.which("rclone") is None, reason="rclone is not installed")
def test_real_rclone_local_transport_round_trip_and_unchanged_backup(
    tmp_path, monkeypatch
):
    """Exercise backend flags and initially absent prefixes without R2 access."""
    monkeypatch.setenv("RCLONE_CONFIG", str(tmp_path / "empty-rclone.conf"))
    store = source(tmp_path)
    backup = Remote(":local:" + str(tmp_path / "real-backup"))
    assert backup.backup(store) == {"uploaded": ["one"], "skipped": []}
    assert backup.backup(store) == {"uploaded": [], "skipped": ["one"]}
    restored = Store(tmp_path / "real-restore")
    assert backup.restore(restored, ["one"]) == ["one"]
    assert (restored.get("one").directory / "payload.bin").read_bytes() == b"payload"


def test_interrupted_delete_retries_without_a_completion_receipt(
    tmp_path, cloud, monkeypatch
):
    root, _, original = cloud
    store = source(tmp_path)
    backup = Remote("r2:archive")
    backup.backup(store)

    def failed(command, *args, **kwargs):
        if command == "purge":
            raise subprocess.CalledProcessError(1, "rclone")
        return original(command, *args, **kwargs)

    monkeypatch.setattr(remote, "rclone", failed)
    with pytest.raises(subprocess.CalledProcessError):
        backup.delete(store, ["one"])
    assert not (root / "archive/receipts/one.json").exists()
    assert (root / "archive/runs/one").exists()

    # The test transport models deletefile's documented not-found exit code.
    def retry(command, *args, **kwargs):
        if (
            command == "deletefile"
            and not (root / "archive/receipts/one.json").exists()
        ):
            return None
        return original(command, *args, **kwargs)

    monkeypatch.setattr(remote, "rclone", retry)
    assert backup.delete(store, ["one"]) == ["one"]
    assert not (root / "archive/runs/one").exists()
