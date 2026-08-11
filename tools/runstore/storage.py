"""Filesystem and rclone storage backends for runstore archives."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .contract import ContractError


@dataclass(frozen=True)
class StoredObject:
    path: str
    size_bytes: int


class Store(Protocol):
    def exists(self, archive_id: str) -> bool: ...

    def put_archive(
        self,
        archive_id: str,
        source: Path,
        run_bytes: bytes,
        inventory_bytes: bytes,
    ) -> None: ...

    def read_bytes(self, archive_id: str, relative: str) -> bytes: ...

    def objects(self, archive_id: str) -> list[StoredObject]: ...

    def sha256(self, archive_id: str, relative: str) -> str: ...

    def restore(self, archive_id: str, destination: Path) -> None: ...

    def logical_uri(self, archive_id: str) -> str: ...


class LocalStore:
    """Filesystem-backed store used by tests and local rehearsals."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def _archive(self, archive_id: str) -> Path:
        return self.root / archive_id

    def exists(self, archive_id: str) -> bool:
        archive = self._archive(archive_id)
        return archive.exists() and any(archive.iterdir())

    def put_archive(
        self,
        archive_id: str,
        source: Path,
        run_bytes: bytes,
        inventory_bytes: bytes,
    ) -> None:
        destination = self._archive(archive_id)
        if destination.exists():
            raise ContractError(f"archive destination already exists: {destination}")
        destination.mkdir(parents=True)
        try:
            for item in sorted(source.rglob("*")):
                if not item.is_file() or item.name in {"run.json", "inventory.json"}:
                    continue
                relative = item.relative_to(source)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(item, target)
            (destination / "run.json").write_bytes(run_bytes)
            (destination / "inventory.json").write_bytes(inventory_bytes)
        except Exception:
            # Preserve partial output for diagnosis; the immutable identity cannot
            # be reused until the operator deliberately removes it.
            raise

    def read_bytes(self, archive_id: str, relative: str) -> bytes:
        return (self._archive(archive_id) / relative).read_bytes()

    def objects(self, archive_id: str) -> list[StoredObject]:
        root = self._archive(archive_id)
        if not root.is_dir():
            raise ContractError(f"archive does not exist: {root}")
        return [
            StoredObject(item.relative_to(root).as_posix(), item.stat().st_size)
            for item in sorted(root.rglob("*"))
            if item.is_file()
        ]

    def sha256(self, archive_id: str, relative: str) -> str:
        digest = hashlib.sha256()
        with (self._archive(archive_id) / relative).open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def restore(self, archive_id: str, destination: Path) -> None:
        shutil.copytree(self._archive(archive_id), destination)

    def logical_uri(self, archive_id: str) -> str:
        return f"file://{self._archive(archive_id)}"


class RcloneStore:
    """Rclone-backed object store. Verification streams remote bytes."""

    def __init__(self, root: str, *, logical_base_uri: str):
        if ":" not in root:
            raise ContractError(
                "rclone store root must include a remote name and colon"
            )
        self.root = root.rstrip("/")
        self.logical_base_uri = logical_base_uri.rstrip("/")
        self._run(["version"], capture=True)

    def _remote(self, archive_id: str, relative: str | None = None) -> str:
        result = f"{self.root}/{archive_id}"
        return f"{result}/{relative}" if relative else result

    @staticmethod
    def _run(args: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
        try:
            result = subprocess.run(
                ["rclone", *args],
                check=False,
                capture_output=capture,
                text=capture,
            )
        except FileNotFoundError as exc:
            raise ContractError("rclone is not installed or not on PATH") from exc
        if result.returncode != 0:
            detail = result.stderr.strip() if capture and result.stderr else ""
            raise ContractError(
                f"rclone {' '.join(args)} failed ({result.returncode})"
                + (f": {detail}" if detail else "")
            )
        return result

    def exists(self, archive_id: str) -> bool:
        result = subprocess.run(
            ["rclone", "lsf", self._remote(archive_id), "--max-depth", "1"],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and bool(result.stdout.strip())

    def put_archive(
        self,
        archive_id: str,
        source: Path,
        run_bytes: bytes,
        inventory_bytes: bytes,
    ) -> None:
        remote = self._remote(archive_id)
        self._run(
            [
                "copy",
                str(source),
                remote,
                "--exclude",
                "/run.json",
                "--exclude",
                "/inventory.json",
            ]
        )
        self._put_bytes(run_bytes, self._remote(archive_id, "run.json"))
        self._put_bytes(inventory_bytes, self._remote(archive_id, "inventory.json"))

    def _put_bytes(self, content: bytes, remote: str) -> None:
        process = subprocess.Popen(
            ["rclone", "rcat", remote],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        _, stderr = process.communicate(content)
        if process.returncode != 0:
            raise ContractError(
                f"rclone rcat {remote} failed ({process.returncode}): "
                f"{stderr.decode(errors='replace').strip()}"
            )

    def read_bytes(self, archive_id: str, relative: str) -> bytes:
        result = subprocess.run(
            ["rclone", "cat", self._remote(archive_id, relative)],
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            raise ContractError(
                f"could not read {self._remote(archive_id, relative)}: "
                f"{result.stderr.decode(errors='replace').strip()}"
            )
        return result.stdout

    def objects(self, archive_id: str) -> list[StoredObject]:
        result = self._run(
            ["lsjson", self._remote(archive_id), "--recursive", "--files-only"],
            capture=True,
        )
        try:
            rows = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ContractError(
                "rclone returned invalid JSON while listing archive"
            ) from exc
        return sorted(
            [StoredObject(row["Path"], int(row["Size"])) for row in rows],
            key=lambda item: item.path,
        )

    def sha256(self, archive_id: str, relative: str) -> str:
        process = subprocess.Popen(
            ["rclone", "cat", self._remote(archive_id, relative)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None
        digest = hashlib.sha256()
        while chunk := process.stdout.read(1024 * 1024):
            digest.update(chunk)
        _, stderr = process.communicate()
        if process.returncode != 0:
            raise ContractError(
                f"could not hash {self._remote(archive_id, relative)}: "
                f"{stderr.decode(errors='replace').strip()}"
            )
        return digest.hexdigest()

    def restore(self, archive_id: str, destination: Path) -> None:
        self._run(["copy", self._remote(archive_id), str(destination)])

    def logical_uri(self, archive_id: str) -> str:
        return f"{self.logical_base_uri}/{archive_id}"


def build_store(spec: str, *, logical_base_uri: str) -> Store:
    if spec.startswith("file://"):
        return LocalStore(Path(spec.removeprefix("file://")))
    if ":" in spec and not os.path.isabs(spec):
        return RcloneStore(spec, logical_base_uri=logical_base_uri)
    return LocalStore(Path(spec))
