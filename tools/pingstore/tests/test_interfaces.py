"""Check the three-file package, shared implementation and command entry points."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pingstore import cli, core, library


def test_package_has_only_the_three_requested_source_modules():
    package = Path(core.__file__).parent
    assert {path.name for path in package.glob("*.py")} == {
        "core.py",
        "cli.py",
        "library.py",
    }
    tree = ast.parse(Path(core.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not node.level
            assert node.module not in {"pingstore.cli", "pingstore.library"}
        elif isinstance(node, ast.Import):
            assert not any(
                alias.name in {"pingstore.cli", "pingstore.library"}
                for alias in node.names
            )


def test_library_exports_the_shared_core_objects():
    assert len(library.__all__) == len(set(library.__all__))
    for name in library.__all__:
        assert getattr(library, name) is getattr(core, name)
    assert "apply_prune" not in library.__all__
    assert "apply_remote_prune" not in library.__all__


@pytest.mark.parametrize(
    ("arguments", "operation", "result"),
    [
        (["discover"], "discover_runs", []),
        (["backup", "--dry-run"], "backup_runs", 0),
        (["prune", "--all", "--dry-run"], "unified_prune_plan", {}),
        (["evict", "--all", "--dry-run"], "eviction_plan", {}),
    ],
)
def test_cli_dispatches_to_core(arguments, operation, result, tmp_path, monkeypatch):
    (tmp_path / ".pingstore/runs").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(core, operation, run)
    for name in (
        "render_unified_prune_plan",
        "render_eviction_plan",
    ):
        monkeypatch.setattr(core, name, lambda plan: json.dumps(plan))
    assert cli.main(arguments) == 0
    assert len(calls) == 1


def test_module_entrypoint_works_outside_the_repository(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "pingstore.cli", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "prune-remote" not in completed.stdout
    assert "backup-papers" not in completed.stdout
    assert "status" in completed.stdout
    assert "discover" in completed.stdout
