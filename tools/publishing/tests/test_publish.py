"""Publication policy and Git transport tests; no writing-content tests."""
import json
import subprocess

import pytest
from pingstore.contracts import PingstoreError
from pingstore.presentation_inputs import projection
from pingstore.tests.test_discovery import make_run

from tools.publishing import publish as publishing


def test_selection_uses_completion_and_honours_pin(tmp_path):
    (tmp_path / 'writings').mkdir()
    (tmp_path / 'writings/exp001.typ').write_text('#let inputs = ("exp001",)')
    store = tmp_path / '.pingstore/runs'
    older = make_run(store, 'exp001-r009-present', execution={'completed_at': '2026-08-27T10:00:00Z'})
    newer = make_run(store, 'exp001-r002-present', execution={'completed_at': '2026-08-28T10:00:00Z'})
    assert publishing.select(tmp_path)['exp001']['exp001'] == newer.name
    (tmp_path / 'writings/run-defaults.json').write_text(json.dumps({'exp001': {'exp001': older.name}}))
    assert publishing.select(tmp_path)['exp001']['exp001'] == older.name


def test_selected_projection_checks_ancestry_not_unrelated_payload(tmp_path):
    (tmp_path / 'writings').mkdir()
    store = tmp_path / '.pingstore/runs'
    parent = make_run(store, 'exp001-r001-compute', stage='compute')
    reference = {'run_id': parent.name, 'payload_digest': json.loads((parent / 'run.json').read_text())['payload_digest']}
    child = make_run(store, 'exp001-r002-present', inputs={'source': reference})
    unrelated = make_run(store, 'exp002-r001-present')
    (unrelated / 'export/numbers.json').write_text('corrupt')
    assert projection(tmp_path, selected_ids={child.name})['runs'][0]['id'] == child.name
    (parent / 'export/numbers.json').write_text('corrupt')
    with pytest.raises(PingstoreError):
        projection(tmp_path, selected_ids={child.name})


def test_projection_rejects_compute_selection(tmp_path):
    (tmp_path / 'writings').mkdir()
    parent = make_run(tmp_path / '.pingstore/runs', 'exp001-r001-compute', stage='compute')
    with pytest.raises(PingstoreError, match='populated present'):
        projection(tmp_path, selected_ids={parent.name})


def test_publish_rejects_changed_output_before_git(tmp_path, monkeypatch):
    output = tmp_path / 'publication'
    site = output / 'site'
    site.mkdir(parents=True)
    (site / 'index.html').write_text('inspected')
    (output / 'build.json').write_text(json.dumps({'files': publishing.inventory(site)}))
    (site / 'index.html').write_text('changed')
    monkeypatch.setattr(publishing, 'OUTPUT', output)
    with pytest.raises(ValueError, match='changed since validation'):
        publishing.publish()


def test_publish_preserves_domain_previews_and_removes_stale_files(tmp_path, monkeypatch):
    def git(*args, cwd=tmp_path):
        return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()
    remote = tmp_path / 'remote.git'
    git('init', '--bare', str(remote))
    repo = tmp_path / 'source'
    git('init', '-b', 'gh-pages', str(repo))
    git('config', 'user.name', 'Publication Test', cwd=repo)
    git('config', 'user.email', 'publication@example.invalid', cwd=repo)
    (repo / 'CNAME').write_text('example.invalid')
    (repo / 'stale.html').write_text('old')
    (repo / 'pr-preview').mkdir()
    (repo / 'pr-preview/keep.html').write_text('preview')
    git('add', '.', cwd=repo)
    git('commit', '-m', 'Initial site', cwd=repo)
    git('remote', 'add', 'origin', str(remote), cwd=repo)
    git('push', 'origin', 'gh-pages', cwd=repo)
    output = tmp_path / 'publication'
    site = output / 'site'
    site.mkdir(parents=True)
    (site / 'index.html').write_text('new')
    (output / 'build.json').write_text(json.dumps({'built_at': 'test', 'files': publishing.inventory(site)}))
    monkeypatch.setattr(publishing, 'OUTPUT', output)
    monkeypatch.setattr(publishing, 'run', lambda *args, cwd=repo: subprocess.check_output(args, cwd=cwd, text=True).strip())
    publishing.publish()
    assert git('--git-dir', str(remote), 'show', 'gh-pages:CNAME') == 'example.invalid'
    assert git('--git-dir', str(remote), 'show', 'gh-pages:pr-preview/keep.html') == 'preview'
    files = git('--git-dir', str(remote), 'ls-tree', '-r', '--name-only', 'gh-pages').splitlines()
    assert 'index.html' in files and 'stale.html' not in files
