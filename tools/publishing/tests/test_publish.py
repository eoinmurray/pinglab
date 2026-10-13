"""Publication policy and Git transport tests; no writing-content tests."""
import json
import subprocess

import pytest
from pingstore.tests.test_discovery import make_run

from tools.publishing import publish as publishing


def test_selection_uses_completion_time_instead_of_run_counter(tmp_path):
    store = tmp_path / '.pingstore/runs'
    make_run(store, 'exp001-r009-present', execution={'completed_at': '2026-08-27T10:00:00Z'})
    newer = make_run(store, 'exp001-r002-present', execution={'completed_at': '2026-08-28T10:00:00Z'})
    assert publishing.select(tmp_path)['exp001'] == newer.name


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
