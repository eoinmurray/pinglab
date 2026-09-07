"""Build a frozen local publication and publish only its verified static output."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT)]
from pingstore.contracts import (  # noqa: E402
    load_json,
    validate_operational_run_directory,
)
from pingstore.presentation_inputs import article_inputs, projection  # noqa: E402

from writings.demolab_pingstore import declared_dependencies  # noqa: E402

OUTPUT = ROOT / '.demolab/publication'


def run(*args, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def inventory(directory):
    result = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Symlink in publication: {path}')
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = {
                'bytes': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    return result


def select(root):
    articles = article_inputs(root)
    pins_path = root / 'writings/run-defaults.json'
    pins = load_json(pins_path) if pins_path.exists() else {}
    latest = {}
    for directory in sorted((root / '.pingstore/runs').iterdir()):
        if directory.name.startswith('.') or directory.is_symlink() or not directory.is_dir():
            continue
        record = load_json(directory / 'run.json')
        if record.get('schema') != 'pingstore.run/v4':
            raise ValueError(f'Non-operational run in store: {directory.name}')
        if record['stage'] != 'present':
            continue
        timestamp = datetime.fromisoformat(record['execution']['completed_at'].replace('Z', '+00:00'))
        if timestamp.utcoffset() is None:
            raise ValueError('Completion timestamp must include a timezone')
        key = (timestamp, record['run_id'])
        experiment = record['experiment']
        if experiment not in latest or key > latest[experiment]:
            latest[experiment] = key
    choices = {}
    for article, keys in articles.items():
        choices[article] = {}
        for key in keys:
            identity = pins.get(article, {}).get(key)
            if identity is None:
                candidate = latest.get(key.split('.')[-1])
                if candidate is None:
                    raise ValueError(f'Missing required presentation: {article}/{key}')
                identity = candidate[1]
            choices[article][key] = identity
    return choices


def build():
    choices = select(ROOT)
    identities = {identity for pins in choices.values() for identity in pins.values()}
    print(f'Validating {len(identities)} selected presentations and their ancestry…', flush=True)
    data = projection(ROOT, selected_ids=identities, declared_dependencies=declared_dependencies())
    by_id = {row['id']: row for row in data['runs']}
    for article, pins in choices.items():
        for key, identity in pins.items():
            if by_id[identity]['experiment'] != key.split('.')[-1]:
                raise ValueError(f'Wrong experiment pin: {article}/{key}')
    data['defaults'] = choices
    OUTPUT.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='publication-', dir=OUTPUT.parent) as temporary:
        staging = Path(temporary)
        workspace = staging / 'workspace'
        workspace.mkdir()
        # Include the current authored working tree, including new unignored files.
        names = subprocess.check_output(
            ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT,
        ).decode().split('\0')
        for name in set(names) - {''}:
            source = ROOT / name
            if source.is_symlink():
                raise ValueError(f'Source symlink requires explicit handling: {name}')
            if source.is_file():
                target = workspace / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        for identity in identities:
            source = ROOT / '.pingstore/runs' / identity
            target = workspace / '.pingstore/runs' / identity
            shutil.copytree(source, target)
            # Revalidate copied bytes to catch modification during the copy.
            validate_operational_run_directory(target)
        prepared = workspace / '.demolab/pinglab-inputs.json'
        prepared.parent.mkdir()
        prepared.write_text(json.dumps(data))
        config = workspace / 'demolab.yaml'
        command = [sys.executable, '-c', 'from pathlib import Path; assert Path(".demolab/pinglab-inputs.json").is_file()']
        config.write_text(re.sub(r'^prepare:.*$', 'prepare: ' + json.dumps(command), config.read_text(), flags=re.M))
        subprocess.run([str(ROOT / '.venv/bin/demolab'), 'build'], cwd=workspace, check=True)
        site = workspace / '.demolab/site'
        for article, pins in choices.items():
            html = site / (article + '.html')
            if not html.is_file():
                raise ValueError(f'Missing article output: {article}')
            if pins and 'A required run is unavailable' in html.read_text():
                raise ValueError(f'Unpopulated article: {article}')
        # Browsers must not download large videos just to open an article.
        for html in site.glob('*.html'):
            html.write_text(re.sub(r'<video\b(?![^>]*\bpreload=)', '<video preload="none"', html.read_text()))
        (site / '.nojekyll').touch()
        receipt = {
            'built_at': datetime.now(timezone.utc).isoformat(),
            'source_commit': run('git', 'rev-parse', 'HEAD'),
            'working_tree_status': run('git', 'status', '--short'),
            'selections': choices,
            'files': inventory(site),
        }
        # Promote only a completed build; a failed build leaves the previous one intact.
        candidate = staging / 'result'
        candidate.mkdir()
        shutil.move(site, candidate / 'site')
        (candidate / 'build.json').write_text(json.dumps(receipt, indent=2) + '\n')
        if OUTPUT.exists():
            shutil.rmtree(OUTPUT)
        shutil.move(candidate, OUTPUT)
    print(f'Built {len(receipt["files"])} files, {sum(v["bytes"] for v in receipt["files"].values()) / 1e6:.1f} MB')
    print(f'Preview: task preview-publication\nOutput: {OUTPUT / "site"}')


def publish():
    receipt = load_json(OUTPUT / 'build.json')
    site = OUTPUT / 'site'
    if inventory(site) != receipt['files']:
        raise ValueError('Built site changed since validation; build again before publishing')
    remote = run('git', 'remote', 'get-url', 'origin')
    with tempfile.TemporaryDirectory(prefix='pinglab-publish-') as temporary:
        checkout = Path(temporary) / 'pages'
        subprocess.run(['git', 'clone', '--depth=1', '--single-branch', '--branch=gh-pages', remote, str(checkout)], check=True)
        for child in checkout.iterdir():
            if child.name in {'.git', 'CNAME', 'pr-preview'}:
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        for child in site.iterdir():
            if child.name in {'.git', 'CNAME', 'pr-preview'}:
                raise ValueError(f'Reserved publication path: {child.name}')
            target = checkout / child.name
            if child.is_dir():
                shutil.copytree(child, target)
            else:
                shutil.copy2(child, target)
        for name, expected in receipt['files'].items():
            copied = checkout / name
            if (copied.stat().st_size != expected['bytes']
                    or hashlib.sha256(copied.read_bytes()).hexdigest() != expected['sha256']):
                raise ValueError('Site changed while copying; rebuild before publishing')
        subprocess.run(['git', 'add', '--all'], cwd=checkout, check=True)
        if not run('git', 'status', '--porcelain', cwd=checkout):
            print('Published site already matches this build.')
            return
        for key in ('user.name', 'user.email'):
            subprocess.run(['git', 'config', key, run('git', 'config', key)], cwd=checkout, check=True)
        subprocess.run(['git', 'commit', '-m', f'Publish site built {receipt["built_at"]}'], cwd=checkout, check=True)
        # A normal push rejects concurrent updates; never overwrite another publisher.
        subprocess.run(['git', 'push', 'origin', 'HEAD:gh-pages'], cwd=checkout, check=True)
    print('Published static site to gh-pages.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build', 'publish'])
    args = parser.parse_args()
    try:
        {'build': build, 'publish': publish}[args.command]()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'Publication failed: {exc}\n')


if __name__ == '__main__':
    main()
