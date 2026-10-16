"""Build a frozen local publication and publish only its verified static output."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / '.publication'


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


def load_json(path):
    return json.loads(path.read_text())


def select(root):
    """Freeze the simple latest-presentation lookup for this Astro build."""
    latest = {}
    for directory in sorted((root / '.pingstore/runs').iterdir()):
        if directory.name.startswith('.') or not directory.is_dir():
            continue
        record = load_json(directory / 'run.json')
        if record.get('stage') != 'present':
            continue
        key = (record['execution']['completed_at'], record['run_id'])
        experiment = record['experiment']
        if experiment not in latest or key > latest[experiment]:
            latest[experiment] = key
    return {experiment: value[1] for experiment, value in latest.items()}


def source_inventory(root):
    """Detect authored-source changes while Astro is building."""
    files = []
    bases = [root / 'site', root / 'md/notes', root / 'md/slides']
    for experiment in sorted((root / 'experiments').glob('exp[0-9][0-9][0-9]')):
        bases.extend(experiment / name for name in ('report', 'js'))
    for base in bases:
        for path in base.rglob('*'):
            relative = path.relative_to(base)
            if any(part in {'node_modules', '.astro', 'dist', '.git', '__pycache__'} for part in relative.parts):
                continue
            if path.is_file():
                files.append(path)
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}


def asset_inventory(path, site):
    """Collect frozen asset choices and verify every bundled fallback."""
    assets = {}
    if not path.exists():
        return assets
    for line in path.read_text().splitlines():
        record = json.loads(line)
        key = f'{record["run"]}/{record["file"]}'
        if key in assets and assets[key] != record:
            raise ValueError(f'Asset changed during build: {key}')
        assets[key] = record
        if record['source'] == 'local':
            bundled = site / 'run-images' / record['run'] / record['file']
            if (bundled.stat().st_size != record['bytes']
                    or hashlib.sha256(bundled.read_bytes()).hexdigest() != record['sha256']):
                raise ValueError(f'Bundled asset changed during build: {key}')
    return assets


def build():
    choices = select(ROOT)
    sources = source_inventory(ROOT)
    OUTPUT.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='astro-publication-', dir=OUTPUT.parent) as temporary:
        staging = Path(temporary)
        selections = staging / 'selections.json'
        selections.write_text(json.dumps(choices))
        site = staging / 'site'
        asset_receipt = staging / 'assets.jsonl'
        environment = dict(os.environ, PINGLAB_RUN_SELECTIONS=str(selections),
                           PINGLAB_ASSET_RECEIPT=str(asset_receipt))
        subprocess.run(['bun', 'run', 'astro', 'build', '--outDir', str(site)],
                       cwd=ROOT / 'site', env=environment, check=True)
        if source_inventory(ROOT) != sources:
            raise ValueError('Authored sources changed during the build; build again')
        if not (site / 'index.html').is_file():
            raise ValueError('Astro did not produce a homepage')
        for html in site.rglob('*.html'):
            html.write_text(re.sub(r'<video\b[^>]*>',
                lambda match: re.sub(r'\s+preload="[^"]*"', '', match[0])[:-1] + ' preload="none">', html.read_text()))
        (site / '.nojekyll').touch()
        receipt = {
            'built_at': datetime.now(timezone.utc).isoformat(),
            'source_commit': run('git', 'rev-parse', 'HEAD'),
            'working_tree_status': run('git', 'status', '--short'),
            'latest_selections': choices,
            'assets': asset_inventory(asset_receipt, site),
            'source_files': sources,
            'files': inventory(site),
        }
        candidate = staging / 'result'
        candidate.mkdir()
        shutil.move(site, candidate / 'site')
        (candidate / 'build.json').write_text(json.dumps(receipt, indent=2) + '\n')
        previous = OUTPUT.with_name('.publication-previous')
        if previous.exists():
            shutil.rmtree(previous)
        if OUTPUT.exists():
            OUTPUT.rename(previous)
        try:
            candidate.rename(OUTPUT)
        except OSError:
            if previous.exists():
                previous.rename(OUTPUT)
            raise
        if previous.exists():
            shutil.rmtree(previous)
    print(f'Built {len(receipt["files"])} files, {sum(v["bytes"] for v in receipt["files"].values()) / 1e6:.1f} MB')
    remote_assets = sum(asset['source'] == 'r2' for asset in receipt['assets'].values())
    print(f'Run assets: {remote_assets} R2, {len(receipt["assets"]) - remote_assets} bundled locally')
    print('Preview: (cd site && bun run astro preview --outDir ../.publication/site --port 3001)')



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
