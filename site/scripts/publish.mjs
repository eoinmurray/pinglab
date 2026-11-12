import { spawnSync } from 'node:child_process';
import { cp, mkdir, mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';

const started = performance.now();
const root = fileURLToPath(new URL('../../', import.meta.url));
function run(command, args, cwd = root, capture = false) {
  const result = spawnSync(command, args, { cwd, encoding: 'utf8', stdio: capture ? ['ignore', 'pipe', 'inherit'] : 'inherit' });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} failed (${result.status})`);
  return result.stdout?.trim();
}
run(process.execPath, [join(root, 'site/scripts/build.mjs')]);
const evidence = JSON.parse(await readFile(join(root, 'site/dist/pingstore-evidence.json'), 'utf8'));
const temporary = await mkdtemp(join(tmpdir(), 'pinglab-pages-'));
try {
  // A separate repository keeps generated files and branch operations out of the source checkout.
  run('git', ['init', '--quiet', '--initial-branch=gh-pages', temporary]);
  run('git', ['remote', 'add', 'origin', run('git', ['remote', 'get-url', 'origin'], root, true)], temporary);
  const refs = run('git', ['ls-remote', '--heads', 'origin', 'gh-pages'], temporary, true);
  if (refs) {
    run('git', ['fetch', '--depth=1', 'origin', 'gh-pages'], temporary);
    run('git', ['reset', '--hard', 'FETCH_HEAD'], temporary);
    // Preserve branch history, but remove files dropped by the new build.
    run('git', ['rm', '-r', '--ignore-unmatch', '.'], temporary);
  }
  await cp(join(root, 'site/dist'), temporary, { recursive: true });
  await mkdir(join(temporary, '.github/workflows'), { recursive: true });
  await cp(join(root, '.github/workflows/pages.yml'), join(temporary, '.github/workflows/pages.yml'));
  run('git', ['add', '--all'], temporary);
  const changes = run('git', ['diff', '--cached', '--name-only'], temporary, true);
  if (!changes) console.log('The published output is already current.');
  else {
    run('git', ['commit', '-m', `Publish site from ${evidence.build.source_revision.slice(0, 12)}`], temporary);
    // A normal push rejects competing publishers instead of overwriting their work.
    run('git', ['push', 'origin', 'HEAD:refs/heads/gh-pages'], temporary);
  }
} finally {
  await rm(temporary, { recursive: true, force: true });
}
console.log(`Publication push finished in ${((performance.now() - started) / 1000).toFixed(1)}s. GitHub Pages deployment follows the gh-pages push.`);
