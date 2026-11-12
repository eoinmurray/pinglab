import { spawnSync } from 'node:child_process';
import { copyFile, mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';

const started = performance.now();
if (process.env.PINGLAB_RUN_SOURCE && process.env.PINGLAB_RUN_SOURCE !== 'local') {
  throw new Error('The publication build uses local evidence. Use bun run build:remote for remote-only validation.');
}
const root = fileURLToPath(new URL('../../', import.meta.url));
const site = join(root, 'site');
const scratch = join(root, '.publication');
const runsRoot = resolve(site, process.env.PINGLAB_RUNS_ROOT ?? '../.pingstore/runs');
await mkdir(scratch, { recursive: true });
const env = { ...process.env, PINGLAB_RUN_SOURCE: 'local', PINGLAB_BUNDLE_REFERENCES: '1', PINGLAB_RUNS_ROOT: runsRoot };
function run(command, args, options = {}) {
  const result = spawnSync(command, args, { cwd: root, env, stdio: 'inherit', ...options });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} failed (${result.status})`);
  return result;
}
// Freeze one metadata-only selection before any MDX module is evaluated.
const { RunStore } = await import('../src/lib/run-store.ts');
const parsed = await new RunStore({ runsRoot, source: 'local' }).index();
const indexPath = join(scratch, 'index.json');
await writeFile(indexPath, JSON.stringify(parsed));
env.PINGLAB_REPORT_INDEX_PATH = indexPath;
run(process.execPath, [join(site, 'node_modules/astro/bin/astro.mjs'), 'build'], { cwd: site });

const dist = join(site, 'dist');
const evidencePath = join(dist, 'pingstore-evidence.json');
const evidence = JSON.parse(await readFile(evidencePath, 'utf8'));
async function files(directory, prefix = '') {
  const result = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const name = prefix + entry.name;
    if (entry.isDirectory()) result.push(...await files(join(directory, entry.name), `${name}/`));
    else result.push(name);
  }
  return result;
}
const output = await files(dist);
const used = new Set();
const pending = output.filter(name => /\.(html|js|css|json)$/.test(name));
// Traverse references, including animation JSON, rather than shipping every export.
while (pending.length) {
  const name = pending.pop();
  const content = (await readFile(join(dist, name), 'utf8')).replaceAll('\\/', '/');
  for (const match of content.matchAll(/(?<![a-zA-Z0-9_/-])\/run-images\/([a-zA-Z0-9][a-zA-Z0-9_.-]{0,159})\/([a-zA-Z0-9_.-]+)/g)) {
    const asset = `run-images/${match[1]}/${match[2]}`;
    if (!used.has(asset)) {
      if (!parsed.runs[match[1]]) throw new Error(`Asset is outside the frozen evidence: ${asset}`);
      if (!/\.(svg|png|mp4|webm|ogv|json)$/.test(match[2])) throw new Error(`Unsupported asset: ${asset}`);
      await mkdir(dirname(join(dist, asset)), { recursive: true });
      await copyFile(join(runsRoot, match[1], 'export', match[2]), join(dist, asset));
      used.add(asset);
      if (asset.endsWith('.json')) pending.push(asset);
    }
  }
  if (/https?:\/\/[^\s"'<>]*\/pingstore(?:-v2)?\/runs\//.test(content)) {
    throw new Error(`Publication still references a remote run asset: ${name}`);
  }
}
evidence.publication_mode = 'bundled';
evidence.build.source_revision = evidence.build.revision;
evidence.build.source_dirty = !!run('git', ['status', '--porcelain'], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'] }).stdout.trim();
await writeFile(evidencePath, JSON.stringify(evidence, null, 2) + '\n');
await writeFile(join(dist, '.nojekyll'), '');
await writeFile(join(dist, 'CNAME'), 'pinglab.eoinmurray.info\n');
console.log(`Built locally in ${((performance.now() - started) / 1000).toFixed(1)}s; bundled ${used.size} referenced assets.`);
