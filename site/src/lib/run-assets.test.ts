import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { selectAsset } from './run-assets.ts';

test('selects matching R2 assets and safely falls back for unavailable or unsuitable copies', async () => {
  const root = await mkdtemp(join(tmpdir(), 'run-assets-'));
  const run = 'exp001-r001-present';
  const directory = join(root, run, 'export');
  await mkdir(directory, { recursive: true });
  await writeFile(join(directory, 'figure.svg'), '<svg/>');
  await writeFile(join(directory, 'training.json'), '{}');
  let requests = 0;
  const server = createServer((request, response) => {
    requests++;
    if (request.url!.startsWith('/missing/')) { response.writeHead(404).end(); return; }
    if (request.url!.startsWith('/timeout/')) return;
    const json = request.url!.endsWith('.json');
    response.setHeader('Content-Type', request.url!.startsWith('/mime/') ? 'text/html' : json ? 'application/json' : 'image/svg+xml');
    if (request.url!.startsWith('/cors/')) response.setHeader('Access-Control-Allow-Origin', '*');
    response.end(request.url!.startsWith('/wrong/') ? '<bad/>' : json ? '{}' : '<svg/>');
  });
  await new Promise<void>(done => server.listen(0, '127.0.0.1', done));
  const address = server.address() as { port: number };
  const base = `http://127.0.0.1:${address.port}`;
  const choose = (scenario: string, file = 'figure.svg') => selectAsset(run, file, {
    runsRoot: root, baseUrl: `${base}/${scenario}`, timeoutMs: 100,
  });
  try {
    const remote = await choose('ok');
    assert.equal(remote.source, 'r2');
    assert.equal(remote.url, `${base}/ok/pingstore/runs/${run}/export/figure.svg`);
    assert.equal(remote.bytes, 6);
    const before = requests;
    assert.deepEqual(await choose('ok'), remote);
    assert.equal(requests, before, 'reuses the frozen choice');
    for (const scenario of ['missing', 'timeout', 'mime', 'wrong']) {
      const local = await choose(scenario);
      assert.equal(local.source, 'local', scenario);
      assert.equal(local.url, `/run-images/${run}/figure.svg`);
      assert.ok(local.reason);
    }
    assert.equal((await choose('ok', 'training.json')).reason, 'CORS unavailable');
    assert.equal((await choose('cors', 'training.json')).source, 'r2');
    assert.equal((await selectAsset(run, 'figure.svg', { runsRoot: root, baseUrl: '' })).reason, 'R2 disabled');
    await assert.rejects(choose('ok', '../run.json'), /Invalid run asset/);
    await assert.rejects(choose('ok', 'absent.png'), /ENOENT/);
    await writeFile(join(directory, 'figure.svg'), '<changed/>');
    assert.equal((await choose('ok')).source, 'local', 'a changed local file cannot reuse an old match');
  } finally {
    server.closeAllConnections();
    await new Promise<void>(done => server.close(() => done()));
    await rm(root, { recursive: true, force: true });
  }
});
