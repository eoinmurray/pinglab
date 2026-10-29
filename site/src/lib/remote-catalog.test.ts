import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { setTimeout } from 'node:timers/promises';
import { mkdtemp, readdir, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { remoteCatalog } from './remote-catalog.ts';
import { createServer as createViteServer } from 'vite';
import { createController } from '../../node_modules/astro/dist/vite-plugin-astro-server/controller.js';

test('refresh revalidates with ETag, reuses checked bytes on 304, and sees changed catalogs immediately', async () => {
  let revision = 1;
  const headers: (string | undefined)[] = [];
  const server = createServer((request, response) => {
    headers.push(request.headers['if-none-match']);
    const etag = `"revision-${revision}"`;
    response.setHeader('ETag', etag);
    if (request.headers['if-none-match'] === etag) return response.writeHead(304).end();
    response.setHeader('Content-Type', 'application/json');
    response.end(JSON.stringify({ schema: 'pingstore.catalog/v1', updated_at: `2026-10-0${revision}T00:00:00Z`, runs: {} }));
  });
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  const base = `http://127.0.0.1:${(server.address() as any).port}`;
  try {
    const first = await remoteCatalog(base, true);
    assert.strictEqual(await remoteCatalog(base, true), first);
    // Repeated 304 responses retain the validator, without a stale-time window.
    assert.strictEqual(await remoteCatalog(base, true), first);
    revision = 2;
    const changed = await remoteCatalog(base, true);
    assert.equal(changed.updated_at, '2026-10-02T00:00:00Z');
    assert.equal(first.updated_at, '2026-10-01T00:00:00Z');
    assert.strictEqual(await remoteCatalog(base, true), changed);
    assert.deepEqual(headers, [undefined, '"revision-1"', '"revision-1"', '"revision-1"', '"revision-2"']);
  } finally { server.close(); await once(server, 'close'); }
});

test('persisting a fetched catalog cannot feed Astro error-recovery reloads, while source edits still recover', async () => {
  const root = await mkdtemp(join(tmpdir(), 'catalog-watch-'));
  const previous = process.env.PINGLAB_REPOSITORY_ROOT;
  process.env.PINGLAB_REPOSITORY_ROOT = root;
  const source = join(root, 'source.js');
  await writeFile(source, 'export const value = 1;');
  const reloads: string[] = [];
  const controller = createController({ reload: () => reloads.push('reload') });
  controller.onHMRError({ err: { message: 'No Astro CSS at index 0' } });
  const watcher = await createViteServer({ root, configFile: false, logLevel: 'silent', server: { middlewareMode: true } });
  watcher.watcher.on('change', () => controller.onFileChange());
  const server = createServer((_request, response) => {
    response.setHeader('ETag', '"catalog"');
    response.end(JSON.stringify({ schema: 'pingstore.catalog/v1', updated_at: '2026-10-01T00:00:00Z', runs: {} }));
  });
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  const base = `http://127.0.0.1:${(server.address() as any).port}`;
  try {
    // Confirm that this watcher really observes ordinary source edits.
    const observed = once(watcher.watcher, 'change');
    await setTimeout(100);
    await writeFile(source, 'export const value = 2;');
    await observed;
    assert.equal(reloads.length, 1);
    reloads.length = 0;
    // A fresh module forces a disk write on every request, as happens when the
    // rendering graph is re-evaluated during error recovery.
    for (let index = 0; index < 3; index++) {
      const fresh = await import(new URL(`./remote-catalog.ts?watch=${index}`, import.meta.url).href);
      await fresh.remoteCatalog(base, true);
      await setTimeout(150);
    }
    assert.equal(reloads.length, 0, 'catalog writes must not trigger Astro error-recovery reloads');
  } finally {
    await watcher.close();
    server.close(); await once(server, 'close');
    if (previous === undefined) delete process.env.PINGLAB_REPOSITORY_ROOT;
    else process.env.PINGLAB_REPOSITORY_ROOT = previous;
    await rm(root, { recursive: true, force: true });
  }
});

test('a changed malformed catalog is never accepted or retained as a validated response', async () => {
  let valid = true;
  const headers: (string | undefined)[] = [];
  const server = createServer((request, response) => {
    headers.push(request.headers['if-none-match']);
    response.setHeader('ETag', valid ? '"valid"' : '"invalid"');
    response.end(JSON.stringify({ schema: valid ? 'pingstore.catalog/v1' : 'broken', updated_at: '2026-10-01T00:00:00Z', runs: {} }));
  });
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  const base = `http://127.0.0.1:${(server.address() as any).port}`;
  try {
    await remoteCatalog(base, true);
    valid = false;
    await assert.rejects(remoteCatalog(base, true), /Invalid R2 catalog/);
    assert.ok(headers.slice(1).every(header => header === '"valid"'));
    valid = true;
    await remoteCatalog(base, true);
    assert.equal(headers.at(-1), undefined);
  } finally { server.close(); await once(server, 'close'); }
});

test('a restarted server revalidates a checked disk snapshot, and corrupt cache bytes force a full fetch', async () => {
  const root = await mkdtemp(join(tmpdir(), 'catalog-restart-'));
  const previous = process.env.PINGLAB_REPOSITORY_ROOT;
  process.env.PINGLAB_REPOSITORY_ROOT = root;
  const headers: (string | undefined)[] = [];
  const server = createServer((request, response) => {
    headers.push(request.headers['if-none-match']);
    response.setHeader('ETag', '"catalog"');
    if (request.headers['if-none-match'] === '"catalog"') return response.writeHead(304).end();
    response.end(JSON.stringify({ schema: 'pingstore.catalog/v1', updated_at: '2026-10-01T00:00:00Z', runs: {} }));
  });
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  const base = `http://127.0.0.1:${(server.address() as any).port}`;
  try {
    const expected = await remoteCatalog(base, true);
    // A fresh module has no in-memory catalog, just like a new dev process.
    const restarted = await import(new URL('./remote-catalog.ts?restart=valid', import.meta.url).href);
    assert.deepEqual(await restarted.remoteCatalog(base, true), expected);
    assert.deepEqual(headers, [undefined, '"catalog"']);
    const directory = join(root, 'site/node_modules/.cache/pinglab/catalogs');
    const [file] = await readdir(directory);
    const path = join(directory, file);
    const saved = JSON.parse(await readFile(path, 'utf8'));
    saved.body += ' ';
    await writeFile(path, JSON.stringify(saved));
    const corrupted = await import(new URL('./remote-catalog.ts?restart=corrupt', import.meta.url).href);
    assert.deepEqual(await corrupted.remoteCatalog(base, true), expected);
    assert.equal(headers.at(-1), undefined);
  } finally {
    server.close(); await once(server, 'close');
    if (previous === undefined) delete process.env.PINGLAB_REPOSITORY_ROOT;
    else process.env.PINGLAB_REPOSITORY_ROOT = previous;
    await rm(root, { recursive: true, force: true });
  }
});
