import { expect, test } from 'bun:test';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { remoteCatalog } from './remote-catalog.ts';
import { RunStore } from './run-store.ts';
import { withRunContext } from './run-context.ts';

test('concurrent refreshes share one fetch, but the next page revalidates', async () => {
  let requests = 0;
  let release!: () => void;
  const gate = new Promise<void>(done => { release = done; });
  const server = Bun.serve({ port: 0, async fetch(request) {
    requests++;
    await gate;
    if (request.headers.get('if-none-match') === '"catalog-v1"') return new Response(null, { status: 304 });
    return Response.json({ schema: 'pingstore.catalog/v1', updated_at: new Date().toISOString(), runs: {} },
      { headers: { ETag: '"catalog-v1"' } });
  } });
  const base = server.url.origin;
  try {
    const reads = Array.from({ length: 6 }, () => remoteCatalog(base, true));
    release();
    const catalogs = await Promise.all(reads);
    expect(requests).toBe(1);
    expect(catalogs.every(catalog => catalog === catalogs[0])).toBe(true);
    expect(await remoteCatalog(base)).toBe(catalogs[0]);
    expect(requests).toBe(1);
    expect(await remoteCatalog(base, true)).toBe(catalogs[0]);
    expect(requests).toBe(2);
  } finally { server.stop(true); }
});

test('asset reads share a request catalog and still reject changed local bytes', async () => {
  const root = await mkdtemp(join(tmpdir(), 'pinglab-assets-'));
  const run = 'exp001-r001-present';
  const bytes = Buffer.from('<svg/>');
  let catalogs = 0;
  try {
    await mkdir(join(root, run, 'export'), { recursive: true });
    const file = join(root, run, 'export/result.svg');
    await writeFile(file, bytes);
    const store = new RunStore({ runsRoot: root, source: 'auto', catalog: async () => {
      catalogs++;
      return { schema: 'pingstore.catalog/v1', updated_at: new Date().toISOString(), runs: {
        [run]: { record: {}, files: { 'export/result.svg': { bytes: bytes.length,
          sha256: createHash('sha256').update(bytes).digest('hex') } } },
      } };
    } });
    await withRunContext(async () => {
      expect(await store.read(run, 'export/result.svg')).toEqual(bytes);
      expect(await store.read(run, 'export/result.svg')).toEqual(bytes);
    });
    expect(catalogs).toBe(1);
    await writeFile(file, '<svg corrupt/>');
    await expect(withRunContext(() => store.read(run, 'export/result.svg'))).rejects.toThrow('checksum mismatch');
    expect(catalogs).toBe(2);
  } finally { await rm(root, { recursive: true, force: true }); }
});
