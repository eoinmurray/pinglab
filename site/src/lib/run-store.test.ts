import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { RunStore } from './run-store.ts';
import { validateCatalog, pathCompare, type Catalog } from './remote-catalog.ts';
import { selectAsset } from './run-assets.ts';
import { readSnapshotEvidence } from './snapshot-data.ts';

const hash = (value: string) => createHash('sha256').update(value).digest('hex');

export async function remoteFixture(callback) {
  const root = await mkdtemp(join(tmpdir(), 'remote-runs-'));
  const bodies = new Map<string, string>();
  const ids = { compute: 'exp124-r001-compute', analyse: 'exp124-r002-analyse', present: 'exp124-r003-present' };
  const catalog: Catalog = { schema: 'pingstore.catalog/v1', updated_at: '2026-10-23T00:00:00Z', runs: {} };
  const snapshot = { schema: 'snnlab.run-snapshot/v1', kind: 'train', stage: 'end', parameters_included: true,
    network: { schema: 'snnlang.graph/v1', populations: [{ id: 'E', size: 73 }] }, training: {}, execution: {} };
  function addRun(id: string, exports: Record<string, string>, parent?: string) {
    const payload = Object.entries(exports).sort(([a], [b]) => pathCompare(a, b))
      .map(([path, value]) => ({ path: `export/${path}`, sha256: hash(value), size_bytes: Buffer.byteLength(value) }));
    const stage = id.split('-').at(-1);
    const record = { schema: 'pingstore.run/v4', run_id: id, experiment: 'exp124', stage,
      payload_digest: `sha256:${hash(JSON.stringify(payload))}`, created_at: '2026-10-23T00:00:00Z',
      execution: { completed_at: '2026-10-23T00:00:00Z', configuration: { duration_ms: 777 } },
      inputs: parent ? { source: { run_id: parent, payload_digest: catalog.runs[parent].record.payload_digest } } : {},
      scientific_artifacts: stage === 'compute' ? { native_paths: { 'coba/train/end/snapshot.json': 'native/snapshot.json' } } : undefined };
    const files = { 'run.json': JSON.stringify(record), 'README.md': 'Run history',
      ...Object.fromEntries(Object.entries(exports).map(([path, value]) => [`export/${path}`, value])) };
    catalog.runs[id] = { record, files: Object.fromEntries(Object.entries(files).map(([path, value]) => {
      bodies.set(`/pingstore/runs/${id}/${path}`, value);
      return [path, { bytes: Buffer.byteLength(value), sha256: hash(value) }];
    })) };
  }
  addRun(ids.compute, { 'native/snapshot.json': JSON.stringify(snapshot), 'native-extra.json': '{}' });
  addRun(ids.analyse, { 'results.json': '{}' }, ids.compute);
  addRun(ids.present, { 'numbers.json': '{"score":42}', 'figure.svg': '<svg/>', 'training.json': '{}' }, ids.analyse);
  addRun('exp124-r999-present', { 'numbers.json': '{"score":999}' }, ids.analyse);
  let requests = 0;
  let cors = true;
  const server = createServer((request, response) => {
    requests++;
    if (request.url === '/pingstore/catalog.json') { response.setHeader('Content-Type', 'application/json'); response.end(JSON.stringify(catalog)); return; }
    const body = bodies.get(decodeURIComponent(request.url!));
    if (body === undefined) { response.writeHead(404).end(); return; }
    if (cors) response.setHeader('Access-Control-Allow-Origin', '*');
    response.setHeader('Content-Type', request.url!.endsWith('.svg') ? 'image/svg+xml' : 'application/json');
    response.end(body);
  });
  await new Promise<void>(done => server.listen(0, '127.0.0.1', done));
  const base = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  const store = new RunStore({ runsRoot: root, baseUrl: base, source: 'auto' });
  try { await callback({ root, ids, catalog, bodies, store, base, setCors: (value: boolean) => { cors = value; }, requests: () => requests }); }
  finally { server.closeAllConnections(); await new Promise<void>(done => server.close(() => done())); await rm(root, { recursive: true, force: true }); }
}

test('empty local store resolves catalog, exact numbers, figures and mapped snapshots from R2', async () => remoteFixture(async f => {
  assert.equal((await f.store.ids()).length, 4);
  assert.equal(await f.store.latest('exp124'), 'exp124-r999-present');
  assert.equal((await f.store.json(f.ids.present, 'export/numbers.json')).score, 42);
  const data = await readSnapshotEvidence(f.ids.present, { experiment: 'exp124', snapshots: { coba: 'coba/train/end/snapshot.json' }, store: f.store });
  assert.equal(data.compute.run_id, f.ids.compute);
  assert.equal(data.configuration.duration_ms, 777);
  assert.equal(data.snapshots.coba.network.populations[0].size, 73);
  assert.equal((await selectAsset(f.ids.present, 'figure.svg', { store: f.store })).source, 'r2');
  assert.equal((await selectAsset(f.ids.present, 'training.json', { store: f.store })).source, 'r2');
}));

test('local exact runs and local latest override remote only in auto mode; corrupt local files fail', async () => remoteFixture(async f => {
  await mkdir(join(f.root, f.ids.present, 'export'), { recursive: true });
  await writeFile(join(f.root, f.ids.present, 'run.json'), JSON.stringify(f.catalog.runs[f.ids.present].record));
  await writeFile(join(f.root, f.ids.present, 'export/numbers.json'), '{"score":7}');
  await writeFile(join(f.root, f.ids.present, 'export/figure.svg'), '<local/>');
  assert.equal(await f.store.latest('exp124'), f.ids.present);
  assert.equal((await f.store.json(f.ids.present, 'export/numbers.json')).score, 7);
  assert.equal((await selectAsset(f.ids.present, 'figure.svg', { store: f.store })).source, 'local');
  const remote = new RunStore({ runsRoot: f.root, baseUrl: f.base, source: 'remote' });
  assert.equal(await remote.latest('exp124'), 'exp124-r999-present');
  assert.equal((await remote.json(f.ids.present, 'export/numbers.json')).score, 42);
  await writeFile(join(f.root, f.ids.present, 'export/numbers.json'), '{');
  await assert.rejects(f.store.json(f.ids.present, 'export/numbers.json'), SyntaxError);
  await assert.rejects(f.store.read(f.ids.present, 'export/training.json'), /ENOENT/);
}));

test('rejects changed remote bytes, missing pins, invalid catalog ancestry and unsafe paths', async () => remoteFixture(async f => {
  f.bodies.set(`/pingstore/runs/${f.ids.present}/export/numbers.json`, '{"score":43}');
  await assert.rejects(f.store.json(f.ids.present, 'export/numbers.json'), /checksum mismatch/);
  await assert.rejects(f.store.json('exp124-r998-present'), /unavailable/);
  await assert.rejects(f.store.read(f.ids.present, '../run.json'), /Invalid run file/);
  await assert.rejects(selectAsset(f.ids.present, '../run.json', { store: f.store }), /Invalid run asset/);
  const corrupt = structuredClone(f.catalog);
  corrupt.runs[f.ids.present].files['export/numbers.json'].sha256 = '0'.repeat(64);
  assert.throws(() => validateCatalog(corrupt), /payload inventory mismatch/);
  delete corrupt.runs[f.ids.compute];
  assert.throws(() => validateCatalog(corrupt), /ancestor/);
}));

test('local-only operation never contacts R2', async () => remoteFixture(async f => {
  const store = new RunStore({ runsRoot: f.root });
  assert.deepEqual(await store.ids(), []);
  assert.equal(await store.latest('exp124'), undefined);
  await assert.rejects(store.read(f.ids.present, 'run.json'), /ENOENT/);
  assert.equal(f.requests(), 0);
}));


test('browser animation JSON requires CORS even though server-side reads do not', async () => remoteFixture(async f => {
  f.setCors(false);
  assert.deepEqual(await f.store.json(f.ids.present, 'export/training.json'), {});
  await assert.rejects(selectAsset(f.ids.present, 'training.json', { store: f.store }), /CORS unavailable/);
}));


test('auto mode can list local history during an R2 outage; remote mode fails explicitly', async t => remoteFixture(async f => {
  t.mock.method(console, 'warn', () => {});
  await mkdir(join(f.root, f.ids.present, 'export'), { recursive: true });
  await writeFile(join(f.root, f.ids.present, 'run.json'), JSON.stringify(f.catalog.runs[f.ids.present].record));
  const catalog = async () => { throw new Error('R2 unavailable'); };
  const auto = new RunStore({ runsRoot: f.root, source: 'auto', catalog });
  assert.deepEqual(await auto.ids('exp124'), [f.ids.present]);
  assert.deepEqual(await auto.ids('exp999'), []);
  assert.equal(await auto.latest('exp124'), f.ids.present);
  const remote = new RunStore({ runsRoot: f.root, source: 'remote', catalog });
  await assert.rejects(remote.ids('exp124'), /R2 unavailable/);
}));


test('an unpublished catalog leaves absent historical pins unavailable in auto mode', async t => {
  t.mock.method(console, 'warn', () => {});
  const root = await mkdtemp(join(tmpdir(), 'missing-catalog-'));
  const catalog = async () => { throw Object.assign(new Error('R2 catalog: HTTP 404'), { code: 'ENOENT' }); };
  const store = new RunStore({ runsRoot: root, source: 'auto', catalog });
  try {
    assert.deepEqual(await store.ids('exp124'), []);
    assert.equal(await store.latest('exp124'), undefined);
    assert.equal(await store.has('exp124-r001-present', 'export/numbers.json'), false);
  } finally { await rm(root, { recursive: true, force: true }); }
});


test('catalog payload order matches Python path components for prefix siblings', () => {
  assert.deepEqual(['export/unit-extra.json', 'export/unit/data.json', 'export/a.json'].sort(pathCompare),
    ['export/a.json', 'export/unit/data.json', 'export/unit-extra.json']);
});
