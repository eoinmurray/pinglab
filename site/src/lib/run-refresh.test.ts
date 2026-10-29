import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile, rm } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { saveLocalRun } from './run-fixtures.ts';
import { withRunContext } from './run-context.ts';
import { resolveRunIndex, validateRunIndex } from './run-index.ts';
import { parseCatalog } from './remote-catalog.ts';

test('catalog transport preserves large integer seeds and still detects real record conflicts', async () => {
  const root = await mkdtemp(join(tmpdir(), 'catalog-integers-'));
  try {
    const record = await saveLocalRun(root, 'exp001-r001-present', undefined, { configuration: { seed: 'SEED' } });
    const rawRecord = JSON.stringify(record).replace('"SEED"', '5272556675475122279');
    await writeFile(join(root, record.run_id, 'run.json'), rawRecord);
    const files: Record<string, { bytes: number; sha256: string }> = {};
    for (const path of ['run.json', 'README.md', 'export/result.json']) {
      const body = await readFile(join(root, record.run_id, path));
      files[path] = { bytes: body.length, sha256: createHash('sha256').update(body).digest('hex') };
    }
    const body = `{"schema":"pingstore.catalog/v1","updated_at":"2026-10-01T00:00:00Z","runs":{"${record.run_id}":{"record":${rawRecord},"files":${JSON.stringify(files)}}}}`;
    const index = await resolveRunIndex({ root, source: 'auto', catalog: parseCatalog(body) });
    assert.equal(index.selections.exp001, record.run_id);
    assert.deepEqual(index.selection_errors, {});
    const changed = await resolveRunIndex({ root, source: 'auto',
      catalog: parseCatalog(body.replace('5272556675475122279', '5272556675475122280')) });
    assert.match(changed.selection_errors.exp001, /local and R2 records differ/);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('a reload refreshes numbers, figures, snapshots and Dataset together; in-flight requests retain their selection', async () => {
  const root = await mkdtemp(join(tmpdir(), 'request-runsets-'));
  process.env.PINGLAB_RUNS_ROOT = root;
  process.env.PINGLAB_RUN_SOURCE = 'local';
  process.env.NODE_ENV = 'development';
  try {
    const compute = await saveLocalRun(root, 'exp001-r001-compute');
    const analysis = await saveLocalRun(root, 'exp001-r002-analyse', undefined, { parent: compute });
    const old = await saveLocalRun(root, 'exp001-r003-present', { 'numbers.json': '{"score":3}', 'figure.svg': '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 400"/>' }, { parent: analysis });
    const { optionalPresentation, dataset, figureUrl, figureImage } = await import('./pingstore.ts');
    const { reportData, reportSnapshots } = await import('./report-data.ts');
    await withRunContext(async () => {
      assert.equal((await optionalPresentation('exp001'))?.id, old.run_id);
      const newest = await saveLocalRun(root, 'exp001-r004-present', { 'numbers.json': '{"score":4}', 'figure.svg': '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 300"/>' }, { parent: analysis, day: '02' });
      // Even an overlapping request gets a fresh index without changing this one.
      await withRunContext(async () => {
        assert.equal((await optionalPresentation('exp001'))?.id, newest.run_id);
        assert.equal((await reportData('exp001')).score, 4);
        assert.match(await figureUrl(newest.run_id, 'figure.png'), /exp001-r004-present/);
        assert.deepEqual(await figureImage(newest.run_id, 'figure.png'), {
          src: '/run-images/exp001-r004-present/figure.svg', width: 800, height: 300,
        });
        assert.equal((await reportSnapshots('exp001', { snapshots: {} }))?.compute.run_id, compute.run_id);
        assert.equal((await dataset('exp001')).runs.find(row => row.selected)?.id, newest.run_id);
      });
      assert.equal((await reportData('exp001')).score, 3);
      assert.deepEqual(await figureImage(old.run_id, 'figure.png'), {
        src: '/run-images/exp001-r003-present/figure.svg', width: 600, height: 400,
      });
      assert.equal((await dataset('exp001')).runs.find(row => row.selected)?.id, old.run_id);
    });
    await withRunContext(async () => {
      assert.equal((await optionalPresentation('exp001'))?.id, 'exp001-r004-present');
      assert.equal((await reportData('exp001')).score, 4);
    });
  } finally {
    await rm(root, { recursive: true, force: true });
    delete process.env.PINGLAB_RUNS_ROOT;
    delete process.env.PINGLAB_RUN_SOURCE;
    delete process.env.NODE_ENV;
  }
});

test('TypeScript consumes the Python contract and rejects incompatible indexes or changed ancestry', async () => {
  const root = await mkdtemp(join(tmpdir(), 'index-contract-'));
  try {
    const parent = await saveLocalRun(root, 'exp001-r001-compute');
    const present = await saveLocalRun(root, 'exp001-r002-present', undefined, { parent });
    const index = await resolveRunIndex({ root, source: 'local' });
    assert.equal(index.selections.exp001, present.run_id);
    assert.equal(index.runs[parent.run_id].storage, 'Local');
    assert.throws(() => validateRunIndex({ ...index, schema: 'pingstore.index/v999' }), /Invalid Pingstore index/);
    const changed = structuredClone(index);
    changed.runs[parent.run_id].record.payload_digest = 'sha256:' + '0'.repeat(64);
    assert.throws(() => validateRunIndex(changed), /Changed indexed pin/);
    const missing = structuredClone(index);
    delete missing.runs[parent.run_id];
    assert.throws(() => validateRunIndex(missing), /Invalid indexed selection/);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('request context freezes remote metadata until reload', async () => {
  const { RunStore } = await import('./run-store.ts');
  let calls = 0;
  const store = new RunStore({ source: 'auto', catalog: async () => ({
    schema: 'pingstore.catalog/v1', updated_at: String(++calls), runs: {},
  }) });
  await withRunContext(async () => {
    assert.equal((await store.catalog()).updated_at, '1');
    assert.equal((await store.catalog()).updated_at, '1');
    await withRunContext(async () => assert.equal((await store.catalog()).updated_at, '2'));
    assert.equal((await store.catalog()).updated_at, '1');
  });
  await withRunContext(async () => assert.equal((await store.catalog()).updated_at, '3'));
});

test('production reads one catalog snapshot throughout the build', async () => {
  const { RunStore } = await import('./run-store.ts');
  const previous = process.env.NODE_ENV;
  process.env.NODE_ENV = 'production';
  let calls = 0;
  try {
    const store = new RunStore({ source: 'remote', catalog: async () => ({
      schema: 'pingstore.catalog/v1', updated_at: String(++calls), runs: {},
    }) });
    assert.equal((await store.catalog()).updated_at, '1');
    await withRunContext(async () => assert.equal((await store.catalog()).updated_at, '1'));
    assert.equal(calls, 1);
  } finally {
    if (previous === undefined) delete process.env.NODE_ENV;
    else process.env.NODE_ENV = previous;
  }
});

test('catalog and publication refreshes start together without sharing snapshots across requests', { timeout: 5000 }, async () => {
  const { RunStore } = await import('./run-store.ts');
  const root = await mkdtemp(join(tmpdir(), 'parallel-metadata-'));
  let finish!: (value: any) => void;
  let catalogs = 0, controls = 0;
  const store = new RunStore({ runsRoot: root, source: 'auto',
    catalog: () => { catalogs++; return new Promise(done => { finish = done; }); },
    control: async () => {
      controls++;
      finish({ schema: 'pingstore.catalog/v1', updated_at: '2026-10-01T00:00:00Z', runs: {} });
      return {};
    },
  });
  try {
    await withRunContext(async () => {
      await store.index(); await store.index();
      assert.equal(catalogs, 1); assert.equal(controls, 1);
    });
    await withRunContext(async () => { await store.index(); });
    assert.equal(catalogs, 2); assert.equal(controls, 2);
  } finally { await rm(root, { recursive: true, force: true }); }
});
