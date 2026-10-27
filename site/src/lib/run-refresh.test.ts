import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { saveLocalRun } from './run-fixtures.ts';
import { withRunContext } from './run-context.ts';
import { resolveRunIndex, validateRunIndex } from './run-index.ts';

test('a reload refreshes numbers, figures, snapshots and Dataset together; in-flight requests retain their selection', async () => {
  const root = await mkdtemp(join(tmpdir(), 'request-runsets-'));
  process.env.PINGLAB_RUNS_ROOT = root;
  process.env.PINGLAB_RUN_SOURCE = 'local';
  process.env.NODE_ENV = 'development';
  try {
    const compute = await saveLocalRun(root, 'exp001-r001-compute');
    const analysis = await saveLocalRun(root, 'exp001-r002-analyse', undefined, { parent: compute });
    const old = await saveLocalRun(root, 'exp001-r003-present', { 'numbers.json': '{"score":3}', 'figure.svg': '<svg/>' }, { parent: analysis });
    const { optionalPresentation, dataset, figureUrl } = await import('./pingstore.ts');
    const { reportData, reportSnapshots } = await import('./report-data.ts');
    await withRunContext(async () => {
      assert.equal((await optionalPresentation('exp001'))?.id, old.run_id);
      const newest = await saveLocalRun(root, 'exp001-r004-present', { 'numbers.json': '{"score":4}', 'figure.svg': '<svg/>' }, { parent: analysis, day: '02' });
      // Even an overlapping request gets a fresh index without changing this one.
      await withRunContext(async () => {
        assert.equal((await optionalPresentation('exp001'))?.id, newest.run_id);
        assert.equal((await reportData('exp001')).score, 4);
        assert.match(await figureUrl(newest.run_id, 'figure.png'), /exp001-r004-present/);
        assert.equal((await reportSnapshots('exp001', { snapshots: {} }))?.compute.run_id, compute.run_id);
        assert.equal((await dataset('exp001')).runs.find(row => row.selected)?.id, newest.run_id);
      });
      assert.equal((await reportData('exp001')).score, 3);
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
