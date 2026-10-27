import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { saveLocalRun } from './run-fixtures.ts';

test('reports share a frozen latest presentation for numbers and native ancestry without metadata pins', async () => {
  const root = await mkdtemp(join(tmpdir(), 'report-selection-'));
  process.env.PINGLAB_RUNS_ROOT = root;
  process.env.PINGLAB_RUN_SOURCE = 'local';
  process.env.NODE_ENV = 'production';
  const records = new Map<string, any>();
  const ids = { compute: 'exp001-r001-compute', analyse: 'exp001-r002-analyse', present: 'exp001-r003-present' };
  async function save(id: string, stage: string, parent?: string, day = '01') {
    records.set(id, await saveLocalRun(root, id,
      stage === 'present' ? { 'numbers.json': '{"score":3}' } : { 'result.json': '{}' },
      { parent: parent ? records.get(parent) : undefined, day }));
  }
  try {
    await save(ids.compute, 'compute');
    await save(ids.analyse, 'analyse', ids.compute);
    await save(ids.present, 'present', ids.analyse);
    const { reportData, reportHasData, reportSnapshots, unavailableReportInputs } = await import('./report-data.ts');
    const { optionalPresentation, dataset } = await import('./pingstore.ts');
    assert.equal(await reportHasData('exp001'), true);
    assert.equal((await optionalPresentation('exp001'))?.id, ids.present);
    // A newly completed run must not split numbers and snapshots across lineages.
    await save('exp001-r004-present', 'present', ids.analyse, '02');
    assert.deepEqual(await reportData('exp001'), { score: 3 });
    assert.equal((await reportSnapshots('exp001', { snapshots: {} }))?.compute.run_id, ids.compute);
    assert.equal((await dataset('exp001')).runs.find(run => run.selected)?.id, ids.present);
    assert.deepEqual(await reportData('exp110', { experiment: 'exp001' }), { score: 3 });
    assert.equal(await reportHasData('exp002'), false);
    assert.deepEqual(await unavailableReportInputs(['exp001', 'exp002'], true), ['exp002']);
    assert.deepEqual(await unavailableReportInputs(['exp002'], false), []);
    assert.equal(await reportSnapshots('exp002', { snapshots: {}, allowUnavailable: true }), undefined);
    await assert.rejects(reportSnapshots('exp002', { snapshots: {} }), /No presentation data/);
    // Native lineage is always read and validated, independently of cached numbers.
    await rm(join(root, ids.compute), { recursive: true });
    await assert.rejects(reportSnapshots('exp001', { snapshots: {} }), /ENOENT/);
  } finally {
    await rm(root, { recursive: true, force: true });
    delete process.env.PINGLAB_RUNS_ROOT;
    delete process.env.PINGLAB_RUN_SOURCE;
    delete process.env.NODE_ENV;
  }
});
