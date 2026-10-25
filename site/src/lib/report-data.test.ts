import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

test('reports share a frozen latest presentation for numbers and native ancestry without metadata pins', async () => {
  const root = await mkdtemp(join(tmpdir(), 'report-selection-'));
  process.env.PINGLAB_RUNS_ROOT = root;
  process.env.PINGLAB_RUN_SOURCE = 'local';
  const digest = `sha256:${'a'.repeat(64)}`;
  const ids = { compute: 'exp001-r001-compute', analyse: 'exp001-r002-analyse', present: 'exp001-r003-present' };
  async function save(id: string, stage: string, parent?: string, day = '01') {
    await mkdir(join(root, id, 'export'), { recursive: true });
    await writeFile(join(root, id, 'run.json'), JSON.stringify({
      schema: 'pingstore.run/v4', run_id: id, experiment: 'exp001', stage, payload_digest: digest,
      execution: { completed_at: `2026-10-${day}T00:00:00Z`, configuration: { origin: id } },
      inputs: parent ? { source: { run_id: parent, payload_digest: digest } } : {},
    }));
  }
  try {
    await save(ids.compute, 'compute');
    await save(ids.analyse, 'analyse', ids.compute);
    await save(ids.present, 'present', ids.analyse);
    await writeFile(join(root, ids.present, 'export/numbers.json'), '{"score":3}');
    const { reportData, reportHasData, reportSnapshots } = await import('./report-data.ts');
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
    assert.equal(await reportSnapshots('exp002', { snapshots: {}, allowUnavailable: true }), undefined);
    await assert.rejects(reportSnapshots('exp002', { snapshots: {} }), /No presentation data/);
    // Native lineage is always read and validated, independently of cached numbers.
    await rm(join(root, ids.compute), { recursive: true });
    await assert.rejects(reportSnapshots('exp001', { snapshots: {} }), /ENOENT/);
  } finally {
    await rm(root, { recursive: true, force: true });
    delete process.env.PINGLAB_RUNS_ROOT;
    delete process.env.PINGLAB_RUN_SOURCE;
  }
});
