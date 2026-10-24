import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { readSnapshotEvidence } from './snapshot-data.ts';

async function fixture(callback) {
  const root = await mkdtemp(join(tmpdir(), 'snapshot-data-'));
  const ids = { compute: 'exp124-r001-compute', analyse: 'exp124-r002-analyse', present: 'exp124-r003-present' };
  const digest = `sha256:${'a'.repeat(64)}`;
  const records = Object.fromEntries(Object.entries(ids).map(([stage, id]) => [stage, {
    schema: 'pingstore.run/v4', run_id: id, experiment: 'exp124', stage, payload_digest: digest,
    execution: { completed_at: '2026-10-20T00:00:00Z', configuration: { models: ['coba'], duration_ms: 777 } },
    inputs: stage === 'compute' ? {} : { source: { run_id: ids[stage === 'present' ? 'analyse' : 'compute'], payload_digest: digest } },
  }]));
  records.compute.scientific_artifacts = { native_paths: { 'coba/train/end/snapshot.json': 'coba--train--end/snapshot.json' } };
  const native = {
    schema: 'snnlab.run-snapshot/v1', kind: 'train', stage: 'end', parameters_included: true,
    network: { schema: 'snnlang.graph/v1', populations: [{ id: 'E1', size: 73 }] },
    training: {}, execution: {},
  };
  const writeRecord = async stage => writeFile(join(root, ids[stage], 'run.json'), JSON.stringify(records[stage]));
  for (const stage of Object.keys(ids)) {
    await mkdir(join(root, ids[stage], 'export'), { recursive: true });
    await writeRecord(stage);
  }
  const nativePath = join(root, ids.compute, 'export/coba--train--end/snapshot.json');
  await mkdir(join(root, ids.compute, 'export/coba--train--end'));
  await writeFile(nativePath, JSON.stringify(native));
  // An unrelated newer presentation and legacy table must never supply values.
  await mkdir(join(root, 'exp124-r999-present'));
  await writeFile(join(root, ids.present, 'export/numbers.json'), JSON.stringify({ parameter_table: [['Wrong', 999]] }));
  const options = { experiment: 'exp124', snapshots: { coba: 'coba/train/end/snapshot.json' }, runsRoot: root, allowUnavailable: true };
  try { await callback({ root, ids, records, native, nativePath, options, writeRecord }); }
  finally { await rm(root, { recursive: true, force: true }); }
}

test('reads original mapped snapshots and compute configuration through the pinned ancestry', async () => fixture(async f => {
  const original = await readFile(f.nativePath, 'utf8');
  const data = await readSnapshotEvidence(f.ids.present, f.options);
  assert.equal(data.compute.run_id, f.ids.compute);
  assert.equal(data.configuration.duration_ms, 777);
  assert.equal(data.snapshots.coba.network.populations[0].size, 73);
  assert.equal(await readFile(f.nativePath, 'utf8'), original);
  assert.equal('parameter_table' in data, false);
}));

test('combined presentations require an explicit analysis branch and validate its digest', async () => fixture(async f => {
  f.records.present.inputs.diagnostic = { run_id: 'exp124-r004-analyse', payload_digest: f.records.analyse.payload_digest };
  await f.writeRecord('present');
  await assert.rejects(readSnapshotEvidence(f.ids.present, f.options), /Expected one analyse input/);
  const options = { ...f.options, analysisInput: 'source' };
  assert.equal((await readSnapshotEvidence(f.ids.present, options)).compute.run_id, f.ids.compute);
  await assert.rejects(readSnapshotEvidence(f.ids.present, { ...options, analysisInput: 'missing' }), /Expected one analyse input/);
  f.records.present.inputs.source.payload_digest = `sha256:${'b'.repeat(64)}`;
  await f.writeRecord('present');
  await assert.rejects(readSnapshotEvidence(f.ids.present, options), /Pinned input digest differs/);
}));

test('missing snapshots remain unavailable instead of using old parameter rows', async () => fixture(async f => {
  f.records.compute.scientific_artifacts.native_paths = {};
  await f.writeRecord('compute');
  assert.equal((await readSnapshotEvidence(f.ids.present, f.options)).snapshots.coba, undefined);
  await assert.rejects(readSnapshotEvidence(f.ids.present, { ...f.options, allowUnavailable: false }), /No saved native snapshot/);
  assert.equal(await readSnapshotEvidence('exp124-r998-present', f.options), undefined);
}));

test('reanalysis follows its authenticated earlier analysis and rejects cycles', async () => fixture(async f => {
  const earlier = { ...f.records.analyse, run_id: 'exp124-r004-analyse' };
  await mkdir(join(f.root, earlier.run_id));
  await writeFile(join(f.root, earlier.run_id, 'run.json'), JSON.stringify(earlier));
  f.records.analyse.inputs = { earlier: { run_id: earlier.run_id, payload_digest: earlier.payload_digest } };
  await f.writeRecord('analyse');
  assert.equal((await readSnapshotEvidence(f.ids.present, f.options)).compute.run_id, f.ids.compute);
  earlier.inputs = { cycle: { run_id: f.ids.analyse, payload_digest: f.records.analyse.payload_digest } };
  await writeFile(join(f.root, earlier.run_id, 'run.json'), JSON.stringify(earlier));
  await assert.rejects(readSnapshotEvidence(f.ids.present, f.options), /Cyclic analysis lineage/);
}));

test('a missing pinned ancestor never selects a different run', async () => fixture(async f => {
  await rm(join(f.root, f.ids.compute), { recursive: true });
  assert.equal(await readSnapshotEvidence(f.ids.present, f.options), undefined);
}));

test('rejects changed input identity, incomplete runs and invalid existing JSON', async () => fixture(async f => {
  f.records.analyse.payload_digest = `sha256:${'b'.repeat(64)}`;
  await f.writeRecord('analyse');
  await assert.rejects(readSnapshotEvidence(f.ids.present, f.options), /Pinned input digest differs/);
  f.records.analyse.payload_digest = f.records.present.inputs.source.payload_digest;
  f.records.analyse.execution.completed_at = '';
  await f.writeRecord('analyse');
  await assert.rejects(readSnapshotEvidence(f.ids.present, f.options), /Invalid completed/);
  await writeFile(join(f.root, f.ids.present, 'run.json'), '{');
  await assert.rejects(readSnapshotEvidence(f.ids.present, f.options), SyntaxError);
}));

test('rejects broken declared snapshots and native schema or stage mismatches', async () => fixture(async f => {
  await writeFile(f.nativePath, JSON.stringify({ ...f.native, stage: 'start' }));
  await assert.rejects(readSnapshotEvidence(f.ids.present, f.options), /Invalid native snapshot/);
  await writeFile(f.nativePath, JSON.stringify({ ...f.native, schema: 'replacement' }));
  await assert.rejects(readSnapshotEvidence(f.ids.present, f.options), /Invalid native snapshot/);
  await rm(f.nativePath);
  await assert.rejects(readSnapshotEvidence(f.ids.present, f.options), /ENOENT/);
}));

test('rejects native path traversal and ambiguous compute lineage', async () => fixture(async f => {
  f.records.compute.scientific_artifacts.native_paths['coba/train/end/snapshot.json'] = '../run.json';
  await f.writeRecord('compute');
  await assert.rejects(readSnapshotEvidence(f.ids.present, f.options), /Invalid native snapshot path/);
  await assert.rejects(readSnapshotEvidence('../escape', f.options), /Invalid present identity/);
  f.records.analyse.inputs.another = { run_id: 'exp124-r004-compute', payload_digest: f.records.compute.payload_digest };
  await f.writeRecord('analyse');
  await assert.rejects(readSnapshotEvidence(f.ids.present, f.options), /Expected one compute input/);
}));

test('combined computations require an explicit compute role and authenticate it', async () => fixture(async f => {
  f.records.analyse.inputs.model = { run_id: 'exp124-r004-compute', payload_digest: f.records.compute.payload_digest };
  await f.writeRecord('analyse');
  await assert.rejects(readSnapshotEvidence(f.ids.present, f.options), /Expected one compute input/);
  const options = { ...f.options, computeInput: 'source' };
  assert.equal((await readSnapshotEvidence(f.ids.present, options)).compute.run_id, f.ids.compute);
  await assert.rejects(readSnapshotEvidence(f.ids.present, { ...options, computeInput: 'missing' }), /Expected one compute input/);
  f.records.analyse.inputs.source.payload_digest = `sha256:${'b'.repeat(64)}`;
  await f.writeRecord('analyse');
  await assert.rejects(readSnapshotEvidence(f.ids.present, options), /Pinned input digest differs/);
}));
