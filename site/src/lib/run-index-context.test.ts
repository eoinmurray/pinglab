import { test } from 'node:test';
import assert from 'node:assert/strict';
import { evidenceForExperiment, type IndexRun } from './run-index.ts';

test('shared historical inputs are required in their own experiment and current in the consuming runset', () => {
  const row: IndexRun = {
    record: {}, current: ['exp153'], status: 'Current', storage: 'Both',
    verification: { local: 'metadata-validated', r2: 'catalog-verified' },
    reasons: ['current:exp153', 'report:exp153'], unresolved: [],
  };
  const displayed = evidenceForExperiment(row, 'exp133');
  assert.equal(displayed.status, 'Required');
  assert.deepEqual(displayed.reasons, row.reasons);
  assert.equal(row.status, 'Current');
  assert.equal(evidenceForExperiment(row, 'exp153'), row);
});

test('unresolved evidence stays unresolved when changing experiment context', () => {
  const row: IndexRun = {
    record: {}, current: [], status: 'Unresolved', storage: 'Unknown',
    verification: { local: 'absent', r2: 'unknown' }, reasons: [],
    unresolved: ['R2 inventory unavailable'],
  };
  assert.equal(evidenceForExperiment(row, 'exp133'), row);
});
