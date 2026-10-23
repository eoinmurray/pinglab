import { test } from 'node:test';
import assert from 'node:assert/strict';
import { RunStore } from './run-store.ts';
import { selectAsset } from './run-assets.ts';

test('asset validation rejects invalid run identifiers and arbitrary exported files', async () => {
  const store = new RunStore({ source: 'local' });
  for (const [run, file] of [['../run', 'figure.svg'], ['exp001-r001-compute', 'figure.svg'],
    ['exp001-r001-present', '../run.json'], ['exp001-r001-present', 'numbers.json']]) {
    await assert.rejects(selectAsset(run, file, { store }), /Invalid run asset/);
  }
});
