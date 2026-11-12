import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { RunStore } from './run-store.ts';
import { withRunContext } from './run-context.ts';
import { buildRunIndex, validateRecord } from './run-index.ts';
import { parseCatalog } from './remote-catalog.ts';
const record = (id: string, stage = 'present', created_at = '2026-11-12T10:00:00Z') => ({
  id, experiment: 'exp001', stage, created_at, depends_on: [] as string[], metadata: {},
});
async function fixture(operation: (root: string) => Promise<void>) {
  const root = await mkdtemp(join(tmpdir(), 'pingstore-reader-'));
  try { await operation(root); } finally { await rm(root, { recursive: true, force: true }); }
}
async function save(root: string, value: ReturnType<typeof record>) {
  await mkdir(join(root, value.id), { recursive: true });
  await writeFile(join(root, value.id, 'run.json'), JSON.stringify(value));
}
test('local reader never uses network and refreshes metadata between requests', () => fixture(async root => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => { throw new Error('unexpected network'); };
  try {
    const store = new RunStore({ runsRoot: root });
    await save(root, record('a'));
    await withRunContext(async () => {
      assert.equal(await store.latest('exp001'), 'a');
      await save(root, record('b', 'present', '2026-11-12T11:00:00Z'));
      assert.equal(await store.latest('exp001'), 'a');
    });
    await withRunContext(async () => assert.equal(await store.latest('exp001'), 'b'));
  } finally { globalThis.fetch = original; }
}));
test('newest malformed run fails instead of silently selecting old evidence', () => fixture(async root => {
  await save(root, record('a'));
  await mkdir(join(root, 'b'));
  await writeFile(join(root, 'b/run.json'), '{"id":"b"}');
  await assert.rejects(new RunStore({ runsRoot: root }).latest('exp001'), /Invalid run metadata/);
}));
test('selection ties and dependencies are deterministic, including missing evidence', () => {
  const a = record('a'), b = record('b'); b.depends_on = ['absent'];
  const index = buildRunIndex({ a: { record: a, storage: 'Local' }, b: { record: b, storage: 'Local' } });
  assert.equal(index.selections.exp001, 'b');
  assert.deepEqual(index.runsets.exp001, ['b', 'absent']);
});
test('paths and symlinks cannot escape the store', () => fixture(async root => {
  await save(root, record('one'));
  const store = new RunStore({ runsRoot: root });
  assert.throws(() => store.path('one', '../secret'));
  await symlink('/etc/hosts', join(root, 'one/link'));
  await assert.rejects(store.read('one', 'link'), /symlink/);
}));
test('old metadata and unsafe backup inventories fail clearly', () => {
  assert.throws(() => validateRecord({ schema: 'pingstore.run/v4' }, 'one'), /migrate/);
  assert.throws(() => parseCatalog(JSON.stringify({ runs: { one: { record: record('one'), bytes: 0, files: { '../escape': 0 } } } })));
});
