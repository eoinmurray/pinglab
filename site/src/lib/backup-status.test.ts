import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { runBackupStatus, type BackupInventory } from './backup-status.ts';

test('distinguishes complete, missing, partial, mismatched and unavailable backups', async () => {
  const root = await mkdtemp(join(tmpdir(), 'backup-status-'));
  const run = 'exp001-r001-compute';
  const directory = join(root, run);
  const inventory = (files: [string, number][]): BackupInventory => ({ files: new Map(files), checkedAt: '2026-10-23T14:00:00Z' });
  const complete: [string, number][] = [[`${run}/run.json`, 2], [`${run}/README.md`, 4], [`${run}/export/data.bin`, 3]];
  try {
    await mkdir(join(directory, 'export'), { recursive: true });
    await writeFile(join(directory, 'run.json'), '{}');
    await writeFile(join(directory, 'README.md'), 'test');
    await writeFile(join(directory, 'export/data.bin'), 'abc');
    await writeFile(join(directory, '.protected'), 'local');
    assert.equal((await runBackupStatus(run, inventory(complete), root)).label, 'Yes');
    assert.equal((await runBackupStatus(run, inventory([...complete, [`${run}/.protected`, 20]]), root)).label, 'Yes');
    assert.equal((await runBackupStatus(run, inventory([]), root)).label, 'No');
    assert.equal((await runBackupStatus(run, inventory(complete.slice(1)), root)).label, 'Partial');
    assert.equal((await runBackupStatus(run, inventory(complete.slice(0, 2)), root)).label, 'Partial');
    assert.equal((await runBackupStatus(run, inventory([...complete, [`${run}/extra`, 1]]), root)).label, 'Partial');
    assert.equal((await runBackupStatus(run, inventory(complete.map(([path, size]) => [path, size + 1])), root)).label, 'Partial');
    assert.equal((await runBackupStatus(run, undefined, root)).label, 'Unknown');
    assert.equal((await runBackupStatus(run, inventory(complete), join(root, 'missing'))).label, 'Yes');
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
