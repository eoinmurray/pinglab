import { readdir, stat } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { remoteCatalog } from './remote-catalog.ts';

export type BackupInventory = { files: Map<string, number>; checkedAt: string };
export type BackupStatus = { label: 'Yes' | 'No' | 'Partial' | 'Unknown'; detail: string };
export async function backupInventory(): Promise<BackupInventory | undefined> {
  if (process.env.PINGLAB_RUN_SOURCE === 'local') return undefined;
  try {
    const catalog = await remoteCatalog();
    return { checkedAt: catalog.updated_at, files: new Map(Object.entries(catalog.runs).flatMap(([id, entry]) =>
      Object.entries(entry.files).map(([path, file]) => [`${id}/${path}`, file.bytes] as [string, number]))) };
  } catch { return undefined; }
}
export async function runBackupStatus(run: string, inventory: BackupInventory | undefined,
  runsRoot = resolve(process.env.PINGLAB_RUNS_ROOT ?? '../.pingstore/runs')): Promise<BackupStatus> {
  if (!inventory) return { label: 'Unknown', detail: 'The verified R2 catalog could not be read.' };
  const remote = new Map([...inventory.files].filter(([path]) => path.startsWith(`${run}/`) && path !== `${run}/.protected`));
  const checked = `Catalog published ${inventory.checkedAt}.`;
  if (!remote.size) return { label: 'No', detail: `Run is not in the verified R2 catalog. ${checked}` };
  if (!remote.has(`${run}/run.json`)) return { label: 'Partial', detail: `R2 completion record is missing. ${checked}` };
  try {
    const local = new Map<string, number>();
    async function visit(directory: string, prefix: string) {
      for (const entry of await readdir(directory, { withFileTypes: true })) {
        if (prefix === run && entry.name === '.protected') continue;
        const path = join(directory, entry.name);
        const key = `${prefix}/${entry.name}`;
        if (entry.isDirectory()) await visit(path, key);
        else if (entry.isFile()) local.set(key, (await stat(path)).size);
        else throw new Error('Unsupported run entry');
      }
    }
    if ((process.env.PINGLAB_RUN_SOURCE ?? (process.env.NODE_ENV === 'production' ? 'remote' : 'auto')) === 'remote') {
      return { label: 'Yes', detail: `Run is in the byte-verified R2 catalog; local storage is not required. ${checked}` };
    }
    await visit(join(runsRoot, run), run);
    if (local.size !== remote.size || [...local].some(([path, size]) => remote.get(path) !== size)) {
      return { label: 'Partial', detail: `Catalog file paths or sizes differ from the local run. ${checked}` };
    }
    return { label: 'Yes', detail: `Verified backup catalog paths and sizes match the local run. Local bytes have not been rehashed for this display. ${checked}` };
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return { label: 'Yes', detail: `Run is in the verified R2 catalog and has no local copy. ${checked}` };
    }
    return { label: 'Unknown', detail: `Local run files could not be compared. ${checked}` };
  }
}
