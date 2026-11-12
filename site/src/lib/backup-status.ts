import { remoteCatalog } from './remote-catalog.ts';
export type BackupInventory = Awaited<ReturnType<typeof remoteCatalog>>;
export type BackupStatus = { label: 'Yes' | 'No' | 'Unknown'; detail: string };
export async function backupInventory(): Promise<BackupInventory | undefined> {
  if (!process.env.PINGLAB_RUN_SOURCE || process.env.PINGLAB_RUN_SOURCE === 'local') return undefined;
  return remoteCatalog().catch(() => undefined);
}
export async function runBackupStatus(run: string, inventory: BackupInventory | undefined): Promise<BackupStatus> {
  if (!inventory) return { label: 'Unknown', detail: 'R2 inventory was not loaded.' };
  return inventory.runs[run]
    ? { label: 'Yes', detail: `Upload completed ${inventory.runs[run].completed_at}. No read-back audit is implied.` }
    : { label: 'No', detail: 'No completed upload in this inventory snapshot.' };
}
