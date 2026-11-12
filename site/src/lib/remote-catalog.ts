/** An optional, rebuildable backup inventory. Contains no retention state. */
import { runPattern, validateRecord, type RunRecord } from './run-index.ts';
export { runPattern };
export const safePath = (path: string) => !!path && !path.includes('\\') && !path.includes('\0')
  && path.split('/').every(part => !!part && part !== '.' && part !== '..');
export interface CatalogEntry { record: RunRecord; bytes: number; files: Record<string, number>; completed_at: string; }
export interface Catalog { updated_at: string; runs: Record<string, CatalogEntry>; }
export const assetBase = () => process.env.PINGLAB_ASSET_BASE ?? 'https://assets.eoinmurray.info';
export function parseCatalog(text: string): Catalog {
  const value = JSON.parse(text);
  if (!value || !value.runs || typeof value.runs !== 'object') throw new Error('Invalid backup inventory');
  for (const [id, entry] of Object.entries(value.runs) as [string, CatalogEntry][]) {
    validateRecord(entry.record, id);
    if (!entry.files || typeof entry.files !== 'object' || !Number.isFinite(entry.bytes) || entry.bytes < 0
        || !Number.isFinite(Date.parse(entry.completed_at))
        || Object.entries(entry.files).some(([path, bytes]) => !safePath(path) || !Number.isSafeInteger(bytes) || bytes < 0)) {
      throw new Error(`Invalid backup inventory entry: ${id}`);
    }
  }
  return value;
}
export async function remoteCatalog(base = assetBase()): Promise<Catalog> {
  const response = await fetch(`${base.replace(/\/$/, '')}/pingstore-v2/index.json`, { signal: AbortSignal.timeout(10_000) });
  if (!response.ok) throw new Error(`Backup inventory: HTTP ${response.status}`);
  return parseCatalog(await response.text());
}
