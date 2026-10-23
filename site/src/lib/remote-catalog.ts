/** Public, checksummed discovery metadata. No R2 credentials reach the site. */
import { createHash } from 'node:crypto';
import { retryRead } from './remote-http.ts';
export interface Fingerprint { bytes: number; sha256: string; }
export interface CatalogRun { record: any; files: Record<string, Fingerprint>; }
export interface Catalog { schema: 'pingstore.catalog/v1'; updated_at: string; runs: Record<string, CatalogRun>; }
export const assetBase = () => (process.env.PINGLAB_ASSET_BASE_URL ?? 'https://assets.eoinmurray.info').replace(/\/$/, '');
export function safePath(path: string): boolean {
  return !!path && !/[\\\0\r\n]/.test(path) && path.split('/').every(part => !!part && part !== '.' && part !== '..');
}
export const runPattern = /^(exp\d{3})-r\d+-(compute(?:-phase-[a-z][a-z0-9-]*)?|analyse|present)$/;
// Match pathlib.Path ordering used by Pingstore's canonical payload inventory.
export function pathCompare(a: string, b: string): number {
  const left = a.split('/'), right = b.split('/');
  for (let index = 0; index < Math.min(left.length, right.length); index++) {
    if (left[index] !== right[index]) return left[index] < right[index] ? -1 : 1;
  }
  return left.length - right.length;
}
const cached = new Map<string, { expires: number; value: Promise<Catalog> }>();

export function validateCatalog(value: any): Catalog {
  if (value?.schema !== 'pingstore.catalog/v1' || !value.runs || typeof value.runs !== 'object'
      || Array.isArray(value.runs) || !Number.isFinite(Date.parse(value.updated_at))) throw new Error('Invalid R2 catalog');
  for (const [id, entry] of Object.entries(value.runs) as [string, CatalogRun][]) {
    const match = runPattern.exec(id);
    const record = entry?.record;
    if (!match || record?.run_id !== id || record.schema !== 'pingstore.run/v4'
        || record.experiment !== match[1] || record.stage !== match[2].split('-')[0]
        || !Number.isFinite(Date.parse(record.execution?.completed_at)) || !record.inputs
        || !/^sha256:[a-f0-9]{64}$/.test(record.payload_digest)
        || !entry.files?.['run.json'] || !entry.files?.['README.md']) throw new Error(`Invalid catalog run: ${id}`);
    for (const [path, file] of Object.entries(entry.files)) {
      if (!safePath(path) || !Number.isSafeInteger(file.bytes) || file.bytes < 0
          || !/^[a-f0-9]{64}$/.test(file.sha256)) throw new Error(`Invalid catalog file: ${id}/${path}`);
    }
    const payload = Object.entries(entry.files).filter(([path]) => path.startsWith('export/'))
      .sort(([a], [b]) => pathCompare(a, b))
      .map(([path, file]) => ({ path, sha256: file.sha256, size_bytes: file.bytes }));
    const encoded = JSON.stringify(payload).replace(/[\u007f-\uffff]/g, character =>
      `\\u${character.charCodeAt(0).toString(16).padStart(4, '0')}`);
    if (`sha256:${createHash('sha256').update(encoded).digest('hex')}` !== record.payload_digest) {
      throw new Error(`Catalog payload inventory mismatch: ${id}`);
    }
    for (const reference of Object.values(record.inputs) as any[]) {
      if (!value.runs[reference.run_id] || value.runs[reference.run_id].record.payload_digest !== reference.payload_digest) {
        throw new Error(`Missing or changed catalog ancestor: ${id}/${reference.run_id}`);
      }
    }
  }
  const visiting = new Set<string>();
  const visited = new Set<string>();
  const visit = (id: string) => {
    if (visiting.has(id)) throw new Error('Catalog inputs contain a cycle');
    if (visited.has(id)) return;
    visiting.add(id);
    for (const reference of Object.values(value.runs[id].record.inputs) as any[]) visit(reference.run_id);
    visiting.delete(id);
    visited.add(id);
  };
  for (const id of Object.keys(value.runs)) visit(id);
  return value;
}

export function remoteCatalog(base = assetBase()): Promise<Catalog> {
  if (!base) return Promise.reject(new Error('R2 asset origin is disabled'));
  const cachedValue = cached.get(base);
  if (cachedValue && cachedValue.expires > Date.now()) return cachedValue.value;
  const value = retryRead('R2 catalog', async () => {
    const response = await fetch(`${base}/pingstore/catalog.json`, { signal: AbortSignal.timeout(30_000), cache: 'no-store' });
    if (!response.ok) throw Object.assign(new Error(`R2 catalog: HTTP ${response.status}`),
      response.status === 404 ? { code: 'ENOENT' } : {});
    return validateCatalog(await response.json());
  });
  cached.set(base, { expires: process.env.NODE_ENV === 'production' ? Infinity : Date.now() + 30_000, value });
  value.catch(() => cached.delete(base));
  return value;
}
