/** Public, checksummed discovery metadata. No R2 credentials reach the site. */
import { createHash, randomUUID } from 'node:crypto';
import { readFile, writeFile, mkdir, rename, unlink } from 'node:fs/promises';
import { dirname, join } from 'node:path';
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
type CatalogCache = { expires: number; value: Promise<Catalog>; etag?: string };
const cached = new Map<string, CatalogCache>();

function catalogCachePath(base: string): string | undefined {
  const root = process.env.PINGLAB_REPOSITORY_ROOT;
  // Astro watches .astro and reloads on any change while recovering from an
  // error. Rendering must never write there, or every reload can trigger another.
  return root ? join(root, 'site/node_modules/.cache/pinglab/catalogs', `catalog-${createHash('sha256').update(base).digest('hex')}.json`) : undefined;
}

async function restoredCatalog(base: string): Promise<CatalogCache | undefined> {
  const path = catalogCachePath(base);
  if (!path) return undefined;
  try {
    const saved = JSON.parse(await readFile(path, 'utf8'));
    if (saved.schema !== 'pinglab.catalog-cache/v1' || saved.base !== base || typeof saved.etag !== 'string'
        || !saved.etag || /[\r\n]/.test(saved.etag) || typeof saved.body !== 'string'
        || createHash('sha256').update(saved.body).digest('hex') !== saved.sha256) return undefined;
    const catalog = validateCatalog(JSON.parse(saved.body));
    // Restarted servers must revalidate this snapshot with R2 before using it.
    return { expires: 0, value: Promise.resolve(catalog), etag: saved.etag };
  } catch { return undefined; }
}

async function persistCatalog(base: string, etag: string | undefined, body: string): Promise<void> {
  const path = catalogCachePath(base);
  if (!path || !etag) return;
  const temporary = `${path}.${process.pid}.${randomUUID()}.tmp`;
  try {
    await mkdir(dirname(path), { recursive: true });
    await writeFile(temporary, JSON.stringify({ schema: 'pinglab.catalog-cache/v1', base, etag, body,
      sha256: createHash('sha256').update(body).digest('hex') }));
    await rename(temporary, path);
  } catch { /* A disposable cache cannot turn valid remote evidence into a failure. */ }
  finally { await unlink(temporary).catch(() => {}); }
}

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

export async function remoteCatalog(base = assetBase(), refresh = false): Promise<Catalog> {
  if (!base) return Promise.reject(new Error('R2 asset origin is disabled'));
  let cachedValue = cached.get(base);
  if (!cachedValue) {
    const restored = await restoredCatalog(base);
    cachedValue = cached.get(base) ?? restored;
  }
  if (!refresh && cachedValue && cachedValue.expires > Date.now()) return cachedValue.value;
  const entry = { expires: process.env.NODE_ENV === 'production' ? Infinity : Date.now() + 30_000,
    value: undefined as unknown as Promise<Catalog>, etag: undefined as string | undefined };
  const value = retryRead('R2 catalog', async () => {
    const response = await fetch(`${base}/pingstore/catalog.json`, {
      signal: AbortSignal.timeout(30_000), cache: 'no-store',
      headers: cachedValue?.etag ? { 'If-None-Match': cachedValue.etag } : {},
    });
    if (response.status === 304) {
      if (!cachedValue?.etag) throw new Error('R2 catalog: unexpected HTTP 304 without a validated catalog');
      entry.etag = cachedValue.etag;
      return cachedValue.value;
    }
    if (!response.ok) throw Object.assign(new Error(`R2 catalog: HTTP ${response.status}`),
      response.status === 404 ? { code: 'ENOENT' } : {});
    const body = await response.text();
    const catalog = validateCatalog(JSON.parse(body));
    entry.etag = response.headers.get('etag') ?? undefined;
    await persistCatalog(base, entry.etag, body);
    return catalog;
  });
  entry.value = value;
  cached.set(base, entry);
  value.catch(() => { if (cached.get(base) === entry) cached.delete(base); });
  return value;
}


export async function remoteControl(base = assetBase()) {
  const unknown = { schema: 'pingstore.control/v1', revision: 0, operation: null, tombstones: {},
    publications: { schema: 'pingstore.publications/v1', known: false, active: null, incoming: {} } };
  try {
    const response = await fetch(`${base}/pingstore/control.json`, { signal: AbortSignal.timeout(30_000), cache: 'no-store' });
    if (response.status === 404) return unknown;
    if (!response.ok) throw new Error(`R2 publication state: HTTP ${response.status}`);
    const control = await response.json();
    if (control?.schema !== 'pingstore.control/v1') throw new Error('Invalid R2 coordination document');
    return control;
  } catch (error) {
    console.warn('[Pingstore] Publication state unavailable; history remains unresolved.', error);
    return unknown;
  }
}
