import { createHash } from 'node:crypto';
import { retryRead } from './remote-http.ts';
import { runStore, RunStore } from './run-store.ts';

export const assetTypes: Record<string, string> = {
  svg: 'image/svg+xml', png: 'image/png', mp4: 'video/mp4',
  webm: 'video/webm', ogv: 'video/ogg', json: 'application/json',
};
export function isRunAsset(file: string): boolean {
  return /^[a-zA-Z0-9_.-]+$/.test(file) &&
    (/\.(png|svg|mp4|webm|ogv)$/.test(file) || ['circuit.json', 'classifier.json', 'training.json', 'lifecycle.json'].includes(file));
}
export function localAssetUrl(run: string, file: string): string {
  return `/run-images/${encodeURIComponent(run)}/${encodeURIComponent(file)}`;
}
interface Options { store?: RunStore; development?: boolean; origin?: string; }

export async function selectAsset(run: string, file: string, options: Options = {}) {
  if (!/^exp\d{3}-r\d+-present$/.test(run) || !isRunAsset(file)) throw new Error('Invalid run asset');
  const store = options.store ?? runStore;
  if (await store.local(run) || store.source === 'local') {
    const bytes = await store.read(run, `export/${file}`);
    return { run, file, source: 'local' as const, url: localAssetUrl(run, file), bytes: bytes.length,
      sha256: createHash('sha256').update(bytes).digest('hex') };
  }
  const entry = await store.entry(run);
  const fingerprint = entry.files[`export/${file}`];
  if (!fingerprint) throw new Error(`Missing catalog asset: ${run}/${file}`);
  // Check actual bytes at build/request time against the verified backup catalog.
  await store.read(run, `export/${file}`);
  const origin = options.origin ?? 'https://pinglab.eoinmurray.info';
  await retryRead(`R2 asset headers ${run}/${file}`, async () => {
    const response = await fetch(store.url(run, `export/${file}`), {
      method: 'HEAD', headers: { Origin: origin }, signal: AbortSignal.timeout(30_000), redirect: 'error',
    });
    if (response.status !== 200 || response.headers.get('content-type')?.split(';')[0].trim() !== assetTypes[file.split('.').at(-1)!]) {
      throw new Error(`R2 asset response is unsuitable: ${run}/${file}`);
    }
    if (file.endsWith('.json') && !options.development && !['*', origin].includes(response.headers.get('access-control-allow-origin') ?? '')) {
      throw new Error(`R2 animation CORS unavailable: ${run}/${file}`);
    }
  });
  return { run, file, source: 'r2' as const, url: store.url(run, `export/${file}`), ...fingerprint };
}
