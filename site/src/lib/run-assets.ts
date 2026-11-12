import { runStore, RunStore } from './run-store.ts';
import { runPattern } from './run-index.ts';
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
  if (!runPattern.test(run) || !isRunAsset(file)) throw new Error('Invalid run asset');
  const store = options.store ?? runStore;
  if (!await store.has(run, `export/${file}`)) throw new Error(`Missing run asset: ${run}/${file}`);
  return { run, file, source: await store.local(run) || store.source === 'local' ? 'local' as const : 'r2' as const,
    url: await store.local(run) || store.source === 'local' ? localAssetUrl(run, file) : store.url(run, `export/${file}`) };
}
