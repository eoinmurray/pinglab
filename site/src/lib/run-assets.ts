import { createHash } from 'node:crypto';
import { appendFile, readFile, stat } from 'node:fs/promises';
import { resolve } from 'node:path';

export const assetTypes: Record<string, string> = {
  svg: 'image/svg+xml', png: 'image/png', mp4: 'video/mp4',
  webm: 'video/webm', ogv: 'video/ogg', json: 'application/json',
};

export function isRunAsset(file: string): boolean {
  return /^[a-zA-Z0-9_.-]+$/.test(file) &&
    (/\.(png|svg|mp4|webm|ogv)$/.test(file) || ['circuit.json', 'classifier.json', 'training.json'].includes(file));
}

export function localAssetUrl(run: string, file: string): string {
  return `/run-images/${encodeURIComponent(run)}/${encodeURIComponent(file)}`;
}

interface Selection {
  run: string;
  file: string;
  source: 'r2' | 'local';
  url: string;
  bytes: number;
  sha256: string;
  reason?: string;
}

interface Options {
  development?: boolean;
  origin?: string;
  runsRoot?: string;
  baseUrl?: string;
  timeoutMs?: number;
}

const selections = new Map<string, { expires: number; value: Promise<Selection> }>();
let active = 0;
const waiting: Array<() => void> = [];

async function remoteMatches(url: string, bytes: number, sha256: string, file: string, origin: string, timeout: number) {
  if (active >= 6) await new Promise<void>(resume => waiting.push(resume));
  active++;
  try {
    const response = await fetch(url, {
      headers: { Origin: origin, 'Accept-Encoding': 'identity' },
      signal: AbortSignal.timeout(timeout), redirect: 'error',
    });
    let reason: string | undefined;
    if (response.status !== 200) reason = `HTTP ${response.status}`;
    else if (response.headers.get('content-type')?.split(';')[0].trim() !== assetTypes[file.split('.').at(-1)!]) reason = 'content type mismatch';
    else if (file.endsWith('.json') && !['*', origin].includes(response.headers.get('access-control-allow-origin') ?? '')) reason = 'CORS unavailable';
    if (reason) {
      await response.body?.cancel();
      return reason;
    }
    const hash = createHash('sha256');
    let received = 0;
    if (!response.body) return 'empty response';
    for await (const chunk of response.body) {
      received += chunk.length;
      if (received > bytes) return 'size mismatch';
      hash.update(chunk);
    }
    return received === bytes && hash.digest('hex') === sha256 ? undefined : 'checksum mismatch';
  } catch {
    return 'R2 unavailable';
  } finally {
    active--;
    waiting.shift()?.();
  }
}

export async function selectAsset(run: string, file: string, options: Options = {}): Promise<Selection> {
  if (!/^exp\d{3}-r\d+-present$/.test(run) || !isRunAsset(file)) throw new Error('Invalid run asset');
  const path = resolve(options.runsRoot ?? '../.pingstore/runs', run, 'export', file);
  const local = await stat(path);
  if (!local.isFile()) throw new Error(`Missing run asset: ${path}`);
  const base = (options.baseUrl ?? process.env.PINGLAB_ASSET_BASE_URL ?? 'https://assets.eoinmurray.info').replace(/\/$/, '');
  const origin = options.origin ?? 'https://pinglab.eoinmurray.info';
  const key = JSON.stringify([path, base, origin, local.size, local.mtimeMs, !!options.development]);
  const cached = selections.get(key);
  if (cached && cached.expires > Date.now()) return cached.value;
  const value = (async (): Promise<Selection> => {
    const bytes = await readFile(path);
    const sha256 = createHash('sha256').update(bytes).digest('hex');
    const remote = `${base}/pingstore/runs/${encodeURIComponent(run)}/export/${encodeURIComponent(file)}`;
    const reason = base ? await remoteMatches(remote, bytes.length, sha256, file, origin, options.timeoutMs ?? 10_000) : 'R2 disabled';
    const selection: Selection = {
      run, file, source: reason ? 'local' : 'r2', url: reason ? localAssetUrl(run, file) : remote,
      bytes: bytes.length, sha256, ...(reason ? { reason } : {}),
    };
    if (!options.development && process.env.PINGLAB_ASSET_RECEIPT) {
      await appendFile(process.env.PINGLAB_ASSET_RECEIPT, JSON.stringify(selection) + '\n');
    }
    return selection;
  })();
  selections.set(key, { expires: options.development ? Date.now() + 30_000 : Infinity, value });
  try { return await value; }
  catch (error) { selections.delete(key); throw error; }
}
