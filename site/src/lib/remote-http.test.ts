import { expect, test } from 'bun:test';
import { createHash } from 'node:crypto';
import { limitedRemoteRead, retryRead } from './remote-http.ts';
import { RunStore } from './run-store.ts';

test('remote read budget queues requests and releases slots after failure', async () => {
  let release!: () => void;
  const gate = new Promise<void>(done => { release = done; });
  let active = 0, peak = 0, started = 0;
  const reads = Array.from({ length: 18 }, (_, index) => limitedRemoteRead(async () => {
    started++;
    peak = Math.max(peak, ++active);
    try {
      await gate;
      if (index === 0) throw new Error('interrupted read');
    } finally { active--; }
  }));
  expect(started).toBe(6);
  release();
  const results = await Promise.allSettled(reads);
  expect(peak).toBe(6);
  expect(started).toBe(18);
  expect(results.filter(result => result.status === 'rejected')).toHaveLength(1);
});

test('interrupted reads can recover on the fourth attempt', async () => {
  let attempts = 0;
  const value = await retryRead('test timeout', async () => {
    if (++attempts < 4) throw new DOMException('timed out', 'TimeoutError');
    return 'verified';
  });
  expect(value).toBe('verified');
  expect(attempts).toBe(4);
});

test('missing files are not retried', async () => {
  let attempts = 0;
  await expect(retryRead('missing', async () => {
    attempts++;
    throw Object.assign(new Error('missing'), { code: 'ENOENT' });
  })).rejects.toThrow('missing');
  expect(attempts).toBe(1);
});

test('remote reads reject corrupt bytes through every retry', async () => {
  const originalFetch = globalThis.fetch;
  let attempts = 0;
  const run = 'exp001-r001-present';
  const bytes = Buffer.from('verified evidence');
  const store = new RunStore({ source: 'remote', catalog: async () => ({
    schema: 'pingstore.catalog/v1', updated_at: new Date().toISOString(),
    runs: { [run]: { record: {}, files: { 'export/result.svg': {
      bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex'),
    } } } },
  }) });
  globalThis.fetch = (async () => {
    attempts++;
    return new Response('corrupt evidence');
  }) as typeof fetch;
  try {
    await expect(store.read(run, 'export/result.svg')).rejects.toThrow('checksum mismatch');
    expect(attempts).toBe(4);
    globalThis.fetch = (async () => new Response(bytes)) as typeof fetch;
    expect(await store.read(run, 'export/result.svg')).toEqual(bytes);
  } finally { globalThis.fetch = originalFetch; }
});
