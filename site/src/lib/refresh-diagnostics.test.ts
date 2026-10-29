import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { setImmediate } from 'node:timers/promises';
import { createServer } from 'vite';
import { createRefreshLog, installRefreshDiagnostics, refreshDiagnosticsPath } from './dev-refresh-diagnostics.ts';
import { createRefreshRecorder } from './refresh-recorder.ts';

test('diagnostics rotate to one previous file and retain server identities across restarts', () => {
  const directory = mkdtempSync(join(tmpdir(), 'refresh-log-'));
  try {
    const first = createRefreshLog(directory, 1);
    first.write('server-start');
    first.write('server-reload-request', { channel: 'ws' });
    assert.equal(JSON.parse(readFileSync(`${first.path}.1`, 'utf8')).event, 'server-start');
    const second = createRefreshLog(directory);
    second.write('server-start');
    const events = second.tail().trim().split('\n').map(line => JSON.parse(line));
    assert.deepEqual(events.map(event => event.event), ['server-start', 'server-reload-request', 'server-start']);
    assert.notEqual(events[0].server, events[2].server);
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

test('connection loss and suspension evidence survive a reload while offline, then drain after reconnection', async () => {
  const values = new Map<string, string>();
  const storage = { getItem: (key: string) => values.get(key) ?? null, setItem: (key: string, value: string) => { values.set(key, value); } };
  const sent: any[] = [];
  let online = false, time = 1_000, sequence = 0;
  const platform = {
    storage, path: '/exp133', now: () => time, id: () => String(++sequence),
    send: async (events: any[]) => { if (!online) throw new Error('server asleep'); sent.push(...events); return true; },
  };
  const first = createRefreshRecorder(platform);
  first.record('vite-disconnect');
  await setImmediate();
  time += 300_000;
  first.tick();
  await setImmediate();
  const second = createRefreshRecorder(platform);
  second.record('boot', { navigation: 'reload' });
  await setImmediate();
  online = true;
  await second.flush();
  assert.deepEqual(sent.map(event => event.event), ['browser-vite-disconnect', 'browser-timer-gap', 'browser-boot']);
  assert.equal(sent[1].details.gapMs, 300_000);
  assert.notEqual(sent[0].document, sent[2].document);
  assert.equal(sent[0].tab, sent[2].tab);
  assert.deepEqual(JSON.parse([...values.values()][0]).events, []);
});

test('blocked storage, bounded offline queues and overlapping sends do not interrupt recording', async () => {
  const sent: any[] = [];
  let resolve!: (ok: boolean) => void;
  let sequence = 0;
  const recorder = createRefreshRecorder({
    storage: { getItem() { throw new Error('blocked'); }, setItem() { throw new Error('blocked'); } },
    path: '/exp133', id: () => String(++sequence),
    send: events => { sent.push(events); return new Promise(done => { resolve = done; }); },
  });
  recorder.record('boot');
  for (let index = 0; index < 150; index++) recorder.record('visibility', { index });
  assert.equal(sent.length, 1);
  resolve(true);
  await setImmediate();
  const flush = recorder.flush();
  assert.equal(sent[1].length, 30);
  assert.equal(sent[1][0].details.index, 50);
  resolve(true);
  await flush;
});

test('live diagnostics accept bounded same-origin events and preserve both reload transports', async () => {
  const root = mkdtempSync(join(tmpdir(), 'refresh-server-'));
  const delivered: any[] = [];
  const originalHot = (...args: any[]) => delivered.push(['client', ...args]);
  const originalWs = (...args: any[]) => delivered.push(['ws', ...args]);
  const server = await createServer({
    root, configFile: false, logLevel: 'silent', server: { port: 0 },
    plugins: [{ name: 'test-diagnostics', configureServer(server) {
      server.environments.client.hot.send = originalHot;
      server.ws.send = originalWs;
      installRefreshDiagnostics(server);
    } }],
  });
  try {
    await server.listen();
    const address = server.httpServer!.address() as { port: number };
    const origin = `http://localhost:${address.port}`;
    const url = origin + refreshDiagnosticsPath;
    const event = { id: 'offline-event', event: 'browser-vite-disconnect', time: '2026-10-29T00:00:00Z' };
    const post = (events: unknown, extra: Record<string, string> = {}) => fetch(url, {
      method: 'POST', headers: { 'Content-Type': 'application/json', Origin: origin, ...extra }, body: JSON.stringify(events),
    });
    assert.equal((await post([event])).status, 204);
    assert.equal((await post([event])).status, 204);
    assert.equal((await post([event], { Origin: 'https://unrelated.example' })).status, 403);
    assert.equal((await post([{ ...event, event: 'unexpected' }])).status, 400);
    assert.equal((await post('x'.repeat(33 * 1024))).status, 413);
    server.environments.client.hot.send({ type: 'full-reload', path: '*' });
    server.ws.send({ type: 'full-reload' });
    assert.equal(delivered.length, 2);
    const response = await fetch(url);
    assert.equal(response.headers.get('cache-control'), 'no-store');
    const events = (await response.text()).trim().split('\n').map(line => JSON.parse(line));
    assert.equal(events.filter(event => event.event === 'browser-vite-disconnect').length, 1);
    const reloads = events.filter(event => event.event === 'server-reload-request');
    assert.equal(reloads.length, 2);
    assert.ok(reloads.every(event => ['client', 'ws'].includes(event.channel)));
    assert.ok(reloads.every(event => event.origin.includes('refresh-diagnostics.test.ts')));
    // Vite's default node_modules exclusion also covers our rotating log files.
    assert.ok(!Object.keys(server.watcher.getWatched()).some(path => path.includes('/node_modules/')));
  } finally {
    await server.close();
    await setImmediate();
    assert.strictEqual(server.ws.send, originalWs);
    rmSync(root, { recursive: true, force: true });
  }
});

test('the browser observer is injected only for development', async () => {
  const { default: config } = await import('../../astro.config.mjs');
  const integration = config.integrations.find((item: any) => item?.name === 'pinglab:dev-run-images');
  for (const command of ['dev', 'build', 'preview', 'sync']) {
    const scripts: string[] = [];
    integration.hooks['astro:config:setup']({
      command, addClientDirective() {}, injectScript: (_stage: string, code: string) => scripts.push(code),
    });
    assert.equal(scripts.length, command === 'dev' ? 1 : 0);
  }
});
