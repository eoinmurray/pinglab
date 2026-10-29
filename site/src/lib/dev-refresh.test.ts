import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, realpath, writeFile, rm } from 'node:fs/promises';
import { once } from 'node:events';
import { setTimeout as wait } from 'node:timers/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'vite';
import { coalesceFullReloads, refreshServerMdx } from './dev-refresh.ts';
import config from '../../astro.config.mjs';

test('background log and lock writes cannot trigger error recovery, while source edits remain watched', { timeout: 5000 }, async () => {
  const root = await mkdtemp(join(tmpdir(), 'background-watch-'));
  await mkdir(join(root, '.astro'));
  const source = join(root, 'source.js');
  await writeFile(source, 'export const value = 1;');
  for (const file of ['dev.log', 'dev.json']) await writeFile(join(root, '.astro', file), 'initial');
  const server = await createServer({
    root, configFile: false, logLevel: 'silent',
    server: { ...config.vite.server, middlewareMode: true },
  });
  const changes: string[] = [];
  server.watcher.on('all', (_event, file) => changes.push(file));
  try {
    await once(server.watcher, 'ready');
    for (let index = 0; index < 3; index++) {
      for (const file of ['dev.log', 'dev.json']) await writeFile(join(root, '.astro', file), `request ${index}`);
      await wait(100);
    }
    assert.deepEqual(changes, [], 'runtime writes must never reach Astro recovery listeners');
    const observed = once(server.watcher, 'change');
    await writeFile(source, 'export const value = 2;');
    await observed;
    assert.deepEqual(changes, [await realpath(source)]);
  } finally {
    await server.close();
    await rm(root, { recursive: true, force: true });
  }
});

test('one editing burst sends one reload, while errors and targeted messages pass immediately', context => {
  context.mock.timers.enable({ apis: ['setTimeout'] });
  const messages: unknown[] = [];
  const queue = coalesceFullReloads((...args) => messages.push(args));
  queue.send({ type: 'full-reload' });
  context.mock.timers.tick(125);
  queue.send({ type: 'full-reload', path: '*' });
  queue.send({ type: 'error', err: { message: 'broken source' } });
  queue.send('custom:event', { value: 1 });
  queue.send({ type: 'full-reload', path: '/only.html' });
  queue.send({ type: 'full-reload', ifFallback: true });
  assert.equal(messages.length, 4);
  context.mock.timers.tick(199);
  assert.equal(messages.length, 4);
  context.mock.timers.tick(1);
  assert.deepEqual(messages.at(-1), [{ type: 'full-reload', path: '*' }]);
  assert.equal(messages.length, 5);
  // A later edit still reloads, and closing/restarting cancels pending work.
  queue.send({ type: 'full-reload' });
  context.mock.timers.tick(200);
  assert.equal(messages.length, 6);
  queue.send({ type: 'full-reload' });
  queue.cancel();
  context.mock.timers.tick(200);
  assert.equal(messages.length, 6);
});

test('request refresh re-evaluates MDX and its server importers while preserving server and client transforms', async () => {
  const root = await mkdtemp(join(tmpdir(), 'mdx-refresh-'));
  const values = join(root, 'numbers.txt');
  await writeFile(values, '1');
  await writeFile(join(root, 'article.mdx'), `
    import { readFileSync } from 'node:fs';
    export const score = Number(readFileSync(${JSON.stringify(values)}, 'utf8'));
  `);
  await writeFile(join(root, 'page.js'), "export { score } from './article.mdx';");
  const server = await createServer({
    root, configFile: false, logLevel: 'silent',
    server: { middlewareMode: true, watch: null },
    environments: { prerender: { consumer: 'server' } },
    plugins: [{ name: 'test-mdx', transform: (code, id) => id.endsWith('.mdx') ? code : undefined }],
  });
  try {
    const client = server.environments.client;
    await client.transformRequest('/article.mdx');
    const module = (await client.moduleGraph.getModuleByUrl('/article.mdx'))!;
    const transform = module.transformResult;
    const timestamp = module.lastInvalidationTimestamp;
    let reloads = 0;
    client.hot.send = () => { reloads++; };
    for (const name of ['ssr', 'prerender']) {
      const environment = server.environments[name] as any;
      assert.equal((await environment.runner.import('/page.js')).score, 1);
    }
    const serverTransforms = ['ssr', 'prerender'].flatMap(name =>
      [...server.environments[name].moduleGraph.idToModuleMap.values()]
        .filter(module => module.file?.endsWith('.mdx') || module.file?.endsWith('/page.js'))
        .map(module => ({ module, transform: module.transformResult, timestamp: module.lastInvalidationTimestamp })),
    );
    await writeFile(values, '2');
    refreshServerMdx(server);
    for (const name of ['ssr', 'prerender']) {
      const environment = server.environments[name] as any;
      assert.equal((await environment.runner.import('/page.js')).score, 2);
    }
    assert.strictEqual(module.transformResult, transform);
    assert.equal(module.lastInvalidationTimestamp, timestamp);
    assert.equal(reloads, 0);
    for (const { module, transform, timestamp } of serverTransforms) {
      assert.strictEqual(module.transformResult, transform, `${module.id} should not be recompiled on navigation`);
      assert.equal(module.lastInvalidationTimestamp, timestamp);
    }
    // Unchanged requests still refresh bindings without creating HMR traffic.
    refreshServerMdx(server);
    assert.equal((await (server.environments.ssr as any).runner.import('/page.js')).score, 2);
    assert.equal(reloads, 0);
  } finally {
    await server.close();
    await rm(root, { recursive: true, force: true });
  }
});
