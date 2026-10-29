import test from 'node:test';
import assert from 'node:assert/strict';
import { setImmediate } from 'node:timers/promises';
import { deferMount, whenVisibleAfterLoad } from './media-start.ts';

function browser(readyState = 'loading', observers = true) {
  const listeners = new Set<() => void>();
  const timers = new Map<number, () => void>();
  const instances: { callback: Function; disconnected: boolean }[] = [];
  const document = { readyState };
  const platform = {
    document,
    window: {
      addEventListener: (_: string, callback: () => void) => listeners.add(callback),
      removeEventListener: (_: string, callback: () => void) => listeners.delete(callback),
      setTimeout: (callback: () => void) => { const id = timers.size + 1; timers.set(id, callback); return id; },
      clearTimeout: (id: number) => timers.delete(id),
    },
    IntersectionObserver: observers ? class {
      callback: Function;
      disconnected = false;
      constructor(callback: Function) { this.callback = callback; instances.push(this); }
      observe() {}
      disconnect() { this.disconnected = true; }
    } : undefined,
  };
  return {
    platform: platform as any, listeners, timers, instances,
    load() { document.readyState = 'complete'; for (const callback of listeners) callback(); },
    task() { const callbacks = [...timers.values()]; timers.clear(); callbacks.forEach(callback => callback()); },
    visible(value = true) { for (const observer of instances) if (!observer.disconnected) observer.callback([{ isIntersecting: value }]); },
  };
}

test('no activation or animation download until both page load and visibility, including cached pages', () => {
  for (const state of ['loading', 'complete']) {
    const env = browser(state);
    let starts = 0;
    whenVisibleAfterLoad({ isConnected: true } as Element, () => starts++, env.platform);
    env.visible();
    assert.equal(starts, 0);
    if (state === 'loading') env.load();
    assert.equal(starts, 0); // The load event itself remains free of media work.
    env.task(); env.visible(false);
    assert.equal(starts, 0);
    env.visible(); env.visible();
    assert.equal(starts, 1);
  }
});

test('removal cancels load listeners, scheduled tasks and viewport observers; old browsers still wait for load', () => {
  for (const stage of ['before-load', 'after-load', 'observing']) {
    const env = browser();
    let starts = 0;
    const cancel = whenVisibleAfterLoad({ isConnected: true } as Element, () => starts++, env.platform);
    if (stage !== 'before-load') env.load();
    if (stage === 'observing') env.task();
    cancel(); env.load(); env.task(); env.visible();
    assert.equal(starts, 0);
    assert.equal(env.listeners.size, 0);
  }
  const env = browser('loading', false);
  let starts = 0;
  whenVisibleAfterLoad({ isConnected: true } as Element, () => starts++, env.platform);
  assert.equal(starts, 0);
  env.load(); env.task();
  assert.equal(starts, 1);
});

test('an animation removed during its module download never fetches data or creates a canvas', async () => {
  const env = browser('complete');
  let resolve!: (mount: any) => void;
  let downloads = 0, mounts = 0;
  const cancel = deferMount({ isConnected: true } as HTMLElement, () => {
    downloads++;
    return new Promise(done => { resolve = done; });
  }, assert.fail, env.platform);
  assert.equal(downloads, 0);
  env.task(); env.visible();
  assert.equal(downloads, 1);
  cancel();
  resolve(async () => { mounts++; return () => {}; });
  await setImmediate();
  assert.equal(mounts, 0);
});

test('an in-flight mount is disposed exactly once when removed, and startup failures reach its status handler', async () => {
  const env = browser('complete');
  let finish!: (cleanup: () => void) => void;
  let disposed = 0;
  const cancel = deferMount({ isConnected: true } as HTMLElement, async () => async () => new Promise(done => { finish = done; }), assert.fail, env.platform);
  env.task(); env.visible(); await setImmediate();
  cancel(); finish(() => disposed++); await setImmediate(); cancel();
  assert.equal(disposed, 1);
  const error = new Error('Download failed');
  const errors: unknown[] = [];
  deferMount({ isConnected: true } as HTMLElement, async () => { throw error; }, failure => errors.push(failure), env.platform);
  env.task(); env.visible(); await setImmediate();
  assert.deepEqual(errors, [error]);
});
