/** A page request shares one selection even across separately loaded MDX modules. */
import { AsyncLocalStorage } from 'node:async_hooks';

// Vite may reload modules while requests overlap. Keep their contexts isolated
// and share the storage instance across module generations.
const key = Symbol.for('pinglab.run-context');
const global = globalThis as any;
const context: AsyncLocalStorage<Map<symbol, Promise<unknown>>> = global[key] ??= new AsyncLocalStorage();
export function withRunContext<T>(operation: () => T): T {
  return context.run(new Map(), operation);
}
export function requestValue<T>(key: symbol, load: () => Promise<T>): Promise<T> {
  const values = context.getStore();
  if (!values) return load();
  if (!values.has(key)) values.set(key, load());
  return values.get(key)! as Promise<T>;
}
