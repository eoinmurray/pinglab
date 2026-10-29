import { isRunnableDevEnvironment, type ViteDevServer } from 'vite';
import type { EvaluatedModuleNode } from 'vite/module-runner';

/** A save can notify both Astro's content loader and Vite's file watcher. */
export function coalesceFullReloads(send: (...args: any[]) => void, delay = 200) {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const cancel = () => {
    clearTimeout(timer);
    timer = undefined;
  };
  return {
    cancel,
    send(...args: any[]) {
      const payload = args[0];
      if (payload?.type !== 'full-reload' || payload.ifFallback
          || (payload.path && payload.path !== '*')) return send(...args);
      cancel();
      timer = setTimeout(() => {
        timer = undefined;
        send(payload);
      }, delay);
    },
  };
}

/** Re-evaluate request bindings without recompiling modules or disturbing HMR. */
export function refreshServerMdx(server: ViteDevServer): void {
  for (const name of ['ssr', 'prerender']) {
    const environment = server.environments[name];
    if (!environment || !isRunnableDevEnvironment(environment)) continue;
    const modules = environment.runner.evaluatedModules;
    const invalidated = new Set<EvaluatedModuleNode>();
    const visit = (module: EvaluatedModuleNode) => {
      if (invalidated.has(module)) return;
      invalidated.add(module);
      for (const id of module.importers) {
        const importer = modules.getModuleById(id);
        if (importer) visit(importer);
      }
    };
    for (const module of modules.idToModuleMap.values()) {
      if (module.file?.endsWith('.mdx')) visit(module);
    }
    // Only top-level exports need to run again. Source edits still invalidate
    // Vite's transform graph normally; HTML requests must leave it alone.
    for (const module of invalidated) modules.invalidateModule(module);
  }
}
