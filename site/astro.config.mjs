// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import react from '@astrojs/react';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// Keep the resolver and manifest independent of Vite's bundled module location.
process.env.PINGLAB_REPOSITORY_ROOT = new URL('../', import.meta.url).pathname;

// https://astro.build/config
export default defineConfig({
  logger: { entrypoint: './src/logger.mjs' },
  site: 'https://pinglab.eoinmurray.info',
  trailingSlash: 'never',
  redirects: { '/exp113': '/notes/coba-ping-dynamics' },
  build: { format: 'file' },
  server: { port: 3000 },
  devToolbar: { enabled: false },
  // Article scripts live outside site/, but share its Pixi dependency.
  vite: {
    resolve: { dedupe: ['pixi.js'] },
    // Discovering these only when the search island loads can invalidate its imports.
    optimizeDeps: { include: ['pixi.js', 'cmdk', '@radix-ui/react-dialog'] },
    plugins: [{
      name: 'pinglab:separate-vite-caches',
      config: (_, { command }) => ({
        cacheDir: `node_modules/.vite/${command === 'build' || process.env.NODE_ENV === 'production' ? 'build' : 'dev'}`,
      }),
    }, {
      name: 'pinglab:invalidate-article-metrics',
      configureServer(server) {
        const metrics = new URL('./src/lib/article-metrics.ts', import.meta.url).pathname;
        const roots = ['../experiments/', '../md/', './src/'].map(path => new URL(path, import.meta.url).pathname);
        const invalidate = (_event, path) => {
          if (!roots.some(root => path.startsWith(root))) return;
          for (const module of server.moduleGraph.getModulesByFile(metrics) ?? []) {
            server.moduleGraph.invalidateModule(module);
          }
        };
        server.watcher.on('all', invalidate);
        server.httpServer?.once('close', () => server.watcher.off('all', invalidate));
      },
    }],
  },
  integrations: [mdx(), react(), {
    name: 'pinglab:dev-run-images',
    hooks: {
      'astro:config:setup': ({ addClientDirective }) => {
        addClientDirective({ name: 'search', entrypoint: './src/scripts/search-hydration.ts' });
      },
      'astro:server:setup': ({ server }) => {
        server.middlewares.use(async (request, response, next) => {
          if (!request.headers.accept?.includes('text/html')) return next();
          try {
            const context = await server.ssrLoadModule('/src/lib/run-context.ts');
            // MDX exports execute on import. Invalidate their importing graph so
            // bindings and components render against the same new request index.
            for (const module of server.moduleGraph.idToModuleMap.values()) {
              if (module.file?.endsWith('.mdx')) server.moduleGraph.invalidateModule(module);
            }
            context.withRunContext(() => next());
          } catch (error) { next(error); }
        });
        server.middlewares.use(async (request, response, next) => {
          const pathname = new URL(request.url, 'http://localhost').pathname;
          if (!pathname.startsWith('/run-images/')) return next();
          try {
            const module = await server.ssrLoadModule('/src/pages/run-images/[run]/[file].ts');
            const result = await module.serveRunAsset(pathname, `http://${request.headers.host ?? 'localhost:3000'}`);
            response.statusCode = result.status;
            result.headers.forEach((value, key) => response.setHeader(key, value));
            response.end(request.method === 'HEAD' ? undefined : Buffer.from(await result.arrayBuffer()));
          } catch (error) {
            next(error);
          }
        });
      },
    },
  }],
  markdown: {
    shikiConfig: { themes: { light: 'github-light', dark: 'github-dark' } },
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [[rehypeKatex, { throwOnError: true }]],
    }),
  },
});
