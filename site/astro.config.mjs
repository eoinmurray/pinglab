// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import react from '@astrojs/react';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { coalesceFullReloads, refreshServerMdx } from './src/lib/dev-refresh.ts';
import { installRefreshDiagnostics } from './src/lib/dev-refresh-diagnostics.ts';

// Keep the resolver and manifest independent of Vite's bundled module location.
process.env.PINGLAB_REPOSITORY_ROOT = new URL('../', import.meta.url).pathname;

// https://astro.build/config
export default defineConfig({
  logger: { entrypoint: './src/logger.mjs' },
  site: 'https://pinglab.eoinmurray.info',
  trailingSlash: 'never',
  redirects: {
    '/exp113': '/notes/coba-ping-dynamics',
    '/notes/pingstore-rules': '/notes/rules/pingstore',
    '/notes/writing-rules': '/notes/rules/writing',
    '/notes/experiment-rules': '/notes/rules/experiment',
    '/notes/compute-rules': '/notes/rules/compute',
    '/notes/rules/pingstore-rules': '/notes/rules/pingstore',
    '/notes/rules/writing-rules': '/notes/rules/writing',
    '/notes/rules/experiment-rules': '/notes/rules/experiment',
    '/notes/rules/compute-rules': '/notes/rules/compute',
    '/notes/guides/compute': '/notes/rules/compute-access',
    '/notes/guides/experiment': '/notes/rules/experiment-procedures',
    '/notes/guides/pingstore': '/notes/rules/pingstore-operations',
    '/notes/guides/parameter-tables': '/notes/rules/parameter-tables',
  },
  build: { format: 'file' },
  server: { port: 3000 },
  devToolbar: { enabled: false },
  // Article scripts live outside site/, but share its Pixi dependency.
  vite: {
    // Background-server logs change on requests; watching them can feed error
    // recovery reloads back into another request and another log write.
    server: { allowedHosts: ['l.hetzner.eoinmurray.info', 'l0.hetzner.eoinmurray.info'], watch: { ignored: ['**/.astro/dev.log', '**/.astro/dev.json'] } },
    resolve: { dedupe: ['pixi.js'] },
    // Transform the common SSR shell ahead of the first navigation. Warmup does
    // not evaluate article bindings or freeze their scientific run selection.
    environments: Object.fromEntries(['ssr', 'prerender'].map(name => [name, {
      dev: {
        warmup: ['./src/pages/**/*.astro', './src/layouts/**/*.astro'],
        preTransformRequests: true,
      },
    }])),
    // Discovering these only when the search island loads can invalidate its imports.
    optimizeDeps: { include: ['pixi.js', 'cmdk', '@radix-ui/react-dialog'] },
    plugins: [{
      name: 'pinglab:separate-vite-caches',
      config: (_, { command }) => ({
        cacheDir: `node_modules/.vite/${command === 'build' || process.env.NODE_ENV === 'production' ? 'build' : 'dev'}`,
      }),
    }],
  },
  integrations: [mdx(), react(), {
    name: 'pinglab:dev-run-images',
    hooks: {
      'astro:config:setup': ({ addClientDirective, injectScript, command }) => {
        addClientDirective({ name: 'search', entrypoint: './src/scripts/search-hydration.ts' });
        if (command === 'dev') injectScript('page', 'import "/src/scripts/refresh-diagnostics.ts";');
      },
      'astro:server:setup': ({ server }) => {
        // A multi-file save can reach HMR through several Astro/Vite paths.
        const reloads = coalesceFullReloads(server.environments.client.hot.send.bind(server.environments.client.hot));
        server.environments.client.hot.send = reloads.send;
        server.httpServer?.once('close', reloads.cancel);
        installRefreshDiagnostics(server);
        server.middlewares.use(async (request, response, next) => {
          if (!request.headers.accept?.includes('text/html')) return next();
          try {
            const context = await server.environments.ssr.runner.import('/src/lib/run-context.ts');
            // MDX exports execute on import. Refresh server evaluations so their
            // bindings share the new request index without recompiling modules.
            refreshServerMdx(server);
            context.withRunContext(() => next());
          } catch (error) { next(error); }
        });
        server.middlewares.use(async (request, response, next) => {
          const pathname = new URL(request.url, 'http://localhost').pathname;
          if (!pathname.startsWith('/run-images/')) return next();
          try {
            const module = await server.environments.ssr.runner.import('/src/pages/run-images/[run]/[file].ts');
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
