// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import react from '@astrojs/react';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// https://astro.build/config
export default defineConfig({
  site: 'https://pinglab.eoinmurray.info',
  trailingSlash: 'never',
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
        cacheDir: `node_modules/.vite/${command === 'serve' ? 'dev' : 'build'}`,
      }),
    }],
  },
  integrations: [mdx(), react(), {
    name: 'pinglab:dev-run-images',
    hooks: {
      'astro:server:setup': ({ server }) => {
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
