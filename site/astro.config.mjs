// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// https://astro.build/config
export default defineConfig({
  site: 'https://pinglab.eoinmurray.info',
  server: { port: 3000 },
  devToolbar: { enabled: false },
  vite: { resolve: { dedupe: ['pixi.js'] } },
  integrations: [mdx(), {
    name: 'pinglab:dev-run-images',
    hooks: {
      'astro:server:setup': ({ server }) => {
        const paths = new Map();
        let refresh;
        server.middlewares.use(async (request, response, next) => {
          const pathname = new URL(request.url, 'http://localhost').pathname;
          if (!pathname.startsWith('/run-images/')) return next();
          try {
            const module = await server.ssrLoadModule('/src/pages/run-images/[run]/[file].ts');
            if (!paths.has(pathname)) {
              refresh ??= module.getStaticPaths().then(entries => {
                for (const { params, props } of entries) {
                  paths.set(`/run-images/${encodeURIComponent(params.run)}/${encodeURIComponent(params.file)}`, props);
                }
              }).finally(() => { refresh = undefined; });
              await refresh;
            }
            const props = paths.get(pathname);
            if (!props) return next();
            const result = await module.GET({ props });
            response.statusCode = result.status;
            result.headers.forEach((value, key) => response.setHeader(key, value));
            response.end(Buffer.from(await result.arrayBuffer()));
          } catch (error) {
            next(error);
          }
        });
      },
    },
  }],
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [[rehypeKatex, { throwOnError: true }]],
    }),
  },
});
