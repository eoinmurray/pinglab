// @ts-check
import { defineConfig } from 'astro/config';

import tailwindcss from '@tailwindcss/vite';

import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import mdx from '@astrojs/mdx';

// https://astro.build/config
export default defineConfig({
  site: 'https://pl.eoinmurray.info',

  devToolbar: { enabled: false },

  server: { port: 3000 },

  redirects: {
    '/notebooks': '/',
  },

  integrations: [mdx()],

  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      theme: 'min-light',
    },
  },

  vite: {
    plugins: [tailwindcss()]
  },
});