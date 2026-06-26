// @ts-check
import { defineConfig } from 'astro/config';

import tailwindcss from '@tailwindcss/vite';

import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import mdx from '@astrojs/mdx';
import react from '@astrojs/react';

// https://astro.build/config
export default defineConfig({
  site: 'https://pl.eoinmurray.info',

  devToolbar: { enabled: false },

  server: { port: 3000 },


  // react() must come before mdx() so MDX inherits React's JSX runtime;
  // the reverse order is what triggers the intermittent "jsxDEV is not a
  // function" / React-in-production-mode error on client:only islands.
  integrations: [react(), mdx()],

  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      themes: {
        light: 'min-light',
        dark: 'github-dark',
      },
      defaultColor: false,
    },
  },

  vite: {
    plugins: [tailwindcss()],
    // Force a single React copy and pre-bundle its jsx runtimes so the
    // client:only island never loads a mismatched (prod) React build.
    resolve: { dedupe: ["react", "react-dom"] },
    optimizeDeps: {
      include: ["react", "react-dom", "react/jsx-runtime", "react/jsx-dev-runtime"],
    },
  },
});