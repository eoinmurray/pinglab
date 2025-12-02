
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import mdx from '@mdx-js/rollup'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import remarkFrontmatter from 'remark-frontmatter'
import remarkGfm from 'remark-gfm'
import remarkMdxFrontmatter from 'remark-mdx-frontmatter'
import path from 'path'
import cathedralPlugin from './plugins/cathedral-plugin/src/plugin.js'
import { cathedralPluginConfig } from './cathedral-plugin.config.js'

export default defineConfig({
  clearScreen: false,
  publicDir: './public', // Disable default public dir - content served via cathedralPlugin
  plugins: [
    // MDX must run before Vite's default transforms
    {
      enforce: 'pre',
      ...mdx({
        remarkPlugins: [
          remarkGfm,
          remarkMath,
          remarkFrontmatter,
          [remarkMdxFrontmatter, { name: 'frontmatter' }],
        ],
        rehypePlugins: [rehypeKatex],
        providerImportSource: '@mdx-js/react',
      }),
    },
    react({ include: /\.(jsx|js|mdx|md|tsx|ts)$/ }),
    cathedralPlugin(cathedralPluginConfig.contentDirs),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@cathedral/vite-plugin': path.resolve(__dirname, './plugins/cathedral-plugin/src'),
      '@biblio': path.resolve(__dirname, '../biblio'),
      // Ensure single React instance for all imports
      'react': path.resolve(__dirname, './node_modules/react'),
      'react-dom': path.resolve(__dirname, './node_modules/react-dom'),
      // MDX runtime for files outside project root
      '@mdx-js/react': path.resolve(__dirname, './node_modules/@mdx-js/react'),
    },
  },
  server: {
    host: '0.0.0.0',
    port: process.env.PORT ? parseInt(process.env.PORT, 10) : 3000,
    strictPort: true,
    fs: {
      allow: ['..', '../..'],
    },
    allowedHosts: true,
  },
  preview: {
    allowedHosts: [
      'pinglab.eoinmurray.info',
      'pl.eoinmurray.info',
    ],
  },
})
