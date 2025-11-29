
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import cathedralPlugin from './plugins/cathedral-plugin/src/plugin.js'
import { cathedralPluginConfig } from './cathedral-plugin.config.js'
import mdx from '@mdx-js/rollup'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import remarkFrontmatter from 'remark-frontmatter'
import remarkMdxFrontmatter from 'remark-mdx-frontmatter'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

export default defineConfig({
  clearScreen: false,
  publicDir: './public', // Disable default public dir - content served via cathedralPlugin
  plugins: [
    react(),
    cathedralPlugin(cathedralPluginConfig.contentDirs),
    mdx({
      remarkPlugins: [remarkGfm, remarkFrontmatter, remarkMdxFrontmatter, remarkMath],
      rehypePlugins: [rehypeHighlight, rehypeKatex],
    })
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@cathedral/vite-plugin': path.resolve(__dirname, './plugins/cathedral-plugin/src'),
    },
  },
  server: {
    host: '0.0.0.0',
    port: process.env.PORT ? parseInt(process.env.PORT, 10) : 3000,
    strictPort: true,
    fs: {
      allow: ['..', '../..'],
    },
  },
  preview: {
    allowedHosts: [
      'pinglab.eoinmurray.info',
      'pl.eoinmurray.info',
    ],
  },
})
