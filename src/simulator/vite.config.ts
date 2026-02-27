import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

// https://vite.dev/config/
export default defineConfig(({ command }) => ({
  clearScreen: false,
  // In local dev, serve simulator assets under /simulator so root (/) can be proxied to Veslx cleanly.
  base: command === 'serve' ? '/simulator/' : '/',
  plugins: [
    react(),
    tailwindcss(),
    {
      name: 'simulator-trailing-slash-redirect',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (req.url === '/simulator') {
            res.statusCode = 302
            res.setHeader('Location', '/simulator/')
            res.end()
            return
          }
          if (req.url === '/streamlit') {
            res.statusCode = 302
            res.setHeader('Location', '/streamlit/')
            res.end()
            return
          }
          next()
        })
      },
    },
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    strictPort: true,
    proxy: {
      '^/streamlit(?:/|$).*': {
        target: 'http://localhost:3002',
        changeOrigin: true,
        ws: true,
        timeout: 30000,
        proxyTimeout: 30000,
      },
      // Match Pages routing:
      // - simulator app at /simulator/*
      // - streamlit app at /streamlit/*
      // - all other routes served by Veslx at :3001.
      '^/(?!simulator(?:/|$)|streamlit(?:/|$)).*': {
        target: 'http://localhost:3001',
        changeOrigin: true,
        // Veslx dev relies on Vite client channels; Firefox is more sensitive
        // to failed upgrade/long-lived proxy connections than Chromium.
        ws: true,
        timeout: 30000,
        proxyTimeout: 30000,
      },
    },
  },
}))
