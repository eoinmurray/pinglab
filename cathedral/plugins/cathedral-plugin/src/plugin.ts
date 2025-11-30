import { type Plugin, type Connect } from 'vite'
import path from 'path'
import fs from 'fs'
import { buildAll } from './lib'
import chokidar from 'chokidar'
import type { IncomingMessage, ServerResponse } from 'http'

export default function contentPlugin(dirs: string[]): Plugin {
  
  if (!dirs) {
    throw new Error('Content directory must be specified.')
  }

  if (dirs && dirs.length === 0) {
    throw new Error('At least one content directory must be specified if "dirs" option is used.')
  }

  const buildFn = () => buildAll(dirs)

  let watchers: chokidar.FSWatcher[] = []

  // Server middleware for serving content files
  const urlToDir = new Map<string, string>()
  dirs.forEach(dir => {
    const dirName = path.basename(dir)
    urlToDir.set(`/${dirName}`, dir)
  })

  const middleware: Connect.NextHandleFunction = (req: IncomingMessage, res: ServerResponse, next: Connect.NextFunction) => {
    // Check if URL matches any registered content directory
    for (const [urlBase, contentDir] of urlToDir.entries()) {
      if (req.url?.startsWith(urlBase + '/')) {
        const relativePath = req.url.slice(urlBase.length + 1)
        const filePath = path.join(contentDir, relativePath)

        if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
          res.setHeader('Access-Control-Allow-Origin', '*')
          const ext = path.extname(filePath).toLowerCase()

          // Set appropriate content types
          const contentTypes: Record<string, string> = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml',
            '.json': 'application/json',
            '.md': 'text/markdown',
            '.yaml': 'text/yaml',
            '.yml': 'text/yaml',
            '.npz': 'application/octet-stream',
          }

          if (contentTypes[ext]) {
            res.setHeader('Content-Type', contentTypes[ext])
          }

          return fs.createReadStream(filePath).pipe(res)
        }
      }
    }
    next()
  }

  return {
    name: 'content',
    async buildStart() {
      await buildFn()
    },
    configureServer(server) {
      // Add middleware for serving content files
      server.middlewares.use(middleware)

      // Watch all content directories and rebuild on changes
      watchers = dirs.map(dir => {
        const watcher = chokidar.watch(dir, {
          ignored: (path: string) => path.endsWith('.cathedral.json'),
          persistent: true,
        })

        watcher.on('change', async (filePath) => {
          const runningFilePath = path.join(dir, '.running')
          if (!fs.existsSync(runningFilePath)) {
            await buildFn()
            server.ws.send({
              type: 'full-reload',
              path: '*',
            })
          }
        })
        
        watcher.on('unlink', async (filePath) => {
          if (path.basename(filePath) === '.running') {
            await buildFn()
            server.ws.send({
              type: 'full-reload',
              path: '*',
            })
          }
        })

        return watcher
      })

    },
    configurePreviewServer(server) {
      // Add middleware for preview server too
      server.middlewares.use(middleware)
    },
    async buildEnd() {
      await Promise.all(watchers.map(w => w.close()))
      watchers = []
    },
  }
}
