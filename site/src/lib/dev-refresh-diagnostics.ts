import { appendFileSync, existsSync, mkdirSync, readFileSync, renameSync, statSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import { join, relative } from 'node:path';
import type { ViteDevServer } from 'vite';

export const refreshDiagnosticsPath = '/__pinglab/dev-refresh';
const maxBytes = 512 * 1024;

/** Keep diagnostic writes outside Astro's watched content and generated files. */
export function createRefreshLog(directory: string, limit = maxBytes) {
  mkdirSync(directory, { recursive: true });
  const path = join(directory, 'dev-refresh.jsonl');
  const server = randomUUID();
  let warned = false;
  return {
    path, server,
    write(event: string, details: Record<string, unknown> = {}) {
      try {
        if (existsSync(path) && statSync(path).size >= limit) renameSync(path, `${path}.1`);
        appendFileSync(path, `${JSON.stringify({ time: new Date().toISOString(), server, event, ...details })}\n`);
      } catch (error) {
        if (!warned) console.warn('[refresh-debug] Cannot write diagnostics:', error);
        warned = true;
      }
    },
    tail() {
      return [ `${path}.1`, path ].filter(existsSync)
        .flatMap(file => readFileSync(file, 'utf8').trim().split('\n'))
        .filter(Boolean).slice(-200).join('\n') + '\n';
    },
  };
}

export function installRefreshDiagnostics(server: ViteDevServer) {
  const log = createRefreshLog(join(server.config.root, 'node_modules/.cache/pinglab'));
  const localPath = (file: string) => relative(server.config.root, file);
  log.write('server-start', { pid: process.pid });
  console.info(`[refresh-debug] Recording reloads and sleep/wake events to ${log.path}`);

  // Astro sometimes sends directly through ws; Vite and content loaders use hot.
  // Observe both without changing their delivery or coalescing behavior.
  const restore: (() => void)[] = [];
  const transports = new Set<object>();
  for (const [channel, transport] of [['client', server.environments.client.hot], ['ws', server.ws]] as const) {
    if (transports.has(transport)) continue;
    transports.add(transport);
    const original = transport.send;
    const send = function (this: unknown, ...args: any[]) {
      const payload = args[0];
      if (payload?.type === 'full-reload' || payload?.type === 'error') {
        log.write(payload.type === 'full-reload' ? 'server-reload-request' : 'server-error', {
          channel, path: payload.path, triggeredBy: payload.triggeredBy,
          ifFallback: payload.ifFallback, message: payload.err?.message,
          origin: new Error().stack?.split('\n').slice(2, 9).join('\n'),
        });
      }
      return original.apply(this, args as any);
    };
    transport.send = send;
    restore.push(() => { if (transport.send === send) transport.send = original; });
  }
  const watched = (kind: string, file: string) => log.write('watch', { kind, file: localPath(file) });
  server.watcher.on('all', watched);
  const connected = (socket: any) => {
    const connection = randomUUID();
    log.write('ws-connect', { connection });
    socket.once('close', (code: number, reason: Buffer) => log.write('ws-disconnect', {
      connection, code, reason: reason.toString().slice(0, 200),
    }));
  };
  server.ws.on('connection', connected);

  const seen = new Set<string>();
  server.middlewares.use(async (request, response, next) => {
    const url = new URL(request.url ?? '/', 'http://localhost');
    if (url.pathname !== refreshDiagnosticsPath) {
      if (request.headers.accept?.includes('text/html')) {
        const start = performance.now();
        const requestId = randomUUID();
        log.write('html-request', { request: requestId, method: request.method, path: url.pathname });
        response.once('finish', () => log.write('html-response', {
          request: requestId, status: response.statusCode, elapsedMs: Math.round(performance.now() - start),
        }));
      }
      return next();
    }
    response.setHeader('Cache-Control', 'no-store');
    if (request.headers['sec-fetch-site'] === 'cross-site'
      || (request.headers.origin && request.headers.origin !== `http${server.config.server.https ? 's' : ''}://${request.headers.host}`)) {
      response.statusCode = 403;
      return response.end();
    }
    if (request.method === 'GET') {
      response.setHeader('Content-Type', 'text/plain; charset=utf-8');
      return response.end(log.tail());
    }
    if (request.method !== 'POST') {
      response.statusCode = 405;
      return response.end();
    }
    try {
      let body = '';
      for await (const chunk of request) {
        body += chunk;
        if (Buffer.byteLength(body) > 32 * 1024) {
          response.statusCode = 413;
          return response.end();
        }
      }
      const events = JSON.parse(body);
      if (!Array.isArray(events) || events.length > 30) throw new Error('Expected up to 30 browser events');
      for (const event of events) {
        if (typeof event?.id !== 'string' || event.id.length > 100 || typeof event?.event !== 'string'
          || !event.event.startsWith('browser-') || JSON.stringify(event).length > 4000) throw new Error('Invalid browser event');
      }
      for (const event of events) {
        if (seen.has(event.id)) continue;
        seen.add(event.id);
        if (seen.size > 2000) seen.delete(seen.values().next().value!);
        log.write(event.event, { browser: event });
      }
      response.statusCode = 204;
      response.end();
    } catch {
      response.statusCode = 400;
      response.end();
    }
  });
  server.httpServer?.once('close', () => {
    log.write('server-close');
    server.watcher.off('all', watched);
    server.ws.off('connection', connected);
    restore.forEach(cleanup => cleanup());
  });
}
