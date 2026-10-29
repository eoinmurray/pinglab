import { defineMiddleware } from 'astro:middleware';
import { withRunContext } from './lib/run-context';
import { deferMedia } from './lib/deferred-media';
import { validateArticleMarkup } from './lib/article-markup';

const escapeHtml = (value: string) => value.replace(/[&<>"']/g, character => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[character]!));

export const onRequest = defineMiddleware((context, next) => withRunContext(async () => {
  try {
    const response = await next();
    if (!response.headers.get('Content-Type')?.includes('text/html') || !response.body) return response;
    // Complete rendering before sending HTML so a deferred component failure
    // can replace this page, rather than leaving a partially rendered response.
    const rendered = await response.text();
    validateArticleMarkup(rendered, context.url.pathname);
    const html = deferMedia(rendered);
    const headers = new Headers(response.headers);
    headers.delete('Content-Length');
    return new Response(html, {
      status: response.status, statusText: response.statusText, headers,
    });
  } catch (error) {
    if (import.meta.env.PROD) throw error;
    console.error(`[page-error] ${context.url.pathname}:`, error);
    const message = error instanceof Error ? error.message : String(error);
    return new Response(`<!doctype html><html lang="en"><head><meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <meta name="robots" content="noindex"><title>An error occurred — Pinglab</title>
      <style>body{font:18px/1.6 system-ui,sans-serif;max-width:900px;margin:10vh auto;padding:24px;color:#202124;background:#fff}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f4f6;color:#b42318;padding:24px;border-radius:8px}a{color:#315bba}</style>
      </head><body><main><h1>An error occurred.</h1>
      <p>This page could not be rendered.</p><pre>${escapeHtml(message)}</pre>
      <p><a href="/">Return to articles</a></p></main></body></html>`, {
      status: 500, headers: { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' },
    });
  }
}));
