import { createRefreshRecorder, diagnosticsEndpoint } from '../lib/refresh-recorder';

const recorder = createRefreshRecorder({
  // Accessors keep a blocked sessionStorage getter inside the recorder's try/catch.
  storage: {
    getItem: key => sessionStorage.getItem(key),
    setItem: (key, value) => sessionStorage.setItem(key, value),
  },
  path: location.pathname,
  send: async events => (await fetch(diagnosticsEndpoint, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(events), keepalive: true,
  })).ok,
});
const status = () => ({ visibility: document.visibilityState, online: navigator.onLine });
const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming | undefined;
recorder.record('boot', { ...status(), navigation: navigation?.type });
window.addEventListener('pageshow', event => recorder.record('pageshow', { ...status(), persisted: event.persisted }));
window.addEventListener('pagehide', event => recorder.record('pagehide', { ...status(), persisted: event.persisted }));
document.addEventListener('visibilitychange', () => recorder.record('visibility', status()));
for (const event of ['online', 'offline', 'focus', 'blur']) {
  window.addEventListener(event, () => { recorder.tick(); recorder.record(event, status()); });
}
window.addEventListener('error', event => recorder.record('error', {
  message: event.message?.slice(0, 600), file: event.filename?.split('?')[0], line: event.lineno,
}));
window.addEventListener('unhandledrejection', event => recorder.record('unhandled-rejection', {
  message: String(event.reason instanceof Error ? event.reason.message : event.reason).slice(0, 600),
}));
document.addEventListener('click', event => {
  const link = event.target instanceof Element ? event.target.closest('a[href]') : null;
  if (!(link instanceof HTMLAnchorElement) || link.origin !== location.origin) return;
  recorder.record('link', { to: link.pathname, hash: !!link.hash });
}, true);
const timer = window.setInterval(() => recorder.tick(), 10_000);

if (import.meta.hot) {
  import.meta.hot.on('vite:beforeFullReload', payload => recorder.record('vite-full-reload', {
    path: payload.path, triggeredBy: payload.triggeredBy, ifFallback: payload.ifFallback,
  }));
  import.meta.hot.on('vite:ws:disconnect', () => recorder.record('vite-disconnect', status()));
  import.meta.hot.on('vite:ws:connect', () => { recorder.tick(); recorder.record('vite-connect', status()); });
  import.meta.hot.on('vite:beforeUpdate', payload => recorder.record('vite-update', {
    paths: payload.updates.slice(0, 10).map(update => update.path),
  }));
  import.meta.hot.on('vite:error', payload => recorder.record('vite-error', { message: payload.err.message.slice(0, 600) }));
  // A source update should reload this small observer instead of duplicating listeners.
  import.meta.hot.dispose(() => window.clearInterval(timer));
}
