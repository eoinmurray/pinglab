export const diagnosticsEndpoint = '/__pinglab/dev-refresh';
const storageKey = 'pinglab:refresh-debug';
const maxEvents = 100;
type Event = { id: string; event: string; time: string; tab: string; document: string; path: string; details: Record<string, unknown> };

/** A per-tab journal survives reloads and buffers events while the server sleeps. */
export function createRefreshRecorder(platform: {
  storage: Pick<Storage, 'getItem' | 'setItem'>;
  send: (events: Event[]) => Promise<boolean>;
  path: string;
  now?: () => number;
  id?: () => string;
}) {
  const now = platform.now ?? Date.now;
  const id = platform.id ?? (() => crypto.randomUUID());
  const document = id();
  let tab = id();
  let events: Event[] = [];
  try {
    const saved = JSON.parse(platform.storage.getItem(storageKey) ?? 'null');
    if (typeof saved?.tab === 'string' && Array.isArray(saved.events)) {
      tab = saved.tab;
      events = saved.events.filter((event: Event) => typeof event?.id === 'string').slice(-maxEvents);
    }
  } catch { /* Storage may be disabled or corrupted. In-memory recording still works. */ }
  let sending = false;
  let previousTick = now();
  const persist = () => {
    try { platform.storage.setItem(storageKey, JSON.stringify({ tab, events })); } catch { /* Optional storage. */ }
  };
  const flush = async () => {
    if (sending || !events.length) return;
    sending = true;
    // Leave room under both the endpoint's 32 KiB cap and fetch's keepalive cap.
    const batch: Event[] = [];
    let bytes = 2;
    for (const event of events.slice(0, 30)) {
      const size = new TextEncoder().encode(JSON.stringify(event)).length + 1;
      if (batch.length && bytes + size > 24 * 1024) break;
      batch.push(event);
      bytes += size;
    }
    try {
      if (await platform.send(batch)) {
        const acknowledged = new Set(batch.map(event => event.id));
        events = events.filter(event => !acknowledged.has(event.id));
        persist();
      }
    } catch { /* Retry on the next lifecycle event or heartbeat, never reload. */ }
    finally { sending = false; }
  };
  const record = (event: string, details: Record<string, unknown> = {}) => {
    const encoded = JSON.stringify(details);
    if (encoded.length > 1800) details = { truncated: true, summary: encoded.slice(0, 1800) };
    events.push({ id: id(), event: `browser-${event}`, time: new Date(now()).toISOString(), tab, document, path: platform.path, details });
    events = events.slice(-maxEvents);
    persist();
    void flush();
  };
  return {
    record, flush,
    tick() {
      const time = now();
      const gapMs = time - previousTick;
      previousTick = time;
      // A delayed timer is evidence of suspension/throttling, not proof of sleep.
      if (gapMs > 45_000 || gapMs < 0) record('timer-gap', { gapMs });
      else void flush();
    },
  };
}
