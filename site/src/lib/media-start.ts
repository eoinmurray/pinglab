type Platform = {
  document: Pick<Document, 'readyState'>;
  window: Pick<Window, 'addEventListener' | 'removeEventListener' | 'setTimeout' | 'clearTimeout'>;
  IntersectionObserver?: typeof IntersectionObserver;
};

/** Both the load event and proximity to the viewport must precede activation. */
export function whenVisibleAfterLoad(host: Element, activate: () => void, platform: Platform = globalThis): () => void {
  let cancelled = false;
  let timer: number | undefined;
  let observer: IntersectionObserver | undefined;
  const start = () => {
    if (cancelled || !host.isConnected) return;
    observer?.disconnect();
    cancelled = true;
    activate();
  };
  const afterLoad = () => {
    // Use a new task so activation cannot extend the window load handler.
    timer = platform.window.setTimeout(() => {
      if (cancelled || !host.isConnected) return;
      if (!platform.IntersectionObserver) return start();
      observer = new platform.IntersectionObserver(entries => {
        if (entries.some(entry => entry.isIntersecting)) start();
      }, { rootMargin: '200px' });
      observer.observe(host);
    }, 0);
  };
  if (platform.document.readyState === 'complete') afterLoad();
  else platform.window.addEventListener('load', afterLoad, { once: true });
  return () => {
    cancelled = true;
    platform.window.removeEventListener('load', afterLoad);
    if (timer !== undefined) platform.window.clearTimeout(timer);
    observer?.disconnect();
  };
}

type Dispose = () => void;
type Mount = (host: HTMLElement) => Promise<Dispose>;

/** Delay the implementation download too, and dispose safely across removal/HMR. */
export function deferMount(host: HTMLElement, load: () => Promise<Mount>, onError: (error: unknown) => void, platform: Platform = globalThis): Dispose {
  let cancelled = false;
  let dispose: Dispose | undefined;
  const cancelStart = whenVisibleAfterLoad(host, () => {
    void (async () => {
      try {
        const mount = await load();
        if (cancelled || !host.isConnected) return;
        const cleanup = await mount(host);
        if (cancelled || !host.isConnected) cleanup();
        else dispose = cleanup;
      } catch (error) { if (!cancelled) onError(error); }
    })();
  }, platform);
  return () => {
    cancelled = true;
    cancelStart();
    dispose?.();
    dispose = undefined;
  };
}
