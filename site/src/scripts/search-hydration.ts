import type { ClientDirective } from 'astro';

// Keep the server-rendered button usable without downloading React on page load.
const search: ClientDirective = (load, _options, element) => {
  let pending = false;
  const button = () => element.querySelector<HTMLButtonElement>('.command-trigger');

  async function open() {
    if (pending) return;
    pending = true;
    element.setAttribute('aria-busy', 'true');
    try {
      const hydrate = await load();
      await hydrate();
      element.removeEventListener('click', onClick, true);
      document.removeEventListener('keydown', onKeyDown);
      // Replay the first activation through React's hydrated trigger.
      button()?.click();
    } catch (error) {
      pending = false;
      console.error('Search could not load. Activate Search to retry.', error);
    } finally {
      element.removeAttribute('aria-busy');
    }
  }

  function onClick(event: MouseEvent) {
    if (!(event.target instanceof Element) || !event.target.closest('.command-trigger')) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    void open();
  }
  function onKeyDown(event: KeyboardEvent) {
    if (event.key.toLowerCase() !== 'k' || !(event.metaKey || event.ctrlKey)
      || event.altKey || event.shiftKey || event.isComposing) return;
    event.preventDefault();
    if (!event.repeat) void open();
  }
  element.addEventListener('click', onClick, true);
  document.addEventListener('keydown', onKeyDown);
};

export default search;
