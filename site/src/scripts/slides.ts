const deck = document.querySelector<HTMLElement>('.deck');

if (deck) {
  // Plain Markdown (or component-free MDX) uses H2 headings as slide boundaries.
  if (!deck.querySelector('[data-slide]')) {
    const nodes = [...deck.childNodes];
    let section: HTMLElement | undefined;
    for (const node of nodes) {
      if (!section && node.nodeType === Node.TEXT_NODE && !node.textContent?.trim()) continue;
      if (!section || (node instanceof HTMLElement && node.tagName === 'H2')) {
        section = document.createElement('section');
        section.className = 'slide';
        section.dataset.slide = '';
        deck.append(section);
      }
      section.append(node);
    }
  }

  const slides = [...deck.querySelectorAll<HTMLElement>(':scope > [data-slide]')];
  const controls = document.querySelector<HTMLElement>('.deck-controls')!;
  const previous = controls.querySelector<HTMLButtonElement>('[data-previous]')!;
  const next = controls.querySelector<HTMLButtonElement>('[data-next]')!;
  const fullscreen = controls.querySelector<HTMLButtonElement>('[data-fullscreen]')!;
  const counter = controls.querySelector<HTMLOutputElement>('[data-counter]')!;
  let current = 0;

  slides.forEach((slide, index) => {
    slide.id ||= `slide-${index + 1}`;
    slide.setAttribute('role', 'group');
    slide.setAttribute('aria-roledescription', 'slide');
    const title = slide.querySelector('h1, h2')?.textContent?.trim();
    slide.setAttribute('aria-label', `${index + 1} of ${slides.length}${title ? `: ${title}` : ''}`);
  });

  function show(index: number, updateHash = true, focus = false) {
    current = Math.max(0, Math.min(slides.length - 1, index));
    slides.forEach((slide, i) => {
      slide.classList.toggle('is-active', i === current);
      slide.inert = i !== current;
      slide.tabIndex = -1;
    });
    previous.disabled = current === 0;
    next.disabled = current === slides.length - 1;
    counter.value = `${current + 1} / ${slides.length}`;
    if (updateHash && location.hash !== `#${slides[current].id}`) history.pushState(null, '', `#${slides[current].id}`);
    deck!.scrollTop = 0;
    if (focus) slides[current].focus({ preventScroll: true });
  }

  function fromHash() {
    const hash = location.hash.slice(1);
    let id: string;
    try { id = decodeURIComponent(hash); } catch { return; }
    const target = document.getElementById(id);
    const index = slides.findIndex(slide => slide === target || (target !== null && slide.contains(target)));
    show(index >= 0 ? index : 0, false);
    if (target && target !== slides[current]) target.scrollIntoView({ block: 'nearest' });
  }

  async function toggleFullscreen() {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await document.documentElement.requestFullscreen();
    } catch {
      fullscreen.textContent = 'Fullscreen unavailable';
    }
  }

  if (slides.length) {
    document.body.classList.add('deck-ready');
    controls.hidden = false;
    fullscreen.hidden = !document.fullscreenEnabled;
    fromHash();
    previous.addEventListener('click', () => show(current - 1));
    next.addEventListener('click', () => show(current + 1));
    fullscreen.addEventListener('click', toggleFullscreen);
    window.addEventListener('hashchange', fromHash);
    window.addEventListener('popstate', fromHash);
    document.addEventListener('fullscreenchange', () => {
      fullscreen.textContent = document.fullscreenElement ? 'Exit fullscreen' : 'Fullscreen';
    });

    document.addEventListener('keydown', event => {
      if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) return;
      const key = event.key.toLowerCase();
      if (event.target instanceof Element && event.target.closest('a, input, textarea, select, [contenteditable], video, audio')) return;
      // Space on a focused button retains the button's native activation.
      if (key === ' ' && event.target instanceof Element && event.target.closest('button')) return;
      if (key === 'f' && document.fullscreenEnabled) { event.preventDefault(); void toggleFullscreen(); return; }
      const index = key === 'home' ? 0 : key === 'end' ? slides.length - 1
        : ['arrowright', 'arrowdown', 'pagedown'].includes(key) || (key === ' ' && !event.shiftKey) ? current + 1
        : ['arrowleft', 'arrowup', 'pageup'].includes(key) || (key === ' ' && event.shiftKey) ? current - 1 : null;
      if (index !== null) { event.preventDefault(); show(index, true, true); }
    });

    let touch: { x: number; y: number } | undefined;
    deck.addEventListener('touchstart', event => {
      if (event.target instanceof Element && event.target.closest('a, button, input, textarea, select, video, audio')) { touch = undefined; return; }
      const point = event.touches[0];
      touch = event.touches.length === 1 ? { x: point.clientX, y: point.clientY } : undefined;
    }, { passive: true });
    deck.addEventListener('touchend', event => {
      const point = event.changedTouches[0];
      if (!touch || !point) return;
      const dx = point.clientX - touch.x;
      const dy = point.clientY - touch.y;
      touch = undefined;
      if (Math.abs(dx) > 70 && Math.abs(dx) > Math.abs(dy) * 2) show(current + (dx < 0 ? 1 : -1));
    }, { passive: true });
  }
}
