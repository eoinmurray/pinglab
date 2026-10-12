let active: HTMLAnchorElement | undefined;
let hideTimer: ReturnType<typeof setTimeout>;
const preview = document.createElement('aside');
preview.className = 'citation-preview';
preview.id = 'citation-preview';
preview.setAttribute('role', 'tooltip');
preview.hidden = true;
document.body.append(preview);

function hide() {
  clearTimeout(hideTimer);
  preview.hidden = true;
  active?.removeAttribute('aria-describedby');
  active = undefined;
}
function scheduleHide() { hideTimer = setTimeout(hide, 180); }
function show(link: HTMLAnchorElement) {
  clearTimeout(hideTimer);
  active?.removeAttribute('aria-describedby');
  const reference = document.getElementById(decodeURIComponent(link.hash.slice(1)));
  const entry = reference?.closest('li') ?? reference?.parentElement;
  if (!entry) return;
  active = link;
  preview.replaceChildren(...Array.from(entry.childNodes).map(node => node.cloneNode(true)));
  preview.querySelectorAll('[id]').forEach(node => node.removeAttribute('id'));
  preview.hidden = false;
  link.setAttribute('aria-describedby', preview.id);
  const rect = link.getBoundingClientRect();
  const width = preview.getBoundingClientRect().width;
  preview.style.left = `${Math.max(12, Math.min(rect.left, innerWidth - width - 12))}px`;
  preview.style.top = `${Math.min(rect.bottom + 8, Math.max(12, innerHeight - preview.offsetHeight - 12))}px`;
}
for (const link of document.querySelectorAll<HTMLAnchorElement>('d-article a[href^="#ref-"]')) {
  link.addEventListener('mouseenter', () => show(link));
  link.addEventListener('focus', () => show(link));
  link.addEventListener('mouseleave', scheduleHide);
  link.addEventListener('blur', scheduleHide);
}
preview.addEventListener('mouseenter', () => clearTimeout(hideTimer));
preview.addEventListener('mouseleave', scheduleHide);
preview.addEventListener('focusin', () => clearTimeout(hideTimer));
preview.addEventListener('focusout', scheduleHide);
document.addEventListener('keydown', event => { if (event.key === 'Escape') hide(); });
window.addEventListener('scroll', hide, { passive: true });
