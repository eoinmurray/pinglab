import { whenVisibleAfterLoad } from '../lib/media-start';

const sources = ['srcset', 'src', 'href', 'xlink-href', 'poster'];
function reveal(root: Element) {
  if (!root.hasAttribute('data-deferred-media')) return;
  // Restore picture/video sources before the image's fallback src.
  const nodes = [...root.querySelectorAll('source, image, img'), root];
  const images = nodes.filter((node): node is HTMLImageElement => node instanceof HTMLImageElement);
  const checkCached = images.map(image => {
    const finish = () => {
      image.removeEventListener('load', finish);
      image.removeEventListener('error', finish);
      image.removeAttribute('data-media-loading');
      if (!image.naturalWidth) image.setAttribute('data-media-error', '');
    };
    image.addEventListener('load', finish, { once: true });
    image.addEventListener('error', finish, { once: true });
    return () => { if (image.complete && image.naturalWidth) finish(); };
  });
  for (const node of nodes) {
    if (node instanceof HTMLImageElement) {
      node.loading = 'eager';
      node.decoding = 'async';
    }
    for (const name of sources) {
      const value = node.getAttribute(`data-media-${name}`);
      if (value === null) continue;
      if (name === 'xlink-href') node.setAttributeNS('http://www.w3.org/1999/xlink', 'xlink:href', value);
      else node.setAttribute(name, value);
      node.removeAttribute(`data-media-${name}`);
    }
  }
  root.removeAttribute('data-deferred-media');
  checkCached.forEach(check => check());
  if (root instanceof HTMLVideoElement) root.load();
}

for (const media of document.querySelectorAll('[data-deferred-media]')) {
  whenVisibleAfterLoad(media, () => reveal(media));
}
// Printing needs figures on hidden slides and below the viewport as well.
window.addEventListener('beforeprint', () => {
  for (const media of document.querySelectorAll('[data-deferred-media]')) reveal(media);
});
