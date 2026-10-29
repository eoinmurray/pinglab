import { parse, parseFragment, serialize, serializeOuter, type DefaultTreeAdapterMap } from 'parse5';

type Node = DefaultTreeAdapterMap['node'];
type Element = DefaultTreeAdapterMap['element'];
const children = (node: Node): Node[] => 'childNodes' in node ? node.childNodes : [];
const elements = (node: Node): Element[] => [
  ...('tagName' in node ? [node] : []), ...children(node).flatMap(elements),
];

/** Remove fetchable media URLs before the browser's preload scanner sees them. */
export function deferMedia(html: string): string {
  if (!/<(?:img|picture|video|svg)\b/i.test(html)) return html;
  const document = parse(html);
  let changed = false;
  function visit(node: Node) {
    if ('tagName' in node) {
      if (['noscript', 'template', 'script', 'style'].includes(node.tagName)
          || node.attrs.some(attribute => attribute.name === 'data-deferred-media')) return;
      if (['img', 'picture', 'video', 'svg'].includes(node.tagName)) {
        const resources = elements(node).flatMap(element => {
          const names = element.tagName === 'image' ? ['href']
            : element.tagName === 'video' ? ['src', 'poster']
            : ['img', 'source'].includes(element.tagName) ? ['src', 'srcset'] : [];
          return element.attrs.filter(attribute => names.includes(attribute.name) && attribute.value && !attribute.value.startsWith('#'))
            .map(attribute => ({ element, attribute }));
        });
        if (resources.length) {
          const original = serializeOuter(node);
          for (const image of elements(node).filter(element => element.tagName === 'img')) {
            image.attrs.push({ name: 'data-media-loading', value: '' });
            const width = Number(image.attrs.find(attribute => attribute.name === 'width')?.value);
            const height = Number(image.attrs.find(attribute => attribute.name === 'height')?.value);
            if (width > 0 && height > 0 && Number.isFinite(width / height)) {
              let style = image.attrs.find(attribute => attribute.name === 'style');
              if (!style) image.attrs.push(style = { name: 'style', value: '' });
              style.value += `;--media-aspect: ${width} / ${height}`;
            }
          }
          for (const { element, attribute } of resources) {
            element.attrs = element.attrs.filter(item => item !== attribute);
            element.attrs.push({ name: `data-media-${attribute.prefix ? `${attribute.prefix}-` : ''}${attribute.name}`, value: attribute.value });
          }
          node.attrs.push({ name: 'data-deferred-media', value: '' });
          const fallback = parseFragment(`<noscript>${original}</noscript>`).childNodes[0];
          const parent = node.parentNode!;
          fallback.parentNode = parent;
          parent.childNodes.splice(parent.childNodes.indexOf(node) + 1, 0, fallback);
          changed = true;
          return;
        }
      }
    }
    for (const child of [...children(node)]) visit(child);
  }
  visit(document);
  return changed ? serialize(document) : html;
}
