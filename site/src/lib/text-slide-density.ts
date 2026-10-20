import { parseFragment, type DefaultTreeAdapterMap } from 'parse5';

/** Conservative body-only enlargement; embedded layouts keep their authored size. */
export function textSlideDensity(html: string): 'sparse' | 'medium' | 'dense' {
  let text = '';
  let blocks = 0;
  let embedded = false;
  function visit(node: DefaultTreeAdapterMap['node']) {
    if ('tagName' in node) {
      if (['table', 'img', 'svg', 'video', 'iframe', 'pre', 'math'].includes(node.tagName)) embedded = true;
      if (['li', 'p', 'h3', 'h4', 'blockquote'].includes(node.tagName)) blocks++;
      if (['script', 'style'].includes(node.tagName)) return;
    }
    if (node.nodeName === '#text') text += (node as DefaultTreeAdapterMap['textNode']).value;
    if ('childNodes' in node) node.childNodes.forEach(visit);
  }
  visit(parseFragment(html));
  const length = text.replace(/\s+/g, ' ').trim().length;
  if (embedded || !length) return 'dense';
  if (length <= 240 && blocks <= 5) return 'sparse';
  if (length <= 520 && blocks <= 6) return 'medium';
  return 'dense';
}
