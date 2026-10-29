import { parse, type DefaultTreeAdapterMap } from 'parse5';

type Node = DefaultTreeAdapterMap['node'];

/** Anonymous text grid items cannot receive Distill's `grid-column: text`. */
export function validateArticleMarkup(html: string, pathname: string): void {
  if (!/<d-(?:article|appendix)\b/i.test(html)) return;

  function visit(node: Node) {
    if ('tagName' in node) {
      if (['script', 'style', 'template', 'noscript'].includes(node.tagName)) return;
      const grid = ['d-article', 'd-appendix'].includes(node.tagName)
        || node.attrs.some(attribute => attribute.name === 'class'
          && attribute.value.split(/\s+/).includes('report-appendix'));
      if (grid) {
        for (const child of node.childNodes) {
          if ('value' in child && child.value.trim()) {
            const excerpt = child.value.trim().replace(/\s+/g, ' ').slice(0, 120);
            throw new Error(`Article layout error at ${pathname}: unwrapped text in <${node.tagName}>: ${JSON.stringify(excerpt)}. Standalone MDX expressions that return prose must render a <p>, for example {data && <p>Result: {data.value}</p>}.`);
          }
        }
      }
    }
    if ('childNodes' in node) node.childNodes.forEach(visit);
  }

  visit(parse(html));
}
