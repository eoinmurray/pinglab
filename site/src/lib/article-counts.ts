import { parseFragment, type DefaultTreeAdapterMap } from 'parse5';

type Node = DefaultTreeAdapterMap['node'];
export interface ArticleCounts { words: number | null; figures: number | null; runs?: number; }

const excludedTags = new Set(['script', 'style', 'template', 'nav', 'aside', 'figcaption', 'table', 'pre', 'code', 'svg', 'math']);
const excludedSections = /^(?:meta|references|bibliography|appendix\b.*|appendices|supplementary details|parameters|network|dataset)$/i;
const segmenter = new Intl.Segmenter('en', { granularity: 'word' });
const children = (node: Node): Node[] => 'childNodes' in node ? node.childNodes : [];
const text = (node: Node): string => node.nodeName === '#text' ? (node as DefaultTreeAdapterMap['textNode']).value : children(node).map(text).join('');

/** Count the rendered main account, before the separately rendered footer. */
export function countArticle(html: string, abstract = ''): Pick<ArticleCounts, 'words' | 'figures'> {
  let figures = 0;
  const prose: string[] = [abstract, ' '];
  let excludedDepth: number | undefined;
  function visit(node: Node) {
    if (node.nodeName === '#text') { if (excludedDepth === undefined) prose.push(text(node)); return; }
    const tag = node.nodeName;
    const heading = /^h([1-6])$/.exec(tag);
    if (heading) {
      const depth = Number(heading[1]);
      if (excludedDepth !== undefined && depth <= excludedDepth) excludedDepth = undefined;
      if (excludedSections.test(text(node).trim())) excludedDepth = depth;
      return;
    }
    if (excludedDepth !== undefined || excludedTags.has(tag)) return;
    const classes = 'attrs' in node ? node.attrs.find(attr => attr.name === 'class')?.value.split(/\s+/) ?? [] : [];
    if (classes.some(name => ['katex', 'report-availability', 'figure-unavailable', 'report-table', 'source-table'].includes(name))) return;
    if (tag === 'figure') {
      figures++;
      // Figure content and captions are not main-report prose.
      return;
    }
    for (const child of children(node)) visit(child);
    if (['p', 'div', 'section', 'li', 'br'].includes(tag)) prose.push(' ');
  }
  visit(parseFragment(html));
  const words = [...segmenter.segment(prose.join(''))].filter(segment => segment.isWordLike).length;
  return { words, figures };
}
