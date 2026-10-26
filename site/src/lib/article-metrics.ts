import { render, type CollectionEntry } from 'astro:content';
import { experimental_AstroContainer as AstroContainer } from 'astro/container';
import mdxRenderer from '@astrojs/mdx/server.js';
import { countArticle, type ArticleCounts } from './article-counts';
import { completedRunCount } from './pingstore';

type Article = CollectionEntry<'reports'> | CollectionEntry<'notes'>;
type BodyCounts = Pick<ArticleCounts, 'words' | 'figures'>;
const cache = new Map<string, Promise<BodyCounts>>();

async function calculate(article: Article): Promise<BodyCounts> {
  try {
    const { Content } = await render(article);
    const container = await AstroContainer.create();
    container.addServerRenderer({ name: 'astro:jsx', renderer: mdxRenderer });
    const html = await container.renderToString(Content);
    return countArticle(html, article.data.abstract);
  } catch (error) {
    // A broken article must not take the homepage and every collection offline.
    console.error(`[article-metrics] ${article.collection}/${article.id}:`, error);
    return { words: null, figures: null };
  }
}

export async function articleMetrics(article: Article): Promise<ArticleCounts> {
  // Vite invalidates this module on source edits; selected evidence is session-frozen.
  const key = `${article.collection}/${article.id}`;
  if (!cache.has(key)) cache.set(key, calculate(article));
  const [body, runs] = await Promise.all([
    cache.get(key)!,
    article.collection === 'reports' ? completedRunCount(article.data.id) : undefined,
  ]);
  // Retry unavailable articles on refresh, and always refresh the run inventory.
  if (body.words === null) cache.delete(key);
  return { ...body, runs };
}
