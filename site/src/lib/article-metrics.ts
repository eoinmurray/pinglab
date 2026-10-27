import { render, type CollectionEntry } from 'astro:content';
import { experimental_AstroContainer as AstroContainer } from 'astro/container';
import mdxRenderer from '@astrojs/mdx/server.js';
import { countArticle, type ArticleCounts } from './article-counts';
import { unavailableReportInputs } from './report-data';
import { completedRunCount } from './pingstore';
import { runStore } from './run-store';

type Article = CollectionEntry<'reports'> | CollectionEntry<'notes'>;
type BodyCounts = Pick<ArticleCounts, 'words' | 'figures'>;
const cache = new Map<string, { generation: string; body: Promise<BodyCounts> }>();

async function calculate(article: Article): Promise<BodyCounts> {
  try {
    if (article.collection === 'reports' && (await unavailableReportInputs(article.data.inputs, article.data.tags.includes('data'))).length) return { words: null, figures: null };
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
  // Vite invalidates source edits; counts also follow the request’s runset generation.
  const key = `${article.collection}/${article.id}`;
  const generation = article.collection === 'reports' ? (await runStore.index()).generation : 'authored';
  let entry = cache.get(key);
  if (!entry || entry.generation !== generation) {
    entry = { generation, body: calculate(article) };
    cache.set(key, entry);
  }
  const [body, runs] = await Promise.all([
    entry.body,
    article.collection === 'reports' ? completedRunCount(article.data.id) : undefined,
  ]);
  // Retry unavailable articles on refresh, and always refresh the run inventory.
  if (body.words === null) cache.delete(key);
  return { ...body, runs };
}
