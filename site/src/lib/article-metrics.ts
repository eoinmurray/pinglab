import { render, type CollectionEntry } from 'astro:content';
import { experimental_AstroContainer as AstroContainer } from 'astro/container';
import mdxRenderer from '@astrojs/mdx/server.js';
import { countArticle, type ArticleCounts } from './article-counts';
import { completedRunCount } from './pingstore';

type Article = CollectionEntry<'reports'> | CollectionEntry<'notes'>;
const cache = new Map<string, Promise<ArticleCounts>>();

async function calculate(article: Article): Promise<ArticleCounts> {
  const runs = article.collection === 'reports' ? await completedRunCount(article.data.id) : undefined;
  try {
    const { Content } = await render(article);
    const container = await AstroContainer.create();
    container.addServerRenderer({ name: 'astro:jsx', renderer: mdxRenderer });
    const html = await container.renderToString(Content);
    return { ...countArticle(html, article.data.abstract), runs };
  } catch (error) {
    // A broken article must not take the homepage and every collection offline.
    console.error(`[article-metrics] ${article.collection}/${article.id}:`, error);
    return { words: null, figures: null, runs };
  }
}

export function articleMetrics(article: Article): Promise<ArticleCounts> {
  if (import.meta.env.DEV) return calculate(article);
  const key = `${article.collection}/${article.id}`;
  if (!cache.has(key)) cache.set(key, calculate(article));
  return cache.get(key)!;
}
