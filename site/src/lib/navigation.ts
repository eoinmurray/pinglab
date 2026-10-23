import { getPosts } from './posts';
import { getPapers, paperCollectionLabel } from './papers';
import { collectionLabel, collectionUrl, collectionDescription, compareCollections } from './collections';

export interface NavigationEntry {
  href: string;
  title: string;
  detail: string;
  subtitle?: string;
  keywords: string[];
}

export async function getNavigation() {
  const [posts, library] = await Promise.all([getPosts(), getPapers()]);
  const names = [...new Set(posts.flatMap(post => post.collections))].sort(compareCollections);
  const collections: NavigationEntry[] = names.map(id => ({
    href: collectionUrl(id), title: collectionLabel(id), detail: 'Collection',
    keywords: [id, collectionDescription(id)],
  }));
  const articles: NavigationEntry[] = posts.map(post => ({
    href: post.href, title: post.data.title, detail: post.label,
    keywords: [post.id, post.data.description, ...post.collections.flatMap(id => [id, collectionLabel(id)])],
  }));
  const papers: NavigationEntry[] = library.map(paper => ({
    href: paper.href,
    title: paper.metadata?.title ?? paper.file,
    subtitle: paper.metadata?.authors.join(', '),
    detail: paper.metadata ? `${paper.metadata.id} · ${paper.metadata.year}` : 'PDF',
    keywords: [paper.metadata?.id ?? '', ...(paper.metadata?.authors ?? []), paper.file,
      String(paper.metadata?.year ?? ''), paper.metadata?.venue ?? '', paper.metadata?.doi ?? '',
      ...paper.collections.flatMap(id => [id, paperCollectionLabel(id)])],
  }));
  return { collections, articles, papers };
}
