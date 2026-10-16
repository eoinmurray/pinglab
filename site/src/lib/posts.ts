import { getReports } from './reports';
import { getCollection, type CollectionEntry } from 'astro:content';
import { collectionMemberships } from './collections';

export interface Post {
  id: string;
  href: string;
  label: string;
  collections: string[];
  article?: CollectionEntry<'reports'> | CollectionEntry<'notes'>;
  data: {
    title: string;
    author: string;
    description: string;
    updated_at: string;
  };
}

export async function getPosts(): Promise<Post[]> {
  const [reports, notes, slides] = await Promise.all([
    getReports(), getCollection('notes'), getCollection('slides'),
  ]);
  return [
    ...reports.map(report => ({
      ...report, href: `/${report.id}`, label: report.id, collections: collectionMemberships(report.data.collection),
      article: report,
    })),
    ...slides.map(slide => ({
      ...slide, href: `/slides/${slide.id}`, label: 'Slides', collections: collectionMemberships(slide.data.collection ?? 'slides'),
    })),
    ...notes.map(note => ({
      ...note, href: `/notes/${note.id}`, label: 'Note', collections: collectionMemberships(note.data.collection),
      article: note,
    })),
  ].sort((a, b) =>
    b.data.updated_at.localeCompare(a.data.updated_at)
    || a.id.localeCompare(b.id)
    || a.href.localeCompare(b.href)
  );
}
