import { getCollection } from 'astro:content';
import { collectionMemberships } from './collections';

export interface Post {
  id: string;
  href: string;
  label: string;
  collections: string[];
  data: {
    title: string;
    author: string;
    description: string;
    updated_at: string;
  };
}

export async function getPosts(): Promise<Post[]> {
  const [reports, notes, slides] = await Promise.all([
    getCollection('reports'), getCollection('notes'), getCollection('slides'),
  ]);
  const dated = [...slides.map(slide => ({
    ...slide, href: `/slides/${slide.id}`, label: 'Slides', collections: collectionMemberships(slide.data.collection ?? 'slides'),
  })), ...notes.map(note => ({
    ...note, href: `/notes/${note.id}`, label: 'Note', collections: collectionMemberships(note.data.collection),
  }))].sort((a, b) =>
    b.data.updated_at.localeCompare(a.data.updated_at) || a.id.localeCompare(b.id)
  );
  return [
    ...dated,
    ...reports.sort((a, b) => b.id.localeCompare(a.id)).map(report => ({
      ...report, href: `/${report.id}`, label: report.id, collections: collectionMemberships(report.data.collection),
    })),
  ];
}
