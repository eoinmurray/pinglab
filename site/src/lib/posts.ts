import { getCollection } from 'astro:content';

export interface Post {
  id: string;
  href: string;
  label: string;
  collection: string;
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
    ...slide, href: `/slides/${slide.id}`, label: 'Slides', collection: 'slides',
  })), ...notes.map(note => ({
    ...note, href: `/notes/${note.id}`, label: 'Note', collection: 'notes',
  }))].sort((a, b) =>
    b.data.updated_at.localeCompare(a.data.updated_at) || a.id.localeCompare(b.id)
  );
  return [
    ...dated,
    ...reports.sort((a, b) => b.id.localeCompare(a.id)).map(report => ({
      ...report, href: `/${report.id}`, label: report.id, collection: report.data.collection,
    })),
  ];
}
