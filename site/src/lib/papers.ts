import { readdir, readFile, stat } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { parse } from 'yaml';

const papersDirectory = resolve('../papers');

interface PaperMetadata {
  id: string;
  title: string;
  authors: string[];
  year: number;
  venue?: string;
  doi?: string;
  url?: string;
  collection?: string | string[];
}

export interface Paper {
  file: string;
  path: string;
  href?: string;
  bytes: number;
  metadata?: PaperMetadata;
  collections: string[];
}

export function paperCollectionLabel(collection: string) {
  if (collection === 'ping') return 'PING';
  const words = collection.replaceAll('-', ' ');
  return words.replace(/\b\w/g, letter => letter.toUpperCase());
}

export function paperCollectionUrl(collection: string) {
  return `/papers/collections/${encodeURIComponent(collection)}`;
}

export function getPaperCollections(papers: Paper[]) {
  return [...new Set(papers.flatMap(paper => paper.collections))].sort((a, b) =>
    Number(a === 'uncategorized') - Number(b === 'uncategorized') || a.localeCompare(b, 'en'));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function validCollections(value: unknown): boolean {
  const values = typeof value === 'string' ? [value] : value;
  return Array.isArray(values) && values.length > 0
    && values.every(item => typeof item === 'string' && /^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(item))
    && new Set(values).size === values.length;
}

export async function readPaperIndex(directory: string): Promise<Record<string, PaperMetadata>> {
  let source: string;
  try { source = await readFile(join(directory, 'index.yaml'), 'utf8'); }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return {};
    throw error;
  }
  const index: unknown = parse(source);
  if (!isRecord(index)) throw new Error('papers/index.yaml must contain a filename-to-metadata mapping.');
  const nonempty = (value: unknown): value is string => typeof value === 'string' && value.trim().length > 0;
  const ids = new Set<string>();
  for (const [file, entry] of Object.entries(index)) {
    if (!isRecord(entry) || typeof entry.id !== 'string' || !/^P(?:0[1-9]|[1-9]\d+)$/.test(entry.id)
      || !nonempty(entry.title)
      || !Array.isArray(entry.authors) || entry.authors.length === 0 || !entry.authors.every(nonempty)
      || !Number.isInteger(entry.year) || Number(entry.year) < 1
      || (entry.venue !== undefined && !nonempty(entry.venue))
      || (entry.url !== undefined && (!nonempty(entry.url) || !/^https:\/\/[^\s]+$/.test(entry.url)))
      || (entry.collection !== undefined && !validCollections(entry.collection))
      || (entry.doi !== undefined && (!nonempty(entry.doi) || !/^10\.\d{4,9}\/\S+$/.test(entry.doi)))) {
      throw new Error(`Invalid papers/index.yaml entry for ${file}: a P01-style id, title, authors and year are required; collection must be a slug or a nonempty list of unique slugs; venue, DOI and an HTTPS source URL are optional.`);
    }
    if (ids.has(entry.id)) throw new Error(`Duplicate paper ID ${entry.id} in papers/index.yaml (${file}).`);
    ids.add(entry.id);
  }
  return index as unknown as Record<string, PaperMetadata>;
}

export async function getPapers(directory = papersDirectory, { published = process.env.NODE_ENV === 'production' } = {}) {
  const index = await readPaperIndex(directory);
  const papers: Paper[] = [];
  if (published) {
    // Public discovery must never depend on local PDFs or a public PDF archive.
    for (const [file, metadata] of Object.entries(index)) {
      const collection = metadata.collection ?? 'uncategorized';
      papers.push({ file, path: '', bytes: 0,
        href: metadata.url ?? (metadata.doi ? `https://doi.org/${metadata.doi}` : undefined),
        metadata, collections: typeof collection === 'string' ? [collection] : collection });
    }
  }
  async function visit(directory: string, prefix = '') {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      if (entry.name.startsWith('.')) continue;
      const file = prefix + entry.name;
      const path = join(directory, entry.name);
      if (entry.isDirectory()) await visit(path, `${file}/`);
      else if (entry.isFile() && /\.pdf$/i.test(entry.name)) {
        const metadata = Object.hasOwn(index, file) ? index[file] : undefined;
        const collection = metadata?.collection ?? 'uncategorized';
        papers.push({ file, path, href: `/papers/${file.split('/').map(encodeURIComponent).join('/')}`, bytes: (await stat(path)).size,
          metadata, collections: typeof collection === 'string' ? [collection] : collection });
      }
    }
  }
  if (!published) {
    try { await visit(directory); }
    catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
    }
    const files = new Set(papers.map(paper => paper.file));
    for (const file of Object.keys(index)) {
      if (!files.has(file)) console.warn(`[papers] Metadata has no matching PDF: ${file}`);
    }
  }
  return papers.sort((a, b) => (b.metadata?.year ?? 0) - (a.metadata?.year ?? 0)
    || (a.metadata?.title ?? a.file).localeCompare(b.metadata?.title ?? b.file, 'en')
    || a.file.localeCompare(b.file, 'en'));
}
