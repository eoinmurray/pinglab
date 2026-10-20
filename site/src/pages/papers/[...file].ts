import type { APIRoute } from 'astro';
import { readFile } from 'node:fs/promises';
import { getPapers } from '../../lib/papers';

export async function getStaticPaths() {
  return (await getPapers()).map(paper => ({
    params: { file: paper.file }, props: { path: paper.path },
  }));
}

export const GET: APIRoute = async ({ props }) => new Response(
  new Uint8Array(await readFile(props.path)),
  { headers: { 'Content-Type': 'application/pdf' } },
);
