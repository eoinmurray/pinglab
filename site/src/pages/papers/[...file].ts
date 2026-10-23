import type { APIRoute } from 'astro';
import { readFile } from 'node:fs/promises';
import { getPapers } from '../../lib/papers';
// Refresh cached development routes when the paper index changes.
import '../../../../papers/index.yaml?raw';

export async function getStaticPaths() {
  // Local PDFs are available in development only, never in published output.
  if (import.meta.env.PROD) return [];
  return (await getPapers()).map(paper => ({
    params: { file: paper.file }, props: { path: paper.path },
  }));
}

export const GET: APIRoute = async ({ props }) => {
  const bytes = await readFile(props.path);
  return new Response(new Uint8Array(bytes), { headers: { 'Content-Type': 'application/pdf' } });
};
