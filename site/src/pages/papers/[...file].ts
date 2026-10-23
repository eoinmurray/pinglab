import type { APIRoute } from 'astro';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { retryRead } from '../../lib/remote-http';
import { getPapers } from '../../lib/papers';
// Refresh cached development routes when the paper index changes.
import '../../../../papers/index.yaml?raw';

export async function getStaticPaths() {
  return (await getPapers()).map(paper => ({
    params: { file: paper.file }, props: { path: paper.path, url: paper.remote ? paper.href : undefined, bytes: paper.bytes, sha256: paper.sha256 },
  }));
}

// Retain existing /papers/*.pdf bookmarks while listings can link directly to R2.
export const GET: APIRoute = async ({ props }) => {
  let bytes: Buffer;
  if (props.url) {
    bytes = await retryRead(`R2 paper ${props.url}`, async () => {
      const response = await fetch(props.url, { signal: AbortSignal.timeout(30_000), redirect: 'error' });
      if (response.status !== 200) throw new Error(`R2 paper unavailable: ${props.url}`);
      const downloaded = Buffer.from(await response.arrayBuffer());
      if (downloaded.length !== props.bytes || createHash('sha256').update(downloaded).digest('hex') !== props.sha256) {
        throw new Error(`R2 paper checksum mismatch: ${props.url}`);
      }
      return downloaded;
    });
  } else bytes = await readFile(props.path);
  return new Response(new Uint8Array(bytes), { headers: { 'Content-Type': 'application/pdf' } });
};
