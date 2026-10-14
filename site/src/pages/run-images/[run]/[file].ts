import type { APIRoute } from 'astro';
import { getReports } from '../../../lib/reports';
import { readdir, readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { latestPresentation } from '../../../lib/pingstore';

export async function getStaticPaths() {
  const paths = [];
  const seen = new Set<string>();
  for (const report of await getReports()) {
    if (!report.data.tags.includes('data')) continue;
    let selections;
    try {
      selections = report.data.runs ? Object.values(report.data.runs).map(id => ({ id, exportDirectory: resolve('../.pingstore/runs', id, 'export') })) : [await latestPresentation(report.id)];
    } catch (error) {
      console.error(`[run-images] ${report.id}:`, error);
      continue;
    }
    for (const run of selections) {
      if (seen.has(run.id)) continue;
      seen.add(run.id);
      let files;
      try {
        files = await readdir(run.exportDirectory);
      } catch (error) {
        console.error(`[run-images] ${report.id}/${run.id}:`, error);
        continue;
      }
      for (const file of files) {
        if (!/\.(png|svg|mp4|webm|ogv)$/.test(file) && !['circuit.json', 'classifier.json', 'training.json'].includes(file)) continue;
        paths.push({ params: { run: run.id, file }, props: { path: join(run.exportDirectory, file) } });
      }
    }
  }
  return paths;
}

async function serveFile(path: string): Promise<Response> {
  const bytes = await readFile(path);
  const type = ({ svg: 'image/svg+xml', png: 'image/png', mp4: 'video/mp4', webm: 'video/webm', ogv: 'video/ogg', json: 'application/json' } as Record<string, string>)[path.split('.').at(-1)];
  return new Response(new Uint8Array(bytes), { headers: {
    'Content-Type': type,
    'Cache-Control': import.meta.env.DEV ? 'no-store' : 'public, max-age=31536000, immutable',
  } });
};

export const GET: APIRoute = async ({ props }) => serveFile(props.path);

// Development serves one requested asset without enumerating unrelated reports.
export async function serveRunAsset(pathname: string): Promise<Response> {
  const match = /^\/run-images\/(exp\d{3}-r\d+-present)\/([^/]+)$/.exec(pathname);
  if (!match) return new Response('Asset not found', { status: 404 });
  let file: string;
  try { file = decodeURIComponent(match[2]); }
  catch { return new Response('Asset not found', { status: 404 }); }
  if (!/^[a-zA-Z0-9_.-]+$/.test(file) || (!/\.(png|svg|mp4|webm|ogv)$/.test(file) && !['circuit.json', 'classifier.json', 'training.json'].includes(file))) {
    return new Response('Asset not found', { status: 404 });
  }
  try {
    return await serveFile(resolve('../.pingstore/runs', match[1], 'export', file));
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === 'ENOENT' || code === 'ENOTDIR') return new Response('Asset not found', { status: 404 });
    throw error;
  }
}
