import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';
import { readdir, readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { latestPresentation } from '../../../lib/pingstore';

export async function getStaticPaths() {
  const paths = [];
  const seen = new Set<string>();
  for (const report of await getCollection('reports')) {
    if (!report.data.tags.includes('data')) continue;
    const selections = report.data.runs ? Object.values(report.data.runs).map(id => ({ id, exportDirectory: resolve('../.pingstore/runs', id, 'export') })) : [await latestPresentation(report.id)];
    for (const run of selections) {
      if (seen.has(run.id)) continue;
      seen.add(run.id);
      for (const file of await readdir(run.exportDirectory)) {
        if (!/\.(png|svg|mp4|webm|ogv)$/.test(file)) continue;
        paths.push({ params: { run: run.id, file }, props: { path: join(run.exportDirectory, file) } });
      }
    }
  }
  return paths;
}

export const GET: APIRoute = async ({ props }) => {
  const bytes = await readFile(props.path);
  const type = ({ svg: 'image/svg+xml', png: 'image/png', mp4: 'video/mp4', webm: 'video/webm', ogv: 'video/ogg' } as Record<string, string>)[props.path.split('.').at(-1)];
  return new Response(new Uint8Array(bytes), { headers: { 'Content-Type': type } });
};
