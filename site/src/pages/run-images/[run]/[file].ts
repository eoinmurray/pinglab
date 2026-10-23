import type { APIRoute } from 'astro';
import { getReports } from '../../../lib/reports';
import { runStore } from '../../../lib/run-store';
import { latestPresentation } from '../../../lib/pingstore';
import { assetTypes, isRunAsset, selectAsset } from '../../../lib/run-assets';

export async function getStaticPaths() {
  const paths = [];
  const seen = new Set<string>();
  for (const report of await getReports()) {
    if (!report.data.tags.includes('data')) continue;
    let selections;
    try {
      selections = report.data.runs ? Object.values(report.data.runs).map(id => ({ id })) : [await latestPresentation(report.id)];
    } catch (error) {
      console.error(`[run-images] ${report.id}:`, error);
      continue;
    }
    for (const run of selections) {
      if (seen.has(run.id)) continue;
      seen.add(run.id);
      let files;
      try {
        files = [...(await runStore.files(run.id)).keys()];
      } catch (error) {
        console.error(`[run-images] ${report.id}/${run.id}:`, error);
        continue;
      }
      const local = await Promise.all(files.filter(isRunAsset).map(async file => {
        if ((await selectAsset(run.id, file)).source === 'r2') return undefined;
        return { params: { run: run.id, file }, props: { run: run.id, file } };
      }));
      paths.push(...local.filter(path => path !== undefined));
    }
  }
  return paths;
}

async function serveFile(run: string, file: string): Promise<Response> {
  const bytes = await runStore.read(run, `export/${file}`);
  const type = assetTypes[file.split('.').at(-1)!];
  return new Response(new Uint8Array(bytes), { headers: {
    'Content-Type': type,
    'Cache-Control': import.meta.env.DEV ? 'no-store' : 'public, max-age=31536000, immutable',
  } });
};

export const GET: APIRoute = async ({ props }) => serveFile(props.run, props.file);

// Development serves one requested asset without enumerating unrelated reports.
export async function serveRunAsset(pathname: string, origin = 'http://localhost:3000'): Promise<Response> {
  const match = /^\/run-images\/(exp\d{3}-r\d+-present)\/([^/]+)$/.exec(pathname);
  if (!match) return new Response('Asset not found', { status: 404 });
  let file: string;
  try { file = decodeURIComponent(match[2]); }
  catch { return new Response('Asset not found', { status: 404 }); }
  if (!isRunAsset(file)) {
    return new Response('Asset not found', { status: 404 });
  }
  try {
    const selection = await selectAsset(match[1], file, { development: true, origin });
    if (selection.source === 'r2' && !file.endsWith('.json')) return new Response(null, { status: 302, headers: {
      Location: selection.url, 'Cache-Control': 'no-store',
    } });
    return await serveFile(match[1], file);
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === 'ENOENT' || code === 'ENOTDIR') return new Response('Asset not found', { status: 404 });
    throw error;
  }
}
