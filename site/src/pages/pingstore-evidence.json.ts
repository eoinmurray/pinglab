/** Deploy this exact evidence inventory with every successful static site build. */
import { readFile } from 'node:fs/promises';
import { isDeepStrictEqual } from 'node:util';
import { execFileSync } from 'node:child_process';
import { runStore } from '../lib/run-store.ts';

export async function GET() {
  const index = await runStore.index();
  if (Object.keys(index.selection_errors).length) throw new Error('Cannot publish unresolved runset selections');
  const identities = [...new Set(Object.values(index.runsets).flat())].sort();
  const revision = process.env.GITHUB_SHA ?? execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
  const value = {
    schema: 'pingstore.evidence/v1',
    build: {
      id: process.env.GITHUB_RUN_ID ? `${process.env.GITHUB_RUN_ID}/${process.env.GITHUB_RUN_ATTEMPT ?? '1'}` : `local:${revision}`,
      revision,
    },
    source: index.source,
    index_generation: index.generation,
    selections: index.selections,
    reports: index.report_selections,
    unavailable_reports: Object.fromEntries(Object.entries(index.report_inputs)
      .map(([article, inputs]) => [article, inputs.filter(experiment => !index.selections[experiment]).sort()])
      .filter(([, inputs]) => inputs.length)),
    runs: Object.fromEntries(identities.map(id => {
      const record = index.runs[id].record;
      return [id, { payload_digest: record.payload_digest, inputs: record.inputs }];
    })),
  };
  if (process.env.PINGLAB_EVIDENCE_PATH) {
    const prepared = JSON.parse(await readFile(process.env.PINGLAB_EVIDENCE_PATH, 'utf8'));
    if (!isDeepStrictEqual(prepared, value)) throw new Error('Rendered evidence differs from the incoming deployment registration');
  }
  return new Response(JSON.stringify(value, null, 2) + '\n', {
    headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-cache' },
  });
}
