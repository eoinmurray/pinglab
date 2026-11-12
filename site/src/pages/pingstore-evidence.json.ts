/** Deploy this exact evidence inventory with every successful static site build. */
import { readFile } from 'node:fs/promises';
import { isDeepStrictEqual } from 'node:util';
import { execFileSync } from 'node:child_process';
import { runStore } from '../lib/run-store.ts';
import { evidenceManifest } from '../lib/run-evidence.ts';

export async function GET() {
  const index = await runStore.index();
  const revision = process.env.GITHUB_SHA ?? execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
  const value = evidenceManifest(index, {
    id: process.env.PINGLAB_BUILD_ID ?? (process.env.GITHUB_RUN_ID ? `${process.env.GITHUB_RUN_ID}/${process.env.GITHUB_RUN_ATTEMPT ?? '1'}` : `local:${revision}`),
    revision,
  });
  if (process.env.PINGLAB_EVIDENCE_PATH) {
    const prepared = JSON.parse(await readFile(process.env.PINGLAB_EVIDENCE_PATH, 'utf8'));
    if (!isDeepStrictEqual(prepared, value)) throw new Error('Rendered evidence differs from the incoming deployment registration');
  }
  return new Response(JSON.stringify(value, null, 2) + '\n', {
    headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-cache' },
  });
}
