/** A build-time provenance snapshot. It does not control storage or deletion. */
import { execFileSync } from 'node:child_process';
import { runStore } from '../lib/run-store.ts';
export async function GET() {
  const index = await runStore.index();
  const revision = execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
  const value = { build: { revision }, selections: index.selections,
    dependencies: Object.fromEntries(Object.entries(index.runs).map(([id, row]) => [id, row.record.depends_on])) };
  return new Response(JSON.stringify(value, null, 2) + '\n', { headers: { 'Content-Type': 'application/json' } });
}
