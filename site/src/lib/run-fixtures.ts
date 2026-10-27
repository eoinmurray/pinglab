/** Real v4 evidence for tests that cross the Python/TypeScript boundary. */
import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { pathCompare } from './remote-catalog.ts';

export async function saveLocalRun(root: string, id: string, exports: Record<string, string> = { 'result.json': '{}' }, options: {
  parent?: any; day?: string; configuration?: any;
} = {}) {
  const hash = (value: string) => createHash('sha256').update(value).digest('hex');
  const payload = Object.entries(exports).sort(([a], [b]) => pathCompare(a, b))
    .map(([path, value]) => ({ path: `export/${path}`, sha256: hash(value), size_bytes: Buffer.byteLength(value) }));
  const time = `2026-10-${options.day ?? '01'}T00:00:00Z`;
  const record = { schema: 'pingstore.run/v4', run_id: id, experiment: id.slice(0, 6), stage: id.split('-').at(-1),
    collection: 'fixture', origin: 'local', created_at: time, provenance: {},
    payload_digest: `sha256:${hash(JSON.stringify(payload))}`,
    execution: { started_at: time, completed_at: time, configuration: options.configuration ?? { origin: id } },
    inputs: options.parent ? { source: { run_id: options.parent.run_id, payload_digest: options.parent.payload_digest } } : {} };
  await mkdir(join(root, id, 'export'), { recursive: true });
  for (const [path, value] of Object.entries(exports)) await writeFile(join(root, id, 'export', path), value);
  await writeFile(join(root, id, 'README.md'), 'Run history');
  await writeFile(join(root, id, 'run.json'), JSON.stringify(record));
  return record;
}
