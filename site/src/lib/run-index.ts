/** Core owns reporting policy. TypeScript validates and consumes its JSON index. */
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { resolve, join } from 'node:path';
import { runPattern, type Catalog } from './remote-catalog.ts';

export interface IndexRun {
  record: any;
  current: string[];
  status: 'Current' | 'Required' | 'Unused' | 'Unresolved';
  storage: 'Local' | 'R2' | 'Both' | 'Unknown' | 'Unavailable';
  verification: { local: string; r2: string };
  reasons: string[];
  unresolved: string[];
}
export interface RunIndex {
  schema: 'pingstore.index/v1';
  generation: string;
  source: 'auto' | 'local' | 'remote';
  remote_known: boolean;
  selections: Record<string, string>;
  runsets: Record<string, string[]>;
  selection_errors: Record<string, string>;
  report_selections: Record<string, Record<string, string>>;
  report_inputs: Record<string, string[]>;
  operation?: { kind: string; plan_hash: string; run_ids?: string[] } | null;
  publication_unavailable?: string[];
  inventory_generation?: string | null;
  publication: 'known' | 'unresolved';
  runs: Record<string, IndexRun>;
}
export function validateRunIndex(value: any): RunIndex {
  const mapping = (item: any) => item && typeof item === 'object' && !Array.isArray(item);
  const strings = (item: any) => Array.isArray(item) && item.every(value => typeof value === 'string');
  if (value?.schema !== 'pingstore.index/v1' || !/^sha256:[a-f0-9]{64}$/.test(value.generation)
      || !['auto', 'local', 'remote'].includes(value.source) || typeof value.remote_known !== 'boolean'
      || !mapping(value.selections) || !mapping(value.runsets) || !mapping(value.runs)
      || !mapping(value.selection_errors) || !mapping(value.report_selections) || !['known', 'unresolved'].includes(value.publication)) {
    throw new Error('Invalid Pingstore index');
  }
  for (const [id, row] of Object.entries(value.runs) as [string, IndexRun][]) {
    // Unresolved malformed holdings have no trustworthy scientific record.
    if (!mapping(row) || !['Current', 'Required', 'Unused', 'Unresolved'].includes(row.status)
        || !['Local', 'R2', 'Both', 'Unknown', 'Unavailable'].includes(row.storage) || !strings(row.reasons) || !strings(row.unresolved)
        || !strings(row.current) || !mapping(row.verification)) throw new Error(`Invalid index row: ${id}`);
    if (row.record && (!runPattern.test(id) || row.record.run_id !== id || row.record.schema !== 'pingstore.run/v4')) {
      throw new Error(`Invalid index identity: ${id}`);
    }
  }
  for (const [experiment, id] of Object.entries(value.selections) as [string, string][]) {
    const row = value.runs[id];
    const members = value.runsets[experiment];
    if (!/^exp\d{3}$/.test(experiment) || !runPattern.test(id) || !id.startsWith(`${experiment}-`) || !id.endsWith('-present')
        || !row?.record || !strings(members) || !members.includes(id)
        || members.some(member => !value.runs[member]?.record)) throw new Error(`Invalid indexed selection: ${experiment}`);
    const seen = new Set<string>();
    const visiting = new Set<string>();
    const visit = (member: string) => {
      if (visiting.has(member)) throw new Error('Indexed runset contains a cycle');
      if (seen.has(member)) return;
      visiting.add(member);
      const record = value.runs[member]?.record;
      if (!record || !members.includes(member)) throw new Error(`Incomplete indexed runset: ${member}`);
      for (const ref of Object.values(record.inputs) as any[]) {
        if (value.runs[ref.run_id]?.record?.payload_digest !== ref.payload_digest) throw new Error(`Changed indexed pin: ${member}`);
        visit(ref.run_id);
      }
      visiting.delete(member);
      seen.add(member);
    };
    visit(id);
    if (seen.size !== members.length) throw new Error(`Extraneous indexed ancestry: ${experiment}`);
  }
  return value;
}

const repository = fileURLToPath(new URL('../../../', import.meta.url));
export async function resolveRunIndex(options: { root: string; source: string; catalog?: Catalog; control?: any; repository?: string }): Promise<RunIndex> {
  const repo = resolve(options.repository ?? process.env.PINGLAB_REPOSITORY_ROOT ?? repository);
  const venv = join(repo, '.venv/bin/python');
  const python = process.env.PINGLAB_PYTHON ?? (existsSync(venv) ? venv : 'python3');
  const args = ['-m', 'pingstore.cli', 'status', '--root', repo, '--runs-root', options.root,
    '--source', options.source, '--catalog-stdin', '--json'];
  return new Promise((done, reject) => {
    const child = spawn(python, args, { cwd: repo, env: { ...process.env,
      PYTHONPATH: [join(repo, 'tools'), process.env.PYTHONPATH].filter(Boolean).join(':') }, stdio: ['pipe', 'pipe', 'pipe'] });
    const stdout: Buffer[] = [], stderr: Buffer[] = [];
    const timeout = setTimeout(() => { child.kill(); reject(new Error('Pingstore index generation timed out')); }, 120_000);
    child.stdout.on('data', chunk => stdout.push(chunk));
    child.stderr.on('data', chunk => stderr.push(chunk));
    child.on('error', error => { clearTimeout(timeout); reject(error); });
    child.stdin.on('error', () => {}); // An early Python error is reported by close.
    child.on('close', code => {
      clearTimeout(timeout);
      if (code !== 0) return reject(new Error(`Pingstore index: ${Buffer.concat(stderr).toString().trim()}`));
      try { done(validateRunIndex(JSON.parse(Buffer.concat(stdout).toString()))); }
      catch (error) { reject(error); }
    });
    child.stdin.end(JSON.stringify({ catalog: options.catalog ?? null, control: options.control ?? null }));
  });
}
