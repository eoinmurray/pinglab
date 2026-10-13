import { readdir, readFile, stat } from 'node:fs/promises';
import { join, resolve } from 'node:path';

const runsRoot = resolve('../.pingstore/runs');

interface Presentation {
  id: string;
  exportDirectory: string;
  completedAt: number;
}

const selections = new Map<string, Promise<Presentation | undefined>>();

const frozenSelections = process.env.PINGLAB_RUN_SELECTIONS
  ? readFile(process.env.PINGLAB_RUN_SELECTIONS, 'utf8').then(text => JSON.parse(text) as Record<string, string>)
  : undefined;

async function findLatestPresentation(experiment: string): Promise<Presentation | undefined> {
  if (frozenSelections) {
    const id = (await frozenSelections)[experiment];
    if (!id) return undefined;
    const record = JSON.parse(await readFile(join(runsRoot, id, 'run.json'), 'utf8'));
    return { id, exportDirectory: join(runsRoot, id, 'export'), completedAt: Date.parse(record.execution.completed_at) };
  }
  const candidates: Presentation[] = [];
  for (const entry of await readdir(runsRoot, { withFileTypes: true })) {
    if (!entry.isDirectory() || entry.name.startsWith('.') || !entry.name.startsWith(`${experiment}-`)) continue;
    const directory = join(runsRoot, entry.name);
    const record = JSON.parse(await readFile(join(directory, 'run.json'), 'utf8'));
    if (record.experiment !== experiment || record.stage !== 'present') continue;
    candidates.push({
      id: entry.name,
      exportDirectory: join(directory, 'export'),
      completedAt: Date.parse(record.execution.completed_at),
    });
  }
  candidates.sort((a, b) => b.completedAt - a.completedAt || b.id.localeCompare(a.id));
  return candidates[0];
}

// Resolve once per build or dev-server session so a report uses one presentation.
export function optionalPresentation(experiment: string): Promise<Presentation | undefined> {
  if (!selections.has(experiment)) selections.set(experiment, findLatestPresentation(experiment));
  return selections.get(experiment)!;
}

export async function latestPresentation(experiment: string): Promise<Presentation> {
  const run = await optionalPresentation(experiment);
  if (!run) throw new Error(`No presentation run found for ${experiment}`);
  return run;
}

export function imageUrl(runId: string, filename: string): string {
  return `/run-images/${encodeURIComponent(runId)}/${encodeURIComponent(filename)}`;
}

export async function presentationJson(experiment: string, filename: string) {
  const run = await latestPresentation(experiment);
  return JSON.parse(await readFile(join(run.exportDirectory, filename), 'utf8'));
}

interface RunRecord {
  experiment: string;
  stage: string;
  origin: string;
  created_at: string;
  inputs: Record<string, { run_id: string }>;
  execution: { started_at: string; completed_at: string; operation?: string };
  scientific_execution?: { started_at: string; completed_at: string; origin: string };
  completed_hpc_import?: {
    source_execution: { started_at: string; completed_at: string };
    source_origin: string;
  };
}

async function exportSize(directory: string): Promise<number> {
  let bytes = 0;
  // Walk only the selected lineage's exports; keep filesystem IO bounded.
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) bytes += await exportSize(path);
    else if (entry.isFile()) bytes += (await stat(path)).size;
  }
  return bytes;
}

async function findDataset(experiment: string, runId?: string) {
  const selected = runId ? { id: runId } : await latestPresentation(experiment);
  const seen = new Set<string>();
  const upstream = new Set<string>();
  const runs: { id: string; record: RunRecord; bytes: number; selected: boolean }[] = [];

  async function visit(id: string) {
    if (seen.has(id)) return;
    seen.add(id);
    const directory = join(runsRoot, id);
    const record: RunRecord = JSON.parse(await readFile(join(directory, 'run.json'), 'utf8'));
    if (record.experiment !== experiment) {
      upstream.add(record.experiment);
      return;
    }
    runs.push({ id, record, bytes: await exportSize(join(directory, 'export')), selected: id === selected.id });
    for (const input of Object.values(record.inputs)) await visit(input.run_id);
  }

  await visit(selected.id);
  // Keep the selected presentation, plus this experiment's completed stage history.
  for (const entry of await readdir(runsRoot, { withFileTypes: true })) {
    if (!entry.isDirectory() || entry.name.startsWith('.') || !entry.name.startsWith(`${experiment}-`) || seen.has(entry.name)) continue;
    const directory = join(runsRoot, entry.name);
    const record: RunRecord = JSON.parse(await readFile(join(directory, 'run.json'), 'utf8'));
    if (record.experiment !== experiment || !['compute', 'analyse', 'present'].includes(record.stage) || !record.execution.completed_at) continue;
    runs.push({ id: entry.name, record, bytes: await exportSize(join(directory, 'export')), selected: false });
  }
  runs.sort((a, b) => Date.parse(b.record.created_at) - Date.parse(a.record.created_at) || b.id.localeCompare(a.id));
  return { runs, upstream: [...upstream].sort() };
}

const datasets = new Map<string, ReturnType<typeof findDataset>>();

export function dataset(experiment: string, runId?: string) {
  const key = runId ?? experiment;
  if (!datasets.has(key)) datasets.set(key, findDataset(experiment, runId));
  return datasets.get(key)!;
}
