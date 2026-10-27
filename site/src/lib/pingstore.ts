import { runStore } from './run-store.ts';
import { localAssetUrl, selectAsset } from './run-assets.ts';

async function countCompletedRuns(experiment: string): Promise<number> {
  const index = await runStore.index();
  return Object.values(index.runs).filter(row => row.record?.experiment === experiment).length;
}

const runCounts = new Map<string, Promise<number>>();
export function completedRunCount(experiment: string): Promise<number> {
  if (process.env.NODE_ENV !== 'production') return countCompletedRuns(experiment);
  if (!runCounts.has(experiment)) runCounts.set(experiment, countCompletedRuns(experiment));
  return runCounts.get(experiment)!;
}

interface Presentation {
  id: string;
  completedAt: number;
}

const selections = new Map<string, Promise<Presentation | undefined>>();

async function findLatestPresentation(experiment: string): Promise<Presentation | undefined> {
  const id = await runStore.latest(experiment);
  if (!id) return undefined;
  const record = await runStore.json(id);
  return { id, completedAt: Date.parse(record.execution.completed_at) };
}

// Freeze builds; development shares the current request’s core selection.
export function optionalPresentation(experiment: string): Promise<Presentation | undefined> {
  if (process.env.NODE_ENV !== 'production') return findLatestPresentation(experiment);
  if (!selections.has(experiment)) selections.set(experiment, findLatestPresentation(experiment));
  return selections.get(experiment)!;
}

export async function latestPresentation(experiment: string): Promise<Presentation> {
  const run = await optionalPresentation(experiment);
  if (!run) throw new Error(`No presentation run found for ${experiment}`);
  return run;
}

export async function imageUrl(runId: string, filename: string): Promise<string> {
  if (process.env.NODE_ENV !== 'production') return localAssetUrl(runId, filename);
  return (await selectAsset(runId, filename)).url;
}

export async function figureUrl(runId: string, filename: string): Promise<string> {
  if (!/\.(svg|png)$/.test(filename)) return imageUrl(runId, filename);
  const stem = filename.replace(/\.(svg|png)$/, '');
  for (const extension of ['svg', 'png']) {
    const candidate = `${stem}.${extension}`;
    if (await runStore.has(runId, `export/${candidate}`)) return imageUrl(runId, candidate);
  }
  throw new Error(`Missing figure ${filename} in presentation ${runId} (SVG or PNG required)`);
}

export async function presentationJson(experiment: string, filename: string) {
  const run = await latestPresentation(experiment);
  return runStore.json(run.id, `export/${filename}`);
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

async function exportSize(id: string): Promise<number> {
  return [...(await runStore.files(id)).values()].reduce((sum, bytes) => sum + bytes, 0);
}

async function findDataset(experiment: string, runId?: string) {
  const selected = runId ? { id: runId } : await optionalPresentation(experiment);
  const index = await runStore.index();
  const seen = new Set<string>();
  const upstream = new Set<string>();
  const runs: { id: string; record: RunRecord; bytes: number; selected: boolean; used: boolean; evidence: import('./run-index.ts').IndexRun }[] = [];

  async function visit(id: string) {
    if (seen.has(id)) return;
    seen.add(id);
    const record: RunRecord = await runStore.json(id);
    if (record.experiment !== experiment) {
      upstream.add(record.experiment);
    }
    runs.push({ id, record, bytes: await exportSize(id), selected: id === selected?.id, used: true, evidence: index.runs[id] });
    for (const input of Object.values(record.inputs)) await visit(input.run_id);
  }

  if (selected) await visit(selected.id);
  // Keep the selected lineage, plus this experiment's completed stage history.
  for (const id of Object.keys(index.runs).filter(id => id.startsWith(`${experiment}-`))) {
    if (seen.has(id)) continue;
    const record: RunRecord = index.runs[id].record;
    if (!record) continue;
    runs.push({ id, record, bytes: await exportSize(id), selected: false, used: false, evidence: index.runs[id] });
  }
  runs.sort((a, b) => Date.parse(b.record.created_at) - Date.parse(a.record.created_at) || b.id.localeCompare(a.id));
  return { runs, upstream: [...upstream].sort() };
}

const datasets = new Map<string, ReturnType<typeof findDataset>>();

export function dataset(experiment: string, runId?: string) {
  if (process.env.NODE_ENV !== 'production') return findDataset(experiment, runId);
  const key = runId ?? experiment;
  if (!datasets.has(key)) datasets.set(key, findDataset(experiment, runId));
  return datasets.get(key)!;
}
