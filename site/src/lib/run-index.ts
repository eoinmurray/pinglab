/** Metadata-only selection. Storage and deployments have no reporting policy. */
export interface RunRecord {
  id: string;
  experiment: string;
  stage?: string;
  created_at: string;
  description?: string;
  depends_on: string[];
  metadata?: { duration_seconds?: number; origin?: string; [key: string]: unknown };
}
export interface IndexRun {
  record: RunRecord;
  storage: 'Local' | 'R2' | 'Both';
  bytes?: number;
  file_count?: number;
}
export interface RunIndex {
  source: string;
  selections: Record<string, string>;
  runsets: Record<string, string[]>;
  runs: Record<string, IndexRun>;
}
export const runPattern = /^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,159}$/;
export function validateRecord(value: any, id: string): RunRecord {
  const fields = new Set(['id', 'experiment', 'stage', 'created_at', 'description', 'depends_on', 'metadata']);
  if (!value || typeof value !== 'object' || Object.keys(value).some(key => !fields.has(key))
      || !runPattern.test(id) || value.id !== id || typeof value.experiment !== 'string'
      || !value.experiment || typeof value.created_at !== 'string' || !Number.isFinite(Date.parse(value.created_at))
      || !/(Z|[+-]\d\d:\d\d)$/.test(value.created_at) || !Array.isArray(value.depends_on)
      || value.depends_on.some((item: unknown) => typeof item !== 'string' || !runPattern.test(item) || item === id)
      || new Set(value.depends_on).size !== value.depends_on.length
      || (value.stage !== undefined && typeof value.stage !== 'string')
      || (value.description !== undefined && typeof value.description !== 'string')
      || (value.metadata !== undefined && (!value.metadata || typeof value.metadata !== 'object' || Array.isArray(value.metadata)))) {
    throw new Error(`Invalid run metadata: ${id}; migrate old records before using the simple store`);
  }
  return value;
}
export function ancestry(runs: Record<string, IndexRun>, root: string): string[] {
  const seen = new Set<string>();
  const visit = (id: string) => {
    if (seen.has(id)) return;
    seen.add(id);
    for (const dependency of runs[id]?.record.depends_on ?? []) visit(dependency);
  };
  visit(root);
  return [...seen];
}
export function buildRunIndex(runs: Record<string, IndexRun>, source = 'local'): RunIndex {
  const selections: Record<string, string> = {};
  for (const [id, row] of Object.entries(runs)) {
    const record = validateRecord(row.record, id);
    if (record.stage !== 'present') continue;
    const previous = runs[selections[record.experiment]]?.record;
    const delta = previous ? Date.parse(record.created_at) - Date.parse(previous.created_at) : 1;
    if (!previous || delta > 0 || (delta === 0 && id > previous.id)) selections[record.experiment] = id;
  }
  return { source, runs, selections, runsets: Object.fromEntries(Object.entries(selections)
    .map(([experiment, id]) => [experiment, ancestry(runs, id)])) };
}
