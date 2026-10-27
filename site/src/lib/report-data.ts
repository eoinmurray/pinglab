import { runStore } from './run-store.ts';
import { optionalPresentation } from './pingstore.ts';
import { readSnapshotEvidence, type SnapshotOptions } from './snapshot-data.ts';

const payloads = new Map<string, Promise<any>>();

interface Options { experiment?: string; allowUnavailable?: boolean; }

export async function reportHasData(report: string, experiment = report): Promise<boolean> {
  return Boolean(await optionalPresentation(experiment));
}

// Absence is distinct from malformed selected evidence, which still throws.
export async function unavailableReportInputs(inputs: string[], requiresEvidence: boolean): Promise<string[]> {
  if (!requiresEvidence) return [];
  const available = await Promise.all(inputs.map(async experiment => ({ experiment, run: await optionalPresentation(experiment) })));
  return available.filter(value => !value.run).map(value => value.experiment);
}

// Numbers, figures and snapshots share the request or build selection.
export async function reportData(report: string, { experiment = report, allowUnavailable = false }: Options = {}): Promise<any> {
  const run = (await optionalPresentation(experiment))?.id;
  if (!run) {
    if (allowUnavailable) return undefined;
    throw new Error(`No presentation data for ${report}/${experiment}`);
  }
  if (!/^exp\d{3}-r\d+-present$/.test(run)) throw new Error(`Invalid presentation ID: ${run}`);
  if (process.env.NODE_ENV !== 'production') return runStore.json(run, 'export/numbers.json');
  if (!payloads.has(run)) {
    payloads.set(run, runStore.json(run, 'export/numbers.json'));
  }
  return payloads.get(run)!;
}

// Follow the selected presentation to its original compute snapshots.
// Missing native evidence never substitutes another lineage.
export async function reportSnapshots(report: string, options: Omit<SnapshotOptions, 'experiment'> & { experiment?: string }) {
  const experiment = options.experiment ?? report;
  const selected = await optionalPresentation(experiment);
  if (!selected) {
    if (options.allowUnavailable) return undefined;
    throw new Error(`No presentation data for ${report}/${experiment}`);
  }
  return readSnapshotEvidence(selected.id, { ...options, experiment });
}

export function display(value: unknown): string {
  if (value === undefined || value === null) throw new Error('Missing authored value');
  if (typeof value === 'number' && !Number.isFinite(value)) throw new Error('Nonfinite authored value');
  if (!['string', 'number', 'boolean'].includes(typeof value)) throw new Error('Expected a scalar authored value');
  return String(value);
}

export function round(value: number, digits = 0): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error('Invalid authored numerical value');
  return Number(value.toFixed(digits));
}

export const mean = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length;
export const percent = (value: number) => `${round(100 * value, 1)}%`;
export const nullableRound = (value: number | null, digits = 3) => value === null ? 'not defined' : String(round(value, digits));
export const joinLast = (items: unknown[]) => items.length < 2 ? items.join('') : `${items.slice(0, -1).join(', ')} and ${items.at(-1)}`;
