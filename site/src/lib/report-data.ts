import { getEntry } from 'astro:content';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { optionalPresentation } from './pingstore';

const payloads = new Map<string, Promise<any>>();

interface Options { experiment?: string; allowUnavailable?: boolean; }

export async function reportHasData(report: string, experiment = report): Promise<boolean> {
  const entry = await getEntry('reports', report);
  return Boolean(entry?.data.runs?.[experiment] || await optionalPresentation(experiment));
}

// Use the article's explicit pin, otherwise the session's frozen presentation selection.
export async function reportData(report: string, { experiment = report, allowUnavailable = false }: Options = {}): Promise<any> {
  const entry = await getEntry('reports', report);
  const run = entry?.data.runs?.[experiment] ?? (await optionalPresentation(experiment))?.id;
  if (!run) {
    if (allowUnavailable) return undefined;
    throw new Error(`No presentation data for ${report}/${experiment}`);
  }
  if (!/^exp\d{3}-r\d+-present$/.test(run)) throw new Error(`Invalid presentation ID: ${run}`);
  if (!payloads.has(run)) {
    payloads.set(run, readFile(resolve('../.pingstore/runs', run, 'export/numbers.json'), 'utf8').then(JSON.parse));
  }
  return payloads.get(run)!;
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
