/** Read original native snapshots through one explicitly pinned stage lineage. */
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

interface Reference { run_id: string; payload_digest: string; }
interface RunRecord extends Reference {
  schema: string;
  experiment: string;
  stage: 'compute' | 'analyse' | 'present';
  inputs: Record<string, Reference>;
  execution: { completed_at: string; configuration: Record<string, any>; dataset?: Record<string, any> };
  scientific_artifacts?: { native_paths?: Record<string, string> };
}
export interface SavedSnapshot {
  schema: string;
  stage: 'start' | 'end';
  kind: 'train' | 'simulate' | 'infer';
  parameters_included: boolean;
  network: Record<string, any>;
  training: Record<string, any>;
  execution: Record<string, any>;
  [key: string]: any;
}
export interface SnapshotEvidence {
  compute: Reference;
  configuration: Record<string, any>;
  dataset?: Record<string, any>;
  snapshots: Record<string, SavedSnapshot | undefined>;
}
export interface SnapshotOptions {
  experiment: string;
  snapshots: Record<string, string>;
  /** Select a named analysis branch when a presentation combines protocols. */
  analysisInput?: string;
  runsRoot?: string;
  allowUnavailable?: boolean;
}

function relativePath(path: string): string {
  if (typeof path !== 'string' || path.split('/').some(part => !part || part === '.' || part === '..')
      || path.includes('\\') || path.includes('\0')) throw new Error(`Invalid native snapshot path: ${path}`);
  return path;
}

export async function readSnapshotEvidence(presentation: string, options: SnapshotOptions): Promise<SnapshotEvidence | undefined> {
  const { experiment, snapshots, allowUnavailable = false } = options;
  if (!/^exp\d{3}$/.test(experiment)) throw new Error('Invalid snapshot experiment');
  const root = resolve(options.runsRoot ?? '../.pingstore/runs');
  const readRun = async (identity: string, stage: RunRecord['stage'], reference?: Reference) => {
    if (!new RegExp(`^${experiment}-r\\d+-${stage}$`).test(identity)) throw new Error(`Invalid ${stage} identity: ${identity}`);
    let record: RunRecord;
    try { record = JSON.parse(await readFile(resolve(root, identity, 'run.json'), 'utf8')); }
    catch (error) {
      if (allowUnavailable && (error as NodeJS.ErrnoException).code === 'ENOENT') return undefined;
      throw error;
    }
    if (record.schema !== 'pingstore.run/v4' || record.run_id !== identity || record.experiment !== experiment
        || record.stage !== stage || !record.execution?.completed_at || !record.execution.configuration
        || !/^sha256:[a-f0-9]{64}$/.test(record.payload_digest)) throw new Error(`Invalid completed ${stage} record: ${identity}`);
    if (reference && record.payload_digest !== reference.payload_digest) throw new Error(`Pinned input digest differs: ${identity}`);
    return record;
  };
  const readInput = async (record: RunRecord, stage: RunRecord['stage'], input?: string) => {
    const candidates = input === undefined ? Object.values(record.inputs) : [record.inputs[input]].filter(Boolean);
    const references = candidates.filter(reference =>
      new RegExp(`^${experiment}-r\\d+-${stage}$`).test(reference.run_id));
    if (references.length !== 1) throw new Error(`Expected one ${stage} input in ${record.run_id}`);
    return readRun(references[0].run_id, stage, references[0]);
  };
  const selected = await readRun(presentation, 'present');
  if (!selected) return undefined;
  const analysis = await readInput(selected, 'analyse', options.analysisInput);
  if (!analysis) return undefined;
  const compute = await readInput(analysis, 'compute');
  if (!compute) return undefined;
  const loaded = await Promise.all(Object.entries(snapshots).map(async ([name, original]) => {
    relativePath(original);
    const retained = compute.scientific_artifacts?.native_paths?.[original];
    if (retained === undefined) {
      if (allowUnavailable) return [name, undefined] as const;
      throw new Error(`No saved native snapshot ${original} in ${compute.run_id}`);
    }
    // Pingstore records the original-to-canonical mapping. Do not invent another
    // serializer, copy metadata into numbers.json, or guess flattened filenames.
    const snapshot: SavedSnapshot = JSON.parse(await readFile(resolve(root, compute.run_id, 'export', relativePath(retained)), 'utf8'));
    if (snapshot.schema !== 'snnlab.run-snapshot/v1' || !snapshot.parameters_included
        || snapshot.stage !== original.split('/').at(-2)
        || !['train', 'simulate', 'infer'].includes(snapshot.kind)
        || snapshot.network?.schema !== 'snnlang.graph/v1') throw new Error(`Invalid native snapshot: ${compute.run_id}/${original}`);
    return [name, snapshot] as const;
  }));
  return {
    compute: { run_id: compute.run_id, payload_digest: compute.payload_digest },
    configuration: compute.execution.configuration,
    dataset: compute.execution.dataset,
    snapshots: Object.fromEntries(loaded),
  };
}
