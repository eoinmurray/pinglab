/** Read the selected report's explicitly named native evidence files. */
import { RunStore, runStore } from './run-store.ts';
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
  compute: { run_id: string };
  configuration: Record<string, any>;
  dataset?: Record<string, any>;
  snapshots: Record<string, SavedSnapshot | undefined>;
}
export interface SnapshotOptions {
  experiment: string;
  snapshots: Record<string, string>;
  analysisInput?: string;
  computeInput?: string;
  runsRoot?: string;
  store?: RunStore;
  allowUnavailable?: boolean;
}
export async function readSnapshotEvidence(presentation: string, options: SnapshotOptions): Promise<SnapshotEvidence | undefined> {
  const store = options.store ?? (options.runsRoot ? new RunStore({ runsRoot: options.runsRoot }) : runStore);
  const input = async (identity: string, stage: string, role?: string) => {
    const record = await store.json(identity);
    const details = await store.json(identity, 'details.json');
    const candidates: string[] = role === undefined ? record.depends_on
      : [details.inputs?.[role]?.run_id].filter(Boolean);
    if (candidates.some(id => !record.depends_on.includes(id))) throw new Error(`Snapshot source is not a dependency of ${identity}`);
    const matches = (await Promise.all(candidates.map(async id => ({ id, record: await store.json(id) }))))
      .filter(({ record }) => record.stage === stage && (role !== undefined || record.experiment === options.experiment));
    if (matches.length !== 1) throw new Error(`Expected one ${stage} source in ${identity}; name its input role explicitly`);
    return matches[0].id;
  };
  try {
    const record = await store.json(presentation);
    if (record.experiment !== options.experiment || record.stage !== 'present') throw new Error('Invalid presentation source');
    let analysis = await input(presentation, 'analyse', options.analysisInput);
    const visited = new Set<string>();
    let compute: string | undefined;
    while (!compute) {
      if (visited.has(analysis)) throw new Error('Cyclic snapshot dependency');
      visited.add(analysis);
      const current = await store.json(analysis);
      const parents = await Promise.all(current.depends_on.map(async (id: string) => ({ id, record: await store.json(id) })));
      if (parents.some(({ record }) => record.stage === 'compute' && record.experiment === options.experiment)) compute = await input(analysis, 'compute', options.computeInput);
      else analysis = await input(analysis, 'analyse');
    }
    const details = await store.json(compute, 'details.json');
    const loaded = await Promise.all(Object.entries(options.snapshots).map(async ([name, original]) => {
      const retained = details.scientific_artifacts?.native_paths?.[original] ?? original;
      const snapshot: SavedSnapshot = await store.json(compute!, `export/${retained}`);
      if (snapshot.schema !== 'snnlab.run-snapshot/v1' || !snapshot.parameters_included
          || !['train', 'simulate', 'infer'].includes(snapshot.kind) || snapshot.network?.schema !== 'snnlang.graph/v1') {
        throw new Error(`Invalid native snapshot: ${compute}/${original}`);
      }
      return [name, snapshot];
    }));
    return { compute: { run_id: compute }, configuration: details.execution.configuration,
      dataset: details.execution.dataset, snapshots: Object.fromEntries(loaded) };
  } catch (error) {
    if (options.allowUnavailable && (error as NodeJS.ErrnoException).code === 'ENOENT') return undefined;
    throw error;
  }
}
