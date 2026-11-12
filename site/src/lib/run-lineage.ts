import type { RunIndex, IndexRun } from './run-index';

interface Pin { run_id: string; payload_digest: string }

function dependencyLinks(record: any): [string, Pin][] {
  const originalSources = record?.execution?.operation === 'consolidate';
  const inputs = Object.entries(record?.inputs ?? {}) as [string, Pin][];
  const runsetInputs = Object.entries(record?.runset_inputs ?? {}) as [string, Pin][];
  const experiment = record?.experiment ?? record?.run_id?.split('-')[0];
  return [
    ...runsetInputs,
    ...inputs.filter(([role, pin]) => {
      if (originalSources && /^original_\d+$/.test(role)) return false;
      // Reanalysis can replace an old analysis link with its exact compute pin.
      const promotedCompute = runsetInputs.some(([, ref]) =>
        ref.run_id.startsWith(`${experiment}-`) && ref.run_id.endsWith('-compute'));
      const historyStage = record?.stage === 'present' || (record?.stage === 'analyse' && promotedCompute);
      const detachedHistory = record?.runset_inputs && historyStage
        && pin.run_id.startsWith(`${experiment}-`) && pin.run_id.endsWith(`-${record.stage}`)
        && !runsetInputs.some(([, ref]) => ref.run_id === pin.run_id && ref.payload_digest === pin.payload_digest);
      return !detachedHistory;
    }),
  ];
}

function resolvedLinks(index: RunIndex, record: any): [string, Pin][] {
  return dependencyLinks(record).map(([role, pin]) => {
    const view = index.source_views?.[pin.run_id];
    return [role, !index.runs[pin.run_id] && view?.source_digest === pin.payload_digest
      ? { run_id: view.run_id, payload_digest: view.payload_digest } : pin];
  });
}
export interface LineageEntry {
  id: string;
  row?: IndexRun;
  reports: string[];
  requiredBy: string[];
  issues: string[];
}

/** Follow dependencies, excluding archived sources already absorbed by consolidation. */
export function lineageLists(index: RunIndex) {
  const entries = new Map<string, LineageEntry>();
  const entry = (id: string) => {
    if (!entries.has(id)) entries.set(id, {
      id, row: index.runs[id], reports: [], requiredBy: [], issues: [],
    });
    return entries.get(id)!;
  };
  const pending: string[] = [];
  for (const [report, selections] of Object.entries(index.report_selections)) {
    for (const id of Object.values(selections)) {
      const item = entry(id);
      if (!item.reports.includes(report)) item.reports.push(report);
      pending.push(id);
    }
  }
  const visited = new Set<string>();
  while (pending.length) {
    const id = pending.pop()!;
    if (visited.has(id)) continue;
    visited.add(id);
    const item = entry(id);
    const record = item.row?.record;
    if (!record) {
      item.issues.push('Run record unavailable');
      continue;
    }
    for (const [, ref] of resolvedLinks(index, record)) {
      const dependency = entry(ref.run_id);
      if (!dependency.requiredBy.includes(id)) dependency.requiredBy.push(id);
      if (dependency.row?.record && dependency.row.record.payload_digest !== ref.payload_digest) {
        const issue = `Digest mismatch in input from ${id}`;
        if (!dependency.issues.includes(issue)) dependency.issues.push(issue);
      }
      pending.push(ref.run_id);
    }
  }
  const descending = (a: LineageEntry, b: LineageEntry) => b.id.localeCompare(a.id, 'en', { numeric: true });
  const inUse = [...entries.values()].sort(descending);
  const other = Object.keys(index.runs).filter(id => !entries.has(id)).map(entry).sort(descending);
  return { inUse, other };
}

export interface RunTreeNode {
  entry: LineageEntry;
  roles: string[];
  issue?: string;
  repeated: boolean;
  children: RunTreeNode[];
}

function cleanTriplet(index: RunIndex, experiment: string, ids: string[], expectedStages: string[]): boolean {
  if (ids.length !== expectedStages.length) return false;
  const stages = ['compute', 'analyse', 'present'];
  if (!expectedStages.every(stage => ids.filter(id => index.runs[id]?.record?.stage === stage).length === 1)) return false;
  const seen = new Set<string>();
  let externalCompute = false;
  const visit = (id: string): boolean => {
    if (seen.has(id)) return true;
    seen.add(id);
    const record = index.runs[id]?.record;
    if (!record) return false;
    return resolvedLinks(index, record).every(([, pin]) => {
      const parent = index.runs[pin.run_id]?.record;
      if ((parent?.experiment ?? pin.run_id.split('-')[0]) !== experiment) {
        if (parent?.stage === 'compute') externalCompute = true;
        return true;
      }
      return ids.includes(pin.run_id) && parent?.payload_digest === pin.payload_digest
        && stages.indexOf(parent.stage) < stages.indexOf(record.stage) && visit(pin.run_id);
    });
  };
  const presentation = ids.find(id => index.runs[id]?.record?.stage === 'present')!;
  return visit(presentation) && seen.size === expectedStages.length && (expectedStages.includes('compute') || externalCompute);
}

/** Group presentation roots and unpresented branches without changing use classification. */
export function lineageGroups(index: RunIndex, workflows: Record<string, string[]> = {}) {
  const lists = lineageLists(index);
  const entries = new Map([...lists.inUse, ...lists.other].map(item => [item.id, item]));
  const used = new Set(lists.inUse.map(item => item.id));
  const experimentOf = (id: string) => index.runs[id]?.record?.experiment ?? id.split('-')[0];
  const links = (id: string) => {
    const record = index.runs[id]?.record;
    const refs = new Map<string, { pin: Pin; roles: string[] }>();
    for (const [role, pin] of resolvedLinks(index, record)) {
        const key = `${pin.run_id}:${pin.payload_digest}`;
        if (!refs.has(key)) refs.set(key, { pin, roles: [] });
        if (!refs.get(key)!.roles.includes(role)) refs.get(key)!.roles.push(role);
    }
    return [...refs.values()];
  };
  const forest = (ids: string[]) => {
    const covered = new Set<string>();
    const tree = (id: string, seen: Set<string>, path: Set<string>, roles: string[] = [], pin?: Pin): RunTreeNode => {
      const entry = entries.get(id) ?? { id, reports: [], requiredBy: [], issues: ['Run record unavailable'] };
      const issue = path.has(id) ? 'Dependency cycle'
        : pin && entry.row?.record && entry.row.record.payload_digest !== pin.payload_digest ? 'Digest mismatch' : undefined;
      const repeated = seen.has(id) && !issue;
      const result: RunTreeNode = { entry, roles, issue, repeated, children: [] };
      covered.add(id);
      if (issue || repeated) return result;
      seen.add(id);
      result.children = links(id).map(({ pin, roles }) => tree(pin.run_id, seen, new Set([...path, id]), roles, pin));
      return result;
    };
    const roots: RunTreeNode[] = [];
    const add = (id: string) => roots.push(tree(id, new Set(), new Set()));
    const presentations = ids.filter(id => index.runs[id]?.record?.stage === 'present');
    for (const id of presentations) add(id);
    const referenced = new Set(ids.flatMap(id => links(id).map(({ pin }) => pin.run_id)));
    for (const id of ids) if (!covered.has(id) && !referenced.has(id)) add(id);
    for (const id of ids) if (!covered.has(id)) add(id);
    return roots;
  };
  const experiments = [...new Set([...entries.keys()].map(experimentOf))]
    .sort((a, b) => b.localeCompare(a, 'en', { numeric: true }));
  const groups = experiments.map(experiment => {
    const ids = [...entries.keys()].filter(id => experimentOf(id) === experiment)
      .sort((a, b) => b.localeCompare(a, 'en', { numeric: true }));
    return { experiment, clean: cleanTriplet(index, experiment, ids, workflows[experiment] ?? ['compute', 'analyse', 'present']),
      inUse: forest(ids.filter(id => used.has(id))), unused: forest(ids.filter(id => !used.has(id))) };
  });
  return { groups, used };
}
