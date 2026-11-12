import { ancestry, type RunIndex, type IndexRun } from './run-index.ts';
export interface RunTreeNode {
  id: string;
  row?: IndexRun;
  issue?: string;
  children: RunTreeNode[];
}
export function runTree(index: RunIndex, root: string, path = new Set<string>()): RunTreeNode {
  const row = index.runs[root];
  const issue = path.has(root) ? 'Dependency cycle' : !row ? 'Run unavailable' : undefined;
  return { id: root, row, issue, children: issue ? [] : row.record.depends_on
    .map(id => runTree(index, id, new Set([...path, root]))) };
}
export function reportRuns(index: RunIndex, experiments: string[]) {
  return new Set(experiments.flatMap(experiment => index.selections[experiment]
    ? ancestry(index.runs, index.selections[experiment]) : []));
}
