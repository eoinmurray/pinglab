import { isDeepStrictEqual } from 'node:util';
import type { RunIndex } from './run-index.ts';

/** Match pingstore.core.evidence_manifest using the frozen core index. */
export function evidenceManifest(index: RunIndex, build: { id: string; revision: string }) {
  if (Object.keys(index.selection_errors).length) throw new Error('Cannot publish unresolved runset selections');
  const selections: Record<string, string> = Object.assign({}, ...Object.values(index.report_selections));
  const identities = [...new Set(Object.keys(selections).flatMap(experiment => index.runsets[experiment]))].sort();
  return {
    schema: 'pingstore.evidence/v1', build, source: index.source, index_generation: index.generation,
    selections, reports: index.report_selections,
    unavailable_reports: Object.fromEntries(Object.entries(index.report_inputs)
      .map(([article, inputs]) => [article, [...new Set(inputs.filter(experiment => !index.selections[experiment]))].sort()])
      .filter(([, inputs]) => inputs.length)),
    runs: Object.fromEntries(identities.map(id => {
      const record = index.runs[id].record;
      return [id, { payload_digest: record.payload_digest, inputs: record.inputs,
        ...(!isDeepStrictEqual(record.dependencies, record.inputs) ? { runset_inputs: record.dependencies } : {}),
      }];
    })),
  };
}
