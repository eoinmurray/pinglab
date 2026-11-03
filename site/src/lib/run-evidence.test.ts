import { expect, test } from 'bun:test';
import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { RunIndex } from './run-index.ts';
import { evidenceManifest } from './run-evidence.ts';

test('publication evidence matches Python with corrected links and unreported selections', () => {
  const digest = `sha256:${'a'.repeat(64)}`;
  const pin = (id: string) => ({ run_id: id, payload_digest: digest });
  const record = (id: string, inputs = {}, dependencies = inputs) => ({
    record: { run_id: id, payload_digest: digest, inputs, dependencies },
  });
  const index = {
    source: 'remote', generation: digest, selection_errors: {},
    selections: { exp001: 'exp001-r003-present', exp002: 'exp002-r001-present' },
    report_selections: { article: { exp001: 'exp001-r003-present' } },
    report_inputs: { article: ['exp001'], unavailable: ['exp003', 'exp003'] },
    runsets: { exp001: ['exp001-r001-compute', 'exp001-r003-present'], exp002: ['exp002-r001-present'] },
    runs: {
      'exp001-r001-compute': record('exp001-r001-compute'),
      'exp001-r003-present': record('exp001-r003-present', { historical: pin('exp001-r002-compute') },
        { compute: pin('exp001-r001-compute') }),
      'exp002-r001-present': record('exp002-r001-present'),
    },
  } as unknown as RunIndex;
  const build = { id: '123/1', revision: 'b'.repeat(40) };
  const root = fileURLToPath(new URL('../../../', import.meta.url));
  const venv = `${root}.venv/bin/python`;
  const expected = JSON.parse(execFileSync(existsSync(venv) ? venv : 'python3', ['-c',
    'import json,sys; from pingstore.core import evidence_manifest; v=json.load(sys.stdin); print(json.dumps(evidence_manifest(v["index"],v["build"]["id"],v["build"]["revision"])))'], {
    cwd: root, env: { ...process.env, PYTHONPATH: `${root}libraries` },
    input: JSON.stringify({ index, build }), encoding: 'utf8',
  }));
  expect(evidenceManifest(index, build)).toEqual(expected);
  expect(Object.keys(expected.runs)).toEqual(['exp001-r001-compute', 'exp001-r003-present']);
  expect(expected.runs['exp001-r003-present'].runset_inputs).toEqual({ compute: pin('exp001-r001-compute') });
});

test('unresolved selections still prevent publication', () => {
  expect(() => evidenceManifest({ selection_errors: { exp001: 'missing' } } as unknown as RunIndex,
    { id: '123/1', revision: 'b'.repeat(40) })).toThrow('unresolved');
});
