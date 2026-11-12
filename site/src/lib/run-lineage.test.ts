import { expect, test } from 'bun:test';
import { lineageLists } from './run-lineage';
import type { RunIndex } from './run-index';

const digest = `sha256:${'a'.repeat(64)}`;
const pin = (run_id: string) => ({ run_id, payload_digest: digest });
const row = (run_id: string, inputs = {}, runset_inputs = inputs) => ({
  record: { run_id, payload_digest: digest, inputs, runset_inputs },
});

test('recursively includes original, promoted and cross-runset dependencies exactly once', () => {
  const index = {
    report_selections: { exp001: { exp001: 'p' }, 'notes/comparison': { exp001: 'p' } },
    runs: {
      p: row('p', { original: pin('old') }, { compute: pin('c') }),
      c: row('c', { bank: pin('bank') }), old: row('old', { bank: pin('bank') }),
      bank: row('bank'), other: row('other', { bank: pin('bank') }), orphan: row('orphan'),
    },
  } as unknown as RunIndex;
  const lists = lineageLists(index);
  expect(lists.inUse.map(item => item.id).sort()).toEqual(['bank', 'c', 'old', 'p']);
  expect(lists.other.map(item => item.id).sort()).toEqual(['orphan', 'other']);
  expect(lists.inUse.find(item => item.id === 'p')!.reports).toEqual(['exp001', 'notes/comparison']);
  expect(lists.inUse.find(item => item.id === 'bank')!.requiredBy.sort()).toEqual(['c', 'old']);
});

test('missing references, mismatched pins and cycles remain in use without infinite traversal', () => {
  const index = {
    report_selections: { report: { exp001: 'p' } },
    runs: {
      p: row('p', { missing: pin('absent'), changed: { ...pin('c'), payload_digest: 'wrong' } }),
      c: row('c', { cycle: pin('p') }), broken: { record: null },
    },
  } as unknown as RunIndex;
  const lists = lineageLists(index);
  expect(lists.inUse.map(item => item.id).sort()).toEqual(['absent', 'c', 'p']);
  expect(lists.inUse.find(item => item.id === 'absent')!.issues).toEqual(['Run record unavailable']);
  expect(lists.inUse.find(item => item.id === 'c')!.issues).toEqual(['Digest mismatch in input from p']);
  expect(lists.other.map(item => item.id)).toEqual(['broken']);
});

test('unpresented selections do not establish use', () => {
  const index = {
    report_selections: {}, selections: { exp001: 'p' },
    runs: { p: row('p', { source: pin('c') }), c: row('c') },
  } as unknown as RunIndex;
  const lists = lineageLists(index);
  expect(lists.inUse).toEqual([]);
  expect(lists.other.map(item => item.id)).toEqual(['p', 'c']);
});

test('groups roots by experiment and preserves shared cross-experiment branches', async () => {
  const { lineageGroups } = await import('./run-lineage');
  const record = (id: string, stage: string, inputs = {}) => ({
    record: { run_id: id, experiment: id.split('-')[0], stage, payload_digest: digest, inputs },
  });
  const p = 'exp001-r003-present', a = 'exp001-r002-analyse', c = 'exp002-r001-compute';
  const old = 'exp001-r001-present', orphan = 'exp003-r001-compute';
  const index = {
    report_selections: { report: { exp001: p } },
    runs: {
      [p]: record(p, 'present', { analysis: pin(a), bank: pin(c) }),
      [a]: record(a, 'analyse', { compute: pin(c) }), [c]: record(c, 'compute'),
      [old]: record(old, 'present', { bank: pin(c) }), [orphan]: record(orphan, 'compute'),
    },
  } as unknown as RunIndex;
  const { groups } = lineageGroups(index);
  const first = groups.find(group => group.experiment === 'exp001')!;
  expect(first.inUse.map(node => node.entry.id)).toEqual([p]);
  expect(first.unused.map(node => node.entry.id)).toEqual([old]);
  expect(first.inUse[0].children[0].children[0].entry.id).toBe(c);
  expect(first.inUse[0].children[1].repeated).toBe(true);
  expect(groups.find(group => group.experiment === 'exp002')!.inUse[0].entry.id).toBe(c);
  expect(groups.find(group => group.experiment === 'exp003')!.unused[0].entry.id).toBe(orphan);
});

test('consolidation sources are provenance, while real dependency links remain', async () => {
  const { lineageGroups } = await import('./run-lineage');
  const p = 'exp172-r203-present', c = 'exp172-r201-compute';
  const old = 'exp172-r132-compute', bank = 'exp022-r015-compute';
  const index = {
    report_selections: { exp172: { exp172: p } },
    runs: {
      [p]: { record: { ...row(p, { compute: pin(c) }).record, stage: 'present' } },
      [c]: { record: { ...row(c, { original_00: pin(old), bank: pin(bank) }, {}).record,
        execution: { operation: 'consolidate' } } },
      [old]: row(old), [bank]: row(bank),
    },
  } as unknown as RunIndex;
  const lists = lineageLists(index);
  expect(lists.inUse.map(item => item.id).sort()).toEqual([bank, c, p].sort());
  expect(lists.other.map(item => item.id)).toEqual([old]);
  const group = lineageGroups(index).groups.find(group => group.experiment === 'exp172')!;
  expect(group.inUse[0].children[0].children.map(node => node.entry.id)).toEqual([bank]);
  expect(group.unused[0].entry.id).toBe(old);
  expect(index.runs[c].record.inputs.original_00.run_id).toBe(old);
});

test('clean requires exactly one connected triplet; external branches do not affect it', async () => {
  const { lineageGroups } = await import('./run-lineage');
  const c = 'exp001-r001-compute', a = 'exp001-r002-analyse', p = 'exp001-r003-present';
  const bank = 'exp002-r001-compute';
  const make = (id: string, stage: string, inputs = {}) => ({
    record: { ...row(id, inputs).record, experiment: id.split('-')[0], stage },
  });
  const index = {
    report_selections: { exp001: { exp001: p } }, runs: {
      [c]: make(c, 'compute', { bank: pin(bank) }), [a]: make(a, 'analyse', { compute: pin(c) }),
      [p]: make(p, 'present', { analysis: pin(a), compute: pin(c) }),
      [bank]: make(bank, 'compute', { unavailable_external: pin('exp002-r000-compute') }),
    },
  } as unknown as RunIndex;
  const clean = () => lineageGroups(index).groups.find(group => group.experiment === 'exp001')!.clean;
  expect(clean()).toBe(true);
  index.runs['exp001-r004-compute'] = make('exp001-r004-compute', 'compute') as any;
  expect(clean()).toBe(false);
  delete index.runs['exp001-r004-compute'];
  index.runs[a].record.inputs = { backwards: pin(p) };
  index.runs[a].record.runset_inputs = {};
  expect(clean()).toBe(false);
  index.runs[a] = make(a, 'analyse') as any;
  expect(clean()).toBe(true); // Presentation also pins compute directly.
  index.runs[p].record.inputs = {};
  index.runs[p].record.runset_inputs = {};
  expect(clean()).toBe(false);
  delete index.runs[c];
  expect(clean()).toBe(false);
});

test('explicitly detached presentation history is excluded without hiding external inputs or stored history', async () => {
  const { lineageGroups } = await import('./run-lineage');
  const c = 'exp141-r012-compute', a = 'exp141-r013-analyse', p = 'exp141-r015-present';
  const old = 'exp141-r014-present', external = 'exp142-r027-present';
  const make = (id: string, stage: string, inputs = {}, links = inputs) => ({
    record: { ...row(id, inputs, links).record, experiment: id.split('-')[0], stage },
  });
  const index = {
    report_selections: { exp141: { exp141: p } }, runs: {
      [c]: make(c, 'compute'), [a]: make(a, 'analyse', { compute: pin(c) }),
      [p]: make(p, 'present', { analysis: pin(a), figure_source: pin(old), external: pin(external) }, { analysis: pin(a) }),
      [external]: make(external, 'present'),
    },
  } as unknown as RunIndex;
  expect(lineageLists(index).inUse.map(item => item.id).sort()).toEqual([c, a, p, external].sort());
  expect(lineageGroups(index).groups.find(group => group.experiment === 'exp141')!.clean).toBe(true);
  expect(index.runs[p].record.inputs.figure_source.run_id).toBe(old);
  index.runs[old] = make(old, 'present') as any;
  expect(lineageLists(index).other.map(item => item.id)).toEqual([old]);
  expect(lineageGroups(index).groups.find(group => group.experiment === 'exp141')!.clean).toBe(false);
  index.runs[p].record.runset_inputs.figure_source = pin(old);
  expect(lineageLists(index).inUse.some(item => item.id === old)).toBe(true);
});

test('promoted compute pins detach historical reanalysis while keeping external dependencies', async () => {
  const { lineageGroups } = await import('./run-lineage');
  const c = 'exp025-r008-compute', a = 'exp025-r013-analyse', p = 'exp025-r014-present';
  const old = 'exp025-r009-analyse', bank = 'exp022-r015-compute';
  const make = (id: string, stage: string, inputs = {}, links = inputs) => ({
    record: { ...row(id, inputs, links).record, experiment: id.split('-')[0], stage },
  });
  const index = {
    report_selections: { exp025: { exp025: p }, 'notes/manuscript': { exp025: p } }, runs: {
      [c]: make(c, 'compute', { bank: pin(bank) }),
      [a]: make(a, 'analyse', { frontier_analysis: pin(old), baseline_bank: pin(bank) }, { compute: pin(c) }),
      [p]: make(p, 'present', { analysis: pin(a) }), [bank]: make(bank, 'compute'),
    },
  } as unknown as RunIndex;
  const lists = lineageLists(index);
  expect(lists.inUse.map(item => item.id).sort()).toEqual([c, a, p, bank].sort());
  expect(lists.inUse.find(item => item.id === p)!.reports).toEqual(['exp025', 'notes/manuscript']);
  expect(lineageGroups(index).groups.find(group => group.experiment === 'exp025')!.clean).toBe(true);
  expect(index.runs[a].record.inputs.frontier_analysis.run_id).toBe(old);
  index.runs[a].record.runset_inputs = {};
  expect(lineageLists(index).inUse.some(item => item.id === old)).toBe(true);
});

test('declared analysis-only workflow stays clean and archived inputs resolve to their physical owner', async () => {
  const { lineageGroups } = await import('./run-lineage');
  const a='exp024-r001-analyse', p='exp024-r002-present', old='exp022-r001-compute', owner='exp022-r020-compute';
  const make=(id: string, stage: string, inputs={})=>({record:{...row(id,inputs).record,experiment:id.split('-')[0],stage}});
  const index={report_selections:{exp024:{exp024:p}},source_views:{[old]:{...pin(owner),source_digest:digest}},runs:{
    [a]:make(a,'analyse',{compute:pin(old)}),[p]:make(p,'present',{analysis:pin(a)}),[owner]:make(owner,'compute'),
  }} as unknown as RunIndex;
  const group=lineageGroups(index,{exp024:['analyse','present']}).groups.find(g=>g.experiment==='exp024')!;
  expect(group.clean).toBe(true);
  expect(group.inUse[0].children[0].children[0].entry.id).toBe(owner);
  expect(lineageLists(index).inUse.some(item=>item.id===old)).toBe(false);
  expect(lineageGroups(index).groups.find(g=>g.experiment==='exp024')!.clean).toBe(false);
});
