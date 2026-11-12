import { expect, test } from 'bun:test';
import { runWalltime } from './run-metrics';

test('consolidation and import overhead are not scientific walltime', () => {
  const timing = { started_at: '2026-11-10T17:35:02Z', completed_at: '2026-11-10T17:35:05Z' };
  expect(runWalltime({ execution: { ...timing, operation: 'consolidate' } })).toBe('—');
  expect(runWalltime({ execution: { ...timing, operation: 'import' } })).toBe('—');
  expect(runWalltime({ execution: timing })).toBe('3.0s');
  expect(runWalltime({ execution: { ...timing, operation: 'consolidate' }, scientific_execution: timing })).toBe('3.0s');
});
