export interface RunMetrics { walltime: string; size: string }

export function runWalltime(record: any): string {
  const timing = record?.scientific_execution ?? record?.completed_hpc_import?.source_execution
    ?? (record?.execution?.operation?.includes('import') || record?.execution?.operation === 'consolidate'
      ? undefined : record?.execution);
  const seconds = (Date.parse(timing?.completed_at) - Date.parse(timing?.started_at)) / 1000;
  if (!Number.isFinite(seconds) || seconds < 0) return '—';
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor(seconds % 3600 / 60)}m`;
}

export function dataSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '—';
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  const unit = bytes === 0 ? 0 : Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** unit).toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}
