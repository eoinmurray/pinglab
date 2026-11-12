/** Read run metadata and requested files directly. No Python, hashing or retention. */
import { readdir, readFile, lstat } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { buildRunIndex, validateRecord, runPattern, type RunIndex, type IndexRun } from './run-index.ts';
import { requestValue } from './run-context.ts';
import { assetBase, remoteCatalog, safePath, type Catalog } from './remote-catalog.ts';

export interface StoreOptions {
  runsRoot?: string;
  baseUrl?: string;
  source?: 'auto' | 'local' | 'remote';
  catalog?: () => Promise<Catalog>;
}
export const absent = (error: unknown) => (error as NodeJS.ErrnoException).code === 'ENOENT';
const missing = (path: string) => Object.assign(new Error(`Run file unavailable: ${path}`), { code: 'ENOENT' });
export class RunStore {
  readonly root: string;
  readonly base: string;
  readonly source: 'auto' | 'local' | 'remote';
  private options: StoreOptions;
  private buildIndex?: Promise<RunIndex>;
  private buildCatalog?: Promise<Catalog>;
  private indexKey = Symbol('run-index');
  private catalogKey = Symbol('backup-inventory');
  constructor(options: StoreOptions = {}) {
    this.options = options;
    this.root = resolve(options.runsRoot ?? process.env.PINGLAB_RUNS_ROOT ?? '../.pingstore/runs');
    this.base = (options.baseUrl ?? assetBase()).replace(/\/$/, '');
    this.source = options.source ?? process.env.PINGLAB_RUN_SOURCE as StoreOptions['source'] ?? 'local';
    if (!['local', 'remote', 'auto'].includes(this.source)) throw new Error('Invalid Pingstore source');
  }
  async catalog(): Promise<Catalog> {
    const load = this.options.catalog ?? (() => remoteCatalog(this.base));
    return process.env.NODE_ENV === 'production' ? this.buildCatalog ??= load() : requestValue(this.catalogKey, load);
  }
  path(run: string, file: string) {
    if (!runPattern.test(run) || !safePath(file)) throw new Error('Invalid run file');
    return join(this.root, run, file);
  }
  private async localPath(run: string, file: string) {
    const target = this.path(run, file);
    // Check every path component; assets cannot escape the run through symlinks.
    for (const path of [this.root, join(this.root, run), ...file.split('/').map((_, i, parts) => join(this.root, run, ...parts.slice(0, i + 1)))]) {
      if ((await lstat(path)).isSymbolicLink()) throw new Error(`Run paths cannot use symlinks: ${path}`);
    }
    return target;
  }
  url(run: string, file: string) {
    this.path(run, file);
    return `${this.base}/pingstore-v2/runs/${encodeURIComponent(run)}/${file.split('/').map(encodeURIComponent).join('/')}`;
  }
  async local(run: string): Promise<boolean> {
    this.path(run, 'run.json');
    if (this.source === 'remote') return false;
    try {
      const entry = await lstat(join(this.root, run));
      if (entry.isSymbolicLink()) throw new Error(`Run directory cannot be a symlink: ${run}`);
      return entry.isDirectory();
    } catch (error) { if (absent(error)) return false; throw error; }
  }
  async entry(run: string) {
    this.path(run, 'run.json');
    const entry = (await this.catalog()).runs[run];
    if (!entry) throw missing(run);
    return entry;
  }
  async read(run: string, file: string): Promise<Buffer> {
    if (await this.local(run) || this.source === 'local') return readFile(await this.localPath(run, file));
    if (!(file in (await this.entry(run)).files)) throw missing(`${run}/${file}`);
    const response = await fetch(this.url(run, file), { signal: AbortSignal.timeout(15_000) });
    if (response.status === 404) throw missing(`${run}/${file}`);
    if (!response.ok) throw new Error(`Run file: HTTP ${response.status}`);
    return Buffer.from(await response.arrayBuffer());
  }
  async json(run: string, file = 'run.json'): Promise<any> {
    if (file === 'run.json' && !await this.local(run) && this.source !== 'local') return validateRecord((await this.entry(run)).record, run);
    const value = JSON.parse((await this.read(run, file)).toString('utf8'));
    return file === 'run.json' ? validateRecord(value, run) : value;
  }
  async files(run: string): Promise<Map<string, number>> {
    if (await this.local(run) || this.source === 'local') {
      const files = new Map<string, number>();
      const visit = async (directory: string, prefix: string) => {
        for (const entry of await readdir(directory, { withFileTypes: true })) {
          if (entry.isSymbolicLink()) throw new Error('Run files cannot use symlinks');
          const name = prefix + entry.name;
          if (entry.isDirectory()) await visit(join(directory, entry.name), `${name}/`);
          else if (entry.isFile()) files.set(name, (await lstat(join(directory, entry.name))).size);
        }
      };
      try { await visit(await this.localPath(run, 'export'), ''); }
      catch (error) { if (!absent(error)) throw error; }
      return files;
    }
    return new Map(Object.entries((await this.entry(run)).files).filter(([path]) => path.startsWith('export/'))
      .map(([path, bytes]) => [path.slice(7), bytes]));
  }
  async has(run: string, file: string): Promise<boolean> {
    if (await this.local(run) || this.source === 'local') {
      try { return (await lstat(await this.localPath(run, file))).isFile(); }
      catch (error) { if (absent(error)) return false; throw error; }
    }
    return file in (await this.entry(run)).files;
  }
  async index(): Promise<RunIndex> {
    const load = async () => {
      if (process.env.PINGLAB_REPORT_INDEX_PATH) {
        const value: RunIndex = JSON.parse(await readFile(process.env.PINGLAB_REPORT_INDEX_PATH, 'utf8'));
        if (value.source !== this.source) throw new Error('Frozen index source differs');
        return buildRunIndex(value.runs, value.source);
      }
      const rows: Record<string, IndexRun> = {};
      if (this.source !== 'local') {
        for (const [id, entry] of Object.entries((await this.catalog()).runs)) {
          rows[id] = { record: entry.record, storage: 'R2', bytes: entry.bytes, file_count: Object.keys(entry.files).length };
        }
      }
      if (this.source !== 'remote') {
        let entries;
        try { entries = await readdir(this.root, { withFileTypes: true }); }
        catch (error) { if (!absent(error)) throw error; entries = []; }
        for (const entry of entries) {
          if (entry.name.startsWith('.') || (!entry.isDirectory() && !entry.isSymbolicLink())) continue;
          const record = await this.json(entry.name);
          let sizes: { bytes?: number; file_count?: number } = {};
          try { sizes = JSON.parse(await readFile(join(this.root, '../cache', `${entry.name}.json`), 'utf8')); }
          catch (error) { if (!absent(error)) throw error; }
          rows[entry.name] = { record, storage: rows[entry.name] ? 'Both' : 'Local', ...sizes };
        }
      }
      return buildRunIndex(rows, this.source);
    };
    return process.env.NODE_ENV === 'production' ? this.buildIndex ??= load() : requestValue(this.indexKey, load);
  }
  async ids(experiment?: string): Promise<string[]> {
    return Object.values((await this.index()).runs).filter(row => !experiment || row.record.experiment === experiment).map(row => row.record.id);
  }
  async latest(experiment: string) { return (await this.index()).selections[experiment]; }
}
export const runStore = new RunStore();
