/** Resolve exact run identities locally in development, otherwise from R2. */
import { createHash } from 'node:crypto';
import { resolveRunIndex, validateRunIndex, type RunIndex } from './run-index.ts';
import { requestValue } from './run-context.ts';
import { retryRead } from './remote-http.ts';
import { readdir, readFile, stat } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { assetBase, remoteCatalog, remoteControl, validateCatalog, runPattern, safePath, type Catalog } from './remote-catalog.ts';

export interface StoreOptions {
  runsRoot?: string;
  baseUrl?: string;
  source?: 'auto' | 'local' | 'remote';
  catalog?: () => Promise<Catalog>;
  control?: () => Promise<any>;
}
export const absent = (error: unknown) => (error as NodeJS.ErrnoException).code === 'ENOENT';
function missing(path: string): Error {
  return Object.assign(new Error(`Run file unavailable: ${path}`), { code: 'ENOENT' });
}
let active = 0;
const waiting: (() => void)[] = [];
async function limited<T>(operation: () => Promise<T>): Promise<T> {
  if (active >= 6) await new Promise<void>(done => waiting.push(done));
  else active++;
  try { return await operation(); }
  finally {
    const next = waiting.shift();
    if (next) next();
    else active--;
  }
}

export class RunStore {
  readonly root: string;
  readonly base: string;
  readonly source: 'auto' | 'local' | 'remote';
  readonly catalog: () => Promise<Catalog>;
  readonly control: () => Promise<any>;
  private warned = false;
  private buildIndex?: Promise<RunIndex>;
  private indexKey = Symbol('run-index');
  private catalogKey = Symbol('run-catalog');
  private controlKey = Symbol('run-control');
  private reads = new Map<string, { expires: number; value: Promise<Buffer> }>();
  constructor(options: StoreOptions = {}) {
    this.root = resolve(options.runsRoot ?? process.env.PINGLAB_RUNS_ROOT ?? '../.pingstore/runs');
    this.base = (options.baseUrl ?? assetBase()).replace(/\/$/, '');
    const source = options.source ?? (options.runsRoot ? 'local' : process.env.PINGLAB_RUN_SOURCE
      ?? (process.env.NODE_ENV === 'production' ? 'remote' : 'auto'));
    if (!['auto', 'local', 'remote'].includes(source)) throw new Error(`Invalid run source: ${source}`);
    this.source = source as StoreOptions['source'] & string;
    const loadCatalog = options.catalog ?? (async () => process.env.PINGLAB_CATALOG_PATH
      ? validateCatalog(JSON.parse(await readFile(process.env.PINGLAB_CATALOG_PATH, 'utf8')))
      : remoteCatalog(this.base, process.env.NODE_ENV !== 'production'));
    this.catalog = () => requestValue(this.catalogKey, loadCatalog);
    this.control = () => requestValue(this.controlKey, options.control ?? (() => remoteControl(this.base)));
  }
  path(run: string, file: string) {
    if (!runPattern.test(run) || !safePath(file)) throw new Error('Invalid run file');
    return join(this.root, run, file);
  }
  url(run: string, file: string) {
    this.path(run, file);
    return `${this.base}/pingstore/runs/${run}/${file.split('/').map(encodeURIComponent).join('/')}`;
  }
  async local(run: string): Promise<boolean> {
    this.path(run, 'run.json');
    if (this.source === 'remote') return false;
    try { return (await stat(join(this.root, run))).isDirectory(); }
    catch (error) { if (absent(error)) return false; throw error; }
  }
  async localIds(experiment?: string): Promise<string[]> {
    if (this.source === 'remote') return [];
    try {
      return (await readdir(this.root, { withFileTypes: true }))
        .filter(entry => entry.isDirectory() && runPattern.test(entry.name) && (!experiment || entry.name.startsWith(`${experiment}-`)))
        .map(entry => entry.name).sort();
    } catch (error) { if (absent(error)) return []; throw error; }
  }
  async ids(experiment?: string): Promise<string[]> {
    const local = await this.localIds(experiment);
    let remote: string[] = [];
    if (this.source !== 'local') {
      try { remote = Object.keys((await this.catalog()).runs); }
      catch (error) {
        if (this.source === 'remote') throw error;
        if (!this.warned) console.warn('[Pingstore] R2 inventory unavailable; showing local runs only.', error);
        this.warned = true;
      }
    }
    return [...new Set([...local, ...remote])].filter(id => !experiment || id.startsWith(`${experiment}-`)).sort();
  }
  async entry(run: string) {
    this.path(run, 'run.json');
    if (this.source === 'local') throw missing(run);
    const entry = (await this.catalog()).runs[run];
    if (!entry) throw missing(run);
    return entry;
  }
  async read(run: string, file: string): Promise<Buffer> {
    const path = this.path(run, file);
    // An existing local run is authoritative, including errors in its files.
    if (await this.local(run) || this.source === 'local') {
      const bytes = await readFile(path);
      if (this.source !== 'local') {
        let catalog: Catalog | undefined;
        try { catalog = await this.catalog(); } catch { /* Index reports the unresolved R2 inventory. */ }
        const entry = catalog?.runs[run];
        if (entry) {
          const fingerprint = entry.files[file];
          if (!fingerprint || fingerprint.bytes !== bytes.length
              || createHash('sha256').update(bytes).digest('hex') !== fingerprint.sha256) {
            throw new Error(`Local checksum mismatch against verified catalog: ${run}/${file}`);
          }
        }
      }
      return bytes;
    }
    const entry = await this.entry(run);
    const fingerprint = entry.files[file];
    if (!fingerprint) throw missing(`${run}/${file}`);
    const key = `${run}/${file}/${fingerprint.sha256}`;
    const cached = this.reads.get(key);
    if (cached && cached.expires > Date.now()) return cached.value;
    const value = limited(() => retryRead(`R2 ${run}/${file}`, async () => {
      const response = await fetch(this.url(run, file), { signal: AbortSignal.timeout(30_000), redirect: 'error' });
      if (response.status !== 200) throw new Error(`R2 ${run}/${file}: HTTP ${response.status}`);
      const bytes = Buffer.from(await response.arrayBuffer());
      if (bytes.length !== fingerprint.bytes || createHash('sha256').update(bytes).digest('hex') !== fingerprint.sha256) {
        throw new Error(`R2 checksum mismatch: ${run}/${file}`);
      }
      return bytes;
    }));
    this.reads.set(key, { expires: process.env.NODE_ENV === 'production' ? Infinity : Date.now() + 30_000, value });
    try { return await value; }
    catch (error) { this.reads.delete(key); throw error; }
  }
  async json(run: string, file = 'run.json'): Promise<any> {
    if (file === 'run.json' && !await this.local(run) && this.source !== 'local') return structuredClone((await this.entry(run)).record);
    const value = JSON.parse((await this.read(run, file)).toString('utf8'));
    if (file === 'run.json' && (value.schema !== 'pingstore.run/v4' || value.run_id !== run
        || value.experiment !== run.slice(0, 6) || !value.execution?.completed_at)) throw new Error(`Invalid completed run record: ${run}`);
    return value;
  }
  async files(run: string): Promise<Map<string, number>> {
    this.path(run, 'run.json');
    if (await this.local(run) || this.source === 'local') {
      const files = new Map<string, number>();
      const visit = async (directory: string, prefix: string) => {
        for (const entry of await readdir(directory, { withFileTypes: true })) {
          const name = prefix + entry.name;
          if (entry.isDirectory()) await visit(join(directory, entry.name), `${name}/`);
          else if (entry.isFile()) files.set(name, (await stat(join(directory, entry.name))).size);
        }
      };
      await visit(join(this.root, run, 'export'), '');
      return files;
    }
    return new Map(Object.entries((await this.entry(run)).files).filter(([path]) => path.startsWith('export/'))
      .map(([path, fingerprint]) => [path.slice(7), fingerprint.bytes]));
  }
  async has(run: string, file: string): Promise<boolean> {
    this.path(run, file);
    if (await this.local(run) || this.source === 'local') {
      try { return (await stat(join(this.root, run, file))).isFile(); }
      catch (error) { if (absent(error)) return false; throw error; }
    }
    try { return !!(await this.entry(run)).files[file]; }
    catch (error) { if (absent(error)) return false; throw error; }
  }
  async index(): Promise<RunIndex> {
    const load = async () => {
      if (process.env.PINGLAB_REPORT_INDEX_PATH) {
        const index = validateRunIndex(JSON.parse(await readFile(process.env.PINGLAB_REPORT_INDEX_PATH, 'utf8')));
        if (this.source !== 'remote' || index.source !== 'remote') throw new Error('Frozen publication indexes require remote-only rendering');
        return index;
      }
      let catalog: Catalog | undefined;
      if (this.source !== 'local') {
        try { catalog = await this.catalog(); }
        catch (error) {
          if (this.source === 'remote') throw error;
          if (!this.warned) console.warn('[Pingstore] R2 inventory unavailable; showing local runs only.', error);
          this.warned = true;
        }
      }
      const control = this.source === 'local' ? undefined : await this.control();
      return resolveRunIndex({ root: this.root, source: this.source, catalog, control });
    };
    if (process.env.NODE_ENV === 'production') return this.buildIndex ??= load();
    return requestValue(this.indexKey, load);
  }
  async latest(experiment: string): Promise<string | undefined> {
    const index = await this.index();
    if (index.selection_errors[experiment]) throw new Error(`Unresolved runset for ${experiment}: ${index.selection_errors[experiment]}`);
    return index.selections[experiment];
  }
}
export const runStore = new RunStore();
