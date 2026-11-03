// Downloads and header checks share a budget so figure validation cannot flood R2.
let active = 0;
const waiting: (() => void)[] = [];
export async function limitedRemoteRead<T>(operation: () => Promise<T>): Promise<T> {
  if (active >= 6) await new Promise<void>(done => waiting.push(done));
  else active++;
  try { return await operation(); }
  finally {
    const next = waiting.shift();
    if (next) next();
    else active--;
  }
}

export const remoteReadTimeoutMs = 60_000;

/** Retry interrupted public-object reads; never accept unchecked replacement bytes. */
export async function retryRead<T>(label: string, operation: () => Promise<T>): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    try { return await operation(); }
    catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') throw error;
      if (attempt === 3) throw new Error(`${label}: ${error instanceof Error ? error.message : String(error)}`, { cause: error });
      console.warn(`[Pingstore] ${label}: retry ${attempt + 1}/3 after ${error instanceof Error ? error.message : String(error)}`);
      await new Promise(done => setTimeout(done, 500 * 2 ** attempt));
    }
  }
}
