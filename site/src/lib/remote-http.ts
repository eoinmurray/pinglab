/** Retry interrupted public-object reads; never accept unchecked replacement bytes. */
export async function retryRead<T>(label: string, operation: () => Promise<T>): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    try { return await operation(); }
    catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') throw error;
      if (attempt === 2) throw new Error(`${label}: ${error instanceof Error ? error.message : String(error)}`, { cause: error });
      await new Promise(done => setTimeout(done, 200 * (attempt + 1)));
    }
  }
}
