import { getEntry } from 'astro:content';
import { runStore } from '../../../site/src/lib/run-store';
import { reportData, reportSnapshots } from '../../../site/src/lib/report-data';

// Missing historical evidence is explicit for this report. Never replace its
// pin with a newer diagnostic run; existing but invalid payloads still fail.
export async function evidence() {
  const article = await getEntry('reports', 'exp124');
  const pin = article?.data.runs?.exp124;
  if (!pin || !/^exp124-r\d+-present$/.test(pin)) throw new Error('exp124 requires an explicit presentation pin');
  try {
    if (!await runStore.has(pin, 'export/numbers.json')) return undefined;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined;
    throw error;
  }
  return reportData('exp124');
}


export async function snapshotEvidence() {
  return reportSnapshots('exp124', {
    snapshots: {
      coba: 'coba/train/end/snapshot.json',
      ping: 'ping/train/end/snapshot.json',
    },
    allowUnavailable: true,
  });
}
