import { reportData } from '../../../site/src/lib/report-data';
import { optionalPresentation } from '../../../site/src/lib/pingstore';
import { readSnapshotEvidence } from '../../../site/src/lib/snapshot-data';

// Numbers, figures and snapshots share the session's latest presentation selection.
export async function evidence() {
  return reportData('exp124', { allowUnavailable: true });
}

export async function snapshotEvidence() {
  const selected = await optionalPresentation('exp124');
  if (!selected) return undefined;
  return readSnapshotEvidence(selected.id, {
    experiment: 'exp124',
    snapshots: {
      coba: 'coba/train/end/snapshot.json',
      ping: 'ping/train/end/snapshot.json',
    },
    allowUnavailable: true,
  });
}
