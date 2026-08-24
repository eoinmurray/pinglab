1. Demolab is Pinglab’s publishing engine, not its source of governance. Pinglab policy wins; Demolab documentation and runbooks are reference material unless explicitly invoked.

2. Use the Demolab-configured writing, artifact, and build interfaces. Do not edit machine-managed paths (`.demolab/` or `temp/bundle/`).

3. Default to local execution. Creating RunPod pods, Modal dispatches, or using other paid compute requires explicit permission naming the target.

4. Prefer Pinglab's managed dataset commands over manually copying campaign data or directly selecting storage paths. Use local ad-hoc runs for routine iteration, explicit upstream dataset identities for dependent experiments, immutable archived campaigns for gold-star collections, local cached baselines for subsequent iteration, and verified campaign identities for publication. Direct filesystem, `rclone`, and SSH manipulation is reserved for implementation, diagnosis, recovery, or functionality not yet exposed by the managed interface. This preference does not authorize paid compute, archival, deletion, publication, or promotion without the authority otherwise required for those actions.
