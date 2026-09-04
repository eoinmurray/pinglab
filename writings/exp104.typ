#import "templates/article-layout.typ": journal-article
#let meta = (
  tags: ("txt", "v35.4.0"),
  title: "Cloudflare R2 archive",
  created_at: "2026-08-11T00:00:00Z",
  updated_at: "2026-08-28T00:00:00Z",
  description: "Configure R2 access, inspect backups, and distinguish campaign payload recovery from restoring a complete validated Pingstore run.",
  collection: "pinglab-docs",
  order: 2,
)

#let body = [
  == What is backed up

  R2 is remote backup storage for expensive-to-reproduce payloads such as trained checkpoint banks. Git records source code; Pingstore records completed scientific runs. An R2 snapshot is not automatically a complete Pingstore run, and a successful download does not make recovered files operational evidence.

  The current helper is `experiments/helpers/archive.py`. Its interfaces differ:

  #table(
    columns: (auto, 1fr),
    [*Command*], [*Current scope*],
    [`list`], [List an experiment's remote snapshots and summary metadata.],
    [`archive-campaign`], [Validate the named `exp022` campaign and back up its complete cell payload bank with an inventory.],
    [`restore-campaign`], [Copy that bank into a separate absent or empty directory and compare the copy with remote files.],
    [`archive` / `restore`], [Older active-view/scratch interfaces. These do not provide complete v3 run backup and recovery; do not use them as that workflow.],
  )

  The generic `archive` resolves an export through the published artifact view rather than accepting an explicit complete run. It can miss the authoritative `run.json`, retained execution provenance, and notes. The generic `restore` copies files into a hidden scratch layout but does not reconstruct or validate a complete manifest. Those limits require implementation work, not a documentation promise.

  == Access and authentication

  Check the machine you are using; remote configuration is local to that account. Do not infer R2 access from an SSH alias or from access working on another host.

  ```sh
  rclone listremotes
  ```

  If `r2:` is missing, create an *Object Read & Write* credential scoped to the `pinglab` bucket in the #link("https://dash.cloudflare.com/")[Cloudflare dashboard], then run:

  ```sh
  rclone config
  ```

  Choose remote name `r2`, storage `s3`, and provider `Cloudflare`. Enter the Access Key ID, Secret Access Key, and endpoint supplied by Cloudflare for the bucket's jurisdiction. For an EU-jurisdiction bucket, use its EU endpoint. Follow the official #link("https://developers.cloudflare.com/r2/examples/rclone/")[rclone setup guide] and #link("https://developers.cloudflare.com/r2/api/tokens/")[token guide]. Use separate scoped credentials per machine where practical; read-only access is sufficient for inspection and download.

  Never commit credentials or print unredacted configuration. Verify the remote without uploading:

  ```sh
  rclone config redacted r2
  rclone lsf --dirs-only r2:pinglab/archive
  ```

  The helper defaults to remote `r2` and bucket `pinglab`; `PINGLAB_R2_REMOTE` and `PINGLAB_R2_BUCKET` override these. R2 credentials are separate from RunPod volume credentials.

  == Inspect snapshots

  ```sh
  uv run python experiments/helpers/archive.py list exp022
  rclone tree r2:pinglab/archive/exp022 --max-depth 2
  rclone size r2:pinglab/archive/exp022
  ```

  For an explicitly chosen snapshot identifier:

  ```sh
  rclone cat r2:pinglab/archive/exp022/<snapshot-id>/MANIFEST.json
  ```

  Replace the placeholder before running. The remote prefix is `archive/<experiment>/<snapshot-id>/`; older snapshots use a producing commit, while campaign snapshots add a manifest-digest suffix. Inspect the manifest's archive type, source, inventory, and identity before choosing a recovery procedure. A commit alone does not uniquely identify every run produced by that code. Inspection for historical migration or recovery requires separate authorization.

  == Archive a completed campaign

  For an explicitly selected `exp022` campaign with all cells complete and valid:

  ```sh
  uv run python experiments/helpers/archive.py archive-campaign \
    /path/to/campaign/campaign.json
  ```

  Replace the example path with the exact manifest. The helper checks the campaign, archives its `cells/` bank, records file sizes and SHA-256 hashes in `MANIFEST.json`, and verifies the uploaded payload against the local source. Its snapshot identity combines the producing commit and campaign manifest digest; an existing destination is refused.

  Expected result: a reported snapshot ID and successful copy verification. Keep the campaign manifest and execution records separately: this command copies the cell bank, not every file in the campaign directory or a complete v3 run. Uploading consumes storage and transfer resources; confirm the source and authorization first.

  == Restore campaign payloads

  Choose a specific snapshot and a separate destination:

  ```sh
  uv run python experiments/helpers/archive.py restore-campaign \
    exp022 <snapshot-id> --destination /path/to/empty-recovery-directory
  ```

  The destination must be absent or empty. The helper copies the payload and runs `rclone check --download`; this checks the downloaded files against the remote copy. It does not independently establish the original scientific provenance or reconstruct a completed run. The current restore path also does not verify every downloaded file against the recorded SHA-256 inventory; retain and check that manifest before treating recovery as authenticated evidence.

  Do not restore over a live bank, an existing completed run, or a published artifact directory. Keep the recovered payload separate until its identity, completeness, checkpoint roles, and intended use have been validated.

  == Completed-run recovery

  The #link("https://github.com/eoinmurray/pinglab/blob/main/tools/pingstore/README.md")[Storage Guide] is authoritative. A complete v3 backup must preserve `run.json`, the full `export/`, and any `README.md` or `provenance/`. Its payload checksum covers every payload file, not only model weights.

  Recovery must validate the schema, exact root layout, payload checksum, and required upstream input references before making a run visible or consuming it. A hidden `.pingstore/runs/.<run-id>.tmp/` directory is incomplete until validation and atomic completion. Do not fabricate missing provenance, choose the latest snapshot implicitly, or rename a payload-only download into a completed run.

  The helper currently has no general command that performs that complete workflow. Recovering historical evidence, importing payloads into new runs, or migrating old schemas requires separate explicit authorization and a source-specific plan. V2 evidence remains non-operational; it must not be silently relabelled as v3.

  == Safety and troubleshooting

  + *Remote missing or access denied:* check the local remote name, bucket scope, endpoint, and permissions. Do not paste secrets into logs or the repository.
  + *Snapshot already exists:* verify its identity rather than overwriting it. `rclone copy` avoids destination deletion, but can still overwrite matching objects; it is not an immutability guarantee by itself.
  + *Copy interrupted or verification failed:* retain the source and incomplete destination for inspection. Do not delete the original or declare recovery complete.
  + *Unexpected snapshot contents:* stop before restoration into operational storage. Distinguish a campaign cell bank, a legacy export snapshot, and a complete run.
  + *Deletion or cleanup:* keep it separate and explicitly authorized. Never run `rclone sync` against an archive prefix. The helper has no delete command.

  #link("/exp103/")[exp103] — #link("/exp103/")[_Compute options_]
]

#let body = journal-article("exp104", (), body)
