#let meta = (
  title: "Cloudflare R2 archive",
  date: "2026-08-11",
  description: "How to access pinglab's Cloudflare R2 bucket and archive or restore provenance-keyed experiment scratch.",
  collection: "ephemeral",
  status: "final",
)

#let body = [
  Cloudflare R2 stores selected experiment scratch that is expensive to reproduce, especially trained checkpoint banks under `temp/experiments/`. Git remains the record for source code and published artifacts. R2 is a manual backup, not a live training filesystem.

  The supported interface is `experiments/helpers/archive.py`. It stores snapshots at:

  ```text
  r2:pinglab/archive/<experiment-slug>/<producing-git-sha>/
  ```

  == Access and authentication

  The Mac currently has the `r2:` rclone remote. Hetzner does not. Credentials consist of an Access Key ID, Secret Access Key, and the EU R2 endpoint. Never commit or paste their values.

  To configure a new machine, create an *Object Read & Write* token scoped to the `pinglab` bucket in the #link("https://dash.cloudflare.com/")[Cloudflare dashboard]. Then configure rclone interactively:

  ```sh
  rclone config
  ```

  Use these choices:

  1. Remote name: `r2`.
  2. Storage: `s3`.
  3. Provider: `Cloudflare`.
  4. Enter the Access Key ID and Secret Access Key manually.
  5. Endpoint: the EU endpoint shown by Cloudflare.
  6. ACL: `private`.

  Prefer a separate bucket-scoped token for each machine. See Cloudflare's #link("https://developers.cloudflare.com/r2/api/tokens/")[token guide] and #link("https://developers.cloudflare.com/r2/examples/rclone/")[rclone guide].

  Verify access without writing:

  ```sh
  rclone listremotes
  rclone config redacted r2
  rclone lsd r2:
  rclone lsf --dirs-only r2:pinglab/archive
  ```

  Do not print the unredacted rclone configuration.

  == Archive a run

  Check the local scratch, then archive it:

  ```sh
  du -sh temp/experiments/exp022
  uv run python experiments/helpers/archive.py archive exp022
  ```

  The helper determines the producing Git commit, copies the tree, uploads `MANIFEST.json`, and verifies it with `rclone check`. It uses `copy`, never `sync`, so a partial local tree cannot delete existing remote objects.

  == List snapshots

  ```sh
  uv run python experiments/helpers/archive.py list exp022

  rclone tree r2:pinglab/archive/exp022 --max-depth 2
  rclone size r2:pinglab/archive/exp022
  rclone cat r2:pinglab/archive/exp022/<sha>/MANIFEST.json
  ```

  == Restore a snapshot

  Restore the latest snapshot:

  ```sh
  uv run python experiments/helpers/archive.py restore exp022
  ```

  Or restore a specific producing commit:

  ```sh
  uv run python experiments/helpers/archive.py restore exp022 <sha>
  ```

  Restore copies into `temp/experiments/exp022/` without deleting unrelated local files. Validate configuration files and load representative checkpoints before using the restored bank.

  == Safety rules

  - Never run `rclone sync` against an archive prefix.
  - Archive only completed runs worth preserving.
  - Do not use R2 as mutable training storage.
  - Keep snapshots from different producing commits separate.
  - Revoke and replace a token immediately if it may have leaked.
  - Do not confuse R2 credentials with `RUNPOD_S3_ACCESS_KEY_ID` and `RUNPOD_S3_SECRET_ACCESS_KEY`; those access RunPod storage.

  The helper deliberately has no delete command. Remote deletion should remain a separate, explicit administrative action.
]
