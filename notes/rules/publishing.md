---
title: Publishing Rules
author: Eoin Murray
collection: notes
description: Local site builds, gh-pages deployment and publication evidence retention.
created_at: "2026-11-12T12:03:36Z"
updated_at: "2026-11-12T12:03:36Z"
---

# Publishing Rules

## Local build and publication

1. Build the production site locally from core's frozen local index. Bundle only
   run assets referenced by rendered pages, notes, slide decks and animation data.
   Serve these assets from Pages; keep raw scientific arrays and checkpoints
   outside the site. No R2 catalog or scientific payload is required for this build.
2. Selected local evidence must be valid and available; restore evicted inputs
   before publishing. A data report with no completed runset shows “Evidence
   unavailable” and skips its results, parameter tables and appendices. Record
   missing inputs in the evidence manifest. Invalid selected evidence fails the
   build; never substitute recipe defaults or fabricate results.
3. Use the supported commands below from the repository root. `build` does not
   push; `publish` builds before pushing. Use `bun run build:remote` separately
   for the remote-only validation required before eviction.

```sh
cd site
bun install --frozen-lockfile
bun run build
bun run preview --port 3001
```

To build and publish:

```sh
cd site
bun run publish
```

## Deployment

1. Push the generated static output and deployment workflow to `gh-pages` using
   the publisher's separate temporary Git repository. Preserve branch history
   with a normal fast-forward push; reject competing pushes rather than
   overwriting them. Preserve the source checkout's branch and uncommitted work.
2. `.github/workflows/pages.yml` deploys pushes to `gh-pages`. Source pushes to
   `main` do not deploy. For a manual retry, dispatch the workflow on `gh-pages`.
   The workflow is included on the publication branch; publication tools are
   checked out from `main`.
3. GitHub Actions uploads the prebuilt site without installing Astro, rebuilding
   the site or fetching scientific payloads from R2. The private
   `PINGLAB_RCLONE_CONFIG` Actions secret remains necessary for small publication
   coordination records. A bundled local site may publish before its scientific
   runs are uploaded to R2; existing cataloged records must still match.
4. Report HTML, bound values and inventory tables are build-time snapshots. New
   results require another local publish. `pingstore backup` neither builds nor
   publishes the site. The Git-ignored PDF library stays local; do not upload it
   to the public assets bucket. Production paper links come from the tracked
   `papers/index.yaml` citation index.

<a id="publication-retention"></a>

## Evidence and retention

1. Ship `/pingstore-evidence.json` with every build. Record the frozen source
   revision, dirty-source flag, index generation, selections, exact payload/input
   digests and unavailable reports. Actions stamps the Pages workflow/revision
   identity separately from the source revision. A manifest alone does not prove
   successful deployment.
2. Preserve the previous active deployment and incoming manifests as retention
   roots under the shared maintenance fence. Register evidence before every
   deployment attempt, including retries. Do not bypass the fence or release
   previous roots after a failed or uncertain deployment.
3. Activate evidence only after the Pages environment job succeeds and GitHub's
   successful active deployment and the served manifest bytes match. A verified
   replacement releases completed incoming attempts and preserves pending workflows.
4. Pruning requires the complete deployed evidence graph across local and R2
   holdings. Missing or changed scientific evidence blocks cleanup. A bundled
   site is not a scientific backup; eviction still requires verified R2 copies.
   Follow [Pingstore Rules](/notes/rules/pingstore) and the relevant
   [maintenance procedure](/notes/rules/pingstore-operations) for storage operations.
5. Legacy publication bootstrap uses the authenticated artifact of the active
   successful Pages run and validates its declared run ancestry against the store.
   The self-contained `recovered/exp124/published.html` is a text archive: its
   historical Dataset labels are not current Pingstore roots. Its bundled assets
   remain part of the Pages artifact.
