---
title: Publishing Rules
author: Eoin Murray
collection: notes
description: Local static builds and independent GitHub Pages deployment.
created_at: "2026-11-12T12:03:36Z"
updated_at: "2026-11-12T20:22:47Z"
---

# Publishing

1. Build locally with `cd site && bun run build`. Astro reads local run metadata directly, selects the newest presentation once per build, and bundles only referenced figures, videos and animation data. No Python resolver, R2 access or scientific array hashing is required. Raw arrays and checkpoints stay outside the site.
2. A report with no presentation displays unavailable evidence. An invalid selected run or missing requested file fails the build; do not substitute recipe defaults or silently select an older presentation. Parameters come from original native snapshots and recorded configuration.
3. `bun run publish` builds, then pushes prebuilt output through a separate temporary Git repository to `gh-pages`. Preserve branch history with a normal push and preserve the source worktree. A push to `main` does not publish.
4. GitHub Actions uploads and deploys the prebuilt artifact. It does not install Python/rclone, rebuild Astro, read R2 or register storage retention roots. Retry by dispatching the Pages workflow. Deployment is independent of backup and deletion.
5. `/pingstore-evidence.json` is a small snapshot of source revision, selected run IDs and dependency IDs. It documents the build; it neither prevents deletion nor proves successful deployment. Bundled pages remain usable after scientific copies are explicitly deleted, but they are not scientific backups.
6. Use `bun run preview --port 3001` to preview the static build. `bun run build:remote` optionally exercises explicit remote reads; it is not required for local deletion. The PDF library stays local, and production paper links use `papers/index.yaml`.
