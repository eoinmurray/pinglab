# Plan

## R2 artifacts storage (status: set up, pending follow‑ups)
- Installed official rclone binary at `~/.local/bin/rclone` (Homebrew rclone cannot mount on macOS).
- macFUSE approved and loaded; rclone mount now works.
- R2 bucket is mounted at `/Users/eoin/pinglab-r2`.
- `posts/_artifacts` is now a symlink to the mount.
- Local backup preserved at `posts/_artifacts.local`.

### Current mount command
```bash
set -a; source .env; set +a
~/.local/bin/rclone mount r2:$R2_BUCKET ~/pinglab-r2 \
  --vfs-cache-mode writes \
  --vfs-cache-max-age 24h \
  --vfs-cache-max-size 10G \
  --dir-cache-time 1h \
  --log-level INFO \
  --log-file /tmp/rclone-mount.log \
  --daemon
```

### Stop mount
```bash
pkill -f "rclone mount r2:pinglab"
```

### Optional full resync from local backup
```bash
rsync -a --delete --exclude .DS_Store posts/_artifacts.local/ /Users/eoin/pinglab-r2/
```

### Follow‑ups
- Rotate R2 keys (previous config commands printed secrets in terminal history).
- Decide on a build-time artifacts manifest for `VeslxGallery` (globs won’t work against CDN listings).
- Optionally add a launch agent to auto-mount on reboot.
