#!/bin/bash

set -euo pipefail  # Shared by reviewed HPC workers.
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 PERSISTENT_MNIST_ROOT [LINK_PATH]" >&2
  exit 2
fi
target="$(readlink -f "$1")"
link_path="${2:-/tmp/mnist}"
[[ -d "$target/MNIST" ]] || {
  echo "MNIST cache missing under $target" >&2
  exit 2
}
if ln -s "$target" "$link_path" 2>/dev/null; then
  exit 0
fi
if [[ -L "$link_path" ]]; then
  # Compute-node /tmp survives between jobs. Replace only a stale symlink;
  # never remove a real file or directory occupying the requested path.
  ln -sfn "$target" "$link_path"
  [[ "$(readlink -f "$link_path")" == "$target" ]] || {
    echo "failed to replace stale MNIST link: $link_path" >&2
    exit 2
  }
  exit 0
fi
echo "MNIST link path exists and is not a symlink: $link_path" >&2
exit 2
