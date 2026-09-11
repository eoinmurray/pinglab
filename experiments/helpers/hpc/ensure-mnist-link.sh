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
[[ -L "$link_path" && "$(readlink -f "$link_path")" == "$target" ]] || {
  echo "MNIST link exists with a different target: $link_path" >&2
  exit 2
}
