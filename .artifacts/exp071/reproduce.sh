#!/usr/bin/env sh
# Explicit reproducer entry point for exp071.
# The canonical Demolab provenance script for this artifact bundle is run.sh.
cd "$(git rev-parse --show-toplevel)"
exec sh artifacts/data/exp071/run.sh
