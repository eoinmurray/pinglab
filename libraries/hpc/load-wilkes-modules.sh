#!/bin/bash

# HPC submissions are deliberately sanitized, so initialise Environment Modules
# explicitly before selecting the reviewed Wilkes3 GPU software stack.
modules_init="${PINGLAB_MODULES_INIT:-/etc/profile.d/modules.sh}"
if [[ ! -r "$modules_init" ]]; then
  echo "Environment Modules initializer is not readable: $modules_init" >&2
  return 2
fi
# shellcheck source=/dev/null
source "$modules_init"
if ! type module >/dev/null 2>&1; then
  echo "Environment Modules did not define the module command" >&2
  return 2
fi
module purge
module load rhel8/default-amp
