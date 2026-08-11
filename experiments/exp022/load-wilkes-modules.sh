#!/bin/bash

# Slurm submissions use a deliberately sanitized environment, so the `module`
# shell function inherited by login shells is absent. Initialise Environment
# Modules explicitly before selecting the reviewed Wilkes3 software stack.
modules_init="${EXP022_MODULES_INIT:-/etc/profile.d/modules.sh}"
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
