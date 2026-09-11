#!/bin/bash

# Slurm exports are deliberately sanitized, so initialise Environment Modules
# explicitly before selecting the reviewed Wilkes3 GPU software stack.
modules_init="${PINGLAB_MODULES_INIT:-/etc/profile.d/modules.sh}"
if [[ ! -r "$modules_init" ]]; then
  echo "Environment Modules initializer is not readable: $modules_init" >&2
  return 2
fi
source "$modules_init"
module purge
module load rhel8/default-amp
