#!/bin/sh
# demolab installer (macOS / Linux).
#
#   curl -LsSf https://demolab.eoinmurray.info/install.sh | sh
#
# Installs the toolchain (uv, typst, go-task), makes you a fresh, owned copy of demolab,
# scaffolds the folder structure, and prints how to continue. It does NOT install or launch a
# coding agent -- you bring your own and paste the prompt it prints.
#
# Pure ASCII on purpose: piped to /bin/sh under `set -u`, a non-ASCII char next to a $var
# gets swallowed into the variable name. Keep it that way.
set -eu

# DEMOLAB_REPO lets a fork/mirror (or the test suite) install from elsewhere; defaults to upstream.
REPO="${DEMOLAB_REPO:-https://github.com/eoinmurray/demolab}"
DIR="${1:-demolab}"
PROMPT="Read AGENTS.md and follow the Getting started runbook to set up my lab."

say()  { printf '\n\033[1m%s\033[0m\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }
# Visible status lines so the user sees each dependency being checked (green ok / yellow
# installing / red missing). Kept ASCII -- no glyphs, just [ok]/[..]/[!!].
ok()      { printf '  \033[32m[ok]\033[0m   %-8s %s\n' "$1" "$(command -v "${2:-$1}" 2>/dev/null)"; }
adding()  { printf '  \033[33m[..]\033[0m   %-8s installing...\n' "$1"; }
missing() { printf '  \033[31m[!!]\033[0m   %-8s %s\n' "$1" "$2"; }

[ -e "$DIR" ] && { echo "A '${DIR}' directory already exists here. Pass another name: ... | sh -s -- my-lab"; exit 1; }

# --- dependencies: report every one; auto-install what we can (typst is the fiddly off-brew one) ---
say "Checking dependencies"

have git || { missing git "required -- install git, then re-run"; exit 1; }
ok git

if have uv; then ok uv
else adding uv; curl -LsSf https://astral.sh/uv/install.sh | sh; fi

if have task; then ok go-task task
else adding go-task; sh -c "$(curl -sL https://taskfile.dev/install.sh)" -- -d -b "$HOME/.local/bin"; fi

if have typst; then ok typst
else
  adding typst
  if   have brew;  then brew install typst
  elif have cargo; then cargo install --locked typst-cli
  else missing typst "install manually: https://github.com/typst/typst#installation"; exit 1
  fi
fi

# Make freshly-installed tools reachable in this same shell.
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) PATH="$HOME/.local/bin:$PATH"; export PATH;; esac

# --- your own copy: clone, strip demolab's history/remote, start a fresh repo ---
say "Fetching demolab into ./${DIR} ..."
git clone --depth 1 "$REPO" "$DIR"
cd "$DIR"
rm -rf .git
# Drop upstream-only deploy workflow; `task deploy-setup` adds a real one later.
rm -rf .github/workflows/landing.yml
git init -q && git add -A && git commit -q -m "Start my lab from demolab"

# --- lay down the bare structure (so there are real folders to review, agent or not) ---
say "Scaffolding the folder structure..."
task scaffold

cat <<EOF

--------------------------------------------------------------
demolab is ready in ./${DIR}

Next, either:
  - Open your coding agent in ./${DIR} and paste:
      ${PROMPT}
  - Or explore by hand:
      cd ${DIR} && task add-demo-content && task dev
--------------------------------------------------------------
EOF
