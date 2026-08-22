#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
skills="lab experiment campaign writing publish lab-doctor"

for name in $skills; do
  file="$repo_dir/.agents/skills/$name/SKILL.md"
  [ -f "$file" ] || {
    printf 'Missing skill: %s\n' "$file" >&2
    exit 1
  }
  grep -q "^name: $name$" "$file" || {
    printf 'Skill name mismatch: %s\n' "$file" >&2
    exit 1
  }
  grep -Fq "description: Use only when the user explicitly invokes \$$name" "$file" || {
    printf 'Skill is not explicit-invocation-only: %s\n' "$file" >&2
    exit 1
  }
done

for command in lab experiment campaign writing publish lab-doctor; do
  grep -q "\`\$$command" "$repo_dir/AGENTS.md" || {
    printf 'Lexicon command missing from AGENTS.md: %s\n' "$command" >&2
    exit 1
  }
done

grep -Fq 'Project skills are opt-in command handlers' "$repo_dir/AGENTS.md" || {
  printf 'Project skill activation guard missing from AGENTS.md\n' >&2
  exit 1
}

if grep -R "\[TODO:" "$repo_dir/.agents/skills" >/dev/null 2>&1; then
  printf 'Unfinished skill placeholder found\n' >&2
  exit 1
fi

printf 'Validated 6 Pinglab skills\n'
