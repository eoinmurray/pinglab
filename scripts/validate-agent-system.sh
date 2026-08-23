#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
skills="abstract hypo pinglab exp publish"

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
  grep -Fq "description: Use only when the user explicitly invokes" "$file" || {
    printf 'Skill is not explicit-invocation-only: %s\n' "$file" >&2
    exit 1
  }
done

for command in abstract hypo pinglab exp publish; do
  grep -q "\`$command" "$repo_dir/AGENTS.md" || {
    printf 'Lexicon command missing from AGENTS.md: %s\n' "$command" >&2
    exit 1
  }
done

grep -Fq 'Project skills are opt-in command handlers' "$repo_dir/AGENTS.md" || {
  printf 'Project skill activation guard missing from AGENTS.md\n' >&2
  exit 1
}

if grep -Fq '| Command | Input noun | Output noun |' "$repo_dir/AGENTS.md"; then
  printf 'Verb signatures must live in skills, not AGENTS.md\n' >&2
  exit 1
fi

sed -n 's/.*| `\(\.agents\/skills\/[^`]*\/SKILL\.md\)` |$/\1/p' \
  "$repo_dir/AGENTS.md" | sort -u | while IFS= read -r handler; do
  [ -f "$repo_dir/$handler" ] || {
    printf 'Lexicon handler missing: %s\n' "$handler" >&2
    exit 1
  }
done

if grep -R "\[TODO:" "$repo_dir/.agents/skills" >/dev/null 2>&1; then
  printf 'Unfinished skill placeholder found\n' >&2
  exit 1
fi

ruby -e '
  root = ARGV.fetch(0)
  registry = File.join(root, ".agents", "NOUNS.md")
  abort "Missing noun registry: #{registry}" unless File.file?(registry)
  nouns = File.read(registry).scan(/^## `([^`]+)`$/).flatten
  abort "No nouns declared: #{registry}" if nouns.empty?

  Dir[File.join(root, ".agents", "skills", "*", "SKILL.md")].sort.each do |file|
    text = File.read(file)
    signature = text[/^## Signature\n(.*?)(?=^## |\z)/m, 1]
    abort "Missing signature: #{file}" unless signature
    abort "Malformed signature table: #{file}" unless signature.include?("| Verb | Input noun | Output noun |")
    rows = signature.lines.grep(/^\| `[^`]+` \|/)
    abort "Empty signature: #{file}" if rows.empty?
    rows.each do |row|
      columns = row.split("|").map(&:strip)
      [columns.fetch(2), columns.fetch(3)].each do |column|
        column.scan(/`([^`]+)`/).flatten.each do |noun|
          abort "Unknown noun #{noun}: #{file}" unless nouns.include?(noun)
        end
      end
    end
  end
' "$repo_dir"

printf 'Validated 5 Pinglab skills\n'
