# Runbook: Doctor the repo (conformance check)

Triggers: **"doctor the repo"**, "check the repo follows the conventions", "audit demolab conformance", "does this repo still obey the rules". Goal: verify the repo obeys demolab's conventions and report every violation with the rule it breaks and a `file:line` — a **health check, not an auto-fixer**. Fix only what the user approves, then re-run.

`../guides/RULES.md` is the source of truth for every rule; this runbook is the *inspection* that the repo obeys it — it cites each rule, it doesn't restate it. (There is no `task doctor` — this replaced it, and folds the toolchain check in as step 0.)

Drive it interactively: run each check, collect the hits, present **one** report grouped by severity, then offer to fix.

## 0. Toolchain present
`command -v uv typst task` — all three must resolve. If any is missing, give the install (macOS: `brew install uv typst go-task`; `uv` also via `curl -LsSf https://astral.sh/uv/install.sh | sh`).

## 1. Build + tests are green (the coarse signal)
- `task build` — compiles all three targets with no error.
- `task test` — `uv run pytest` passes.
A red build or test is the first thing to fix; everything below is finer-grained.

## 2. Mechanical checks (run these — each hit is a violation)

```sh
# RULES §8.4 — every tool ships a test. A "tool" is a dir with a tool.py
# (skips __pycache__ and other non-tool dirs).
for d in tools/*/; do t=$(basename "$d"); [ -f "$d/tool.py" ] || continue; \
  [ -f "$d/test_$t.py" ] || echo "MISSING TEST: $d (needs test_$t.py)"; done

# RULES §4.5 — import boundary.
grep -rnE '^\s*(from|import) experiments' tools/ && echo "VIOLATION: a tool imports experiments/"
grep -rnE '^\s*(from|import) tools'       experiments/ && echo "VIOLATION: a runner imports a tool (use its CLI)"

# RULES §6.1 — every non-slide writing is well-formed (#let meta + #let body).
for f in writings/*.typ; do case "$f" in *.slide.typ) continue;; esac; \
  { grep -qE '^#let meta' "$f" && grep -qE '^#let body' "$f"; } || echo "MALFORMED WRITING: $f"; done

# RULES §7 — every experiment has its committed record + writeup (id threads through).
for f in experiments/exp*.py; do id=$(basename "$f" .py); \
  [ -f "artifacts/data/$id/numbers.json" ] || echo "MISSING RECORD: artifacts/data/$id/numbers.json"; \
  [ -f "writings/$id.typ" ] || echo "MISSING WRITEUP: writings/$id.typ"; done

# RULES §4.7 — provenance stamped into every committed record.
grep -rL '_provenance' artifacts/data/*/numbers.json 2>/dev/null | sed 's/^/MISSING PROVENANCE: /'

# RULES §2.1 — no agent authorship anywhere in history.
git log --format='%an|%cn|%b' | grep -iE 'co-authored-by:.*(claude|anthropic|\[bot\])|generated with|claude' \
  && echo "VIOLATION: agent authorship in git history"

# RULES §5.1 / §5.3 — scratch is gitignored, never tracked.
git ls-files temp artifacts/site | head -1 | grep -q . && echo "VIOLATION: temp/ or artifacts/site/ is tracked"

# RULES §3.3 — branding belongs in demolab.yaml, not hacked into the black box.
# (Informational — compare against upstream during "update demolab".)
```

## 3. Judgment checks (no grep suffices — the agent decides)

- **§4.2 — tools emit data, not plots.** Scan each `tools/*/tool.py` for figure-drawing (`savefig`, `plt.`, writing a `.png`); the exception is a *rendering* tool writing an `.mp4` (declared `headline_video`).
- **§4.1 — no forced one-off tools.** Flag any `tools/<t>` used by exactly one experiment and unlikely to be reused — it may belong inline; don't force a change.
- **§6.2 — numbers don't drift.** Spot-check that each `writings/*.typ` pulls figures/tables from `json(...)` / `#image(...)` / `#numbers-table(...)`, not hand-typed literals.
- **§3 — docs match reality.** Runbook counts, path references, and the firewall in `../guides/RULES.md` still describe the actual tree.

## 4. Report
Present one grouped report:

- **Broken** (build/test red, agent authorship, import-boundary breach) — fix now.
- **Conformance** (missing tests, malformed writings, missing records/provenance, tracked scratch) — fix before publishing.
- **Advisory** (judgment flags: plots-in-tools, forced one-offs, possible drift, stale docs) — the user's call.

For each finding: cite the rule by its `§N.M` in [`../guides/RULES.md`](../guides/RULES.md), the `file:line`, and the fix. Apply only what the user approves, then re-run the relevant checks to confirm green.
