# Runbook: Update demolab from upstream

Triggers: **"update demolab"**, "update from upstream", "pull the latest demolab", "get the newest engine". Goal: refresh the demolab **engine** to the latest upstream while leaving everything that's *yours* — branding, content, deps — untouched.

**The model: the engine is a black box; your stuff lives outside it.** Updating is a **vendor-copy**, not a merge. Three tiers:

- **Swap wholesale** (pure upstream — the user never edits these): the entire **`demolab-engine/`** — `build/` (the Typst engine), `runbooks/` (these runbooks), and `guides/` (`RULES.md`, `GLOSSARY.md`, `HOUSE-STYLE.md`, `STRUCTURE.md`). Nothing user-owned lives in there, so it's a clean overwrite.
- **Never touch** (100% the user's): `demolab.yaml` (their branding, if present), `writings/`, `tools/`, `experiments/`, `artifacts/`, `temp/`.
- **Reconcile carefully** (framework, but often customised or pinned to root): `AGENTS.md`, `Taskfile.yml`, `.github/workflows/deploy.yml`, `pyproject.toml`, `README.md`. Diff these and apply changes *with the user's say-so* — don't clobber their added tasks, deps, or prose.

Branding (the wordmark + PDF titles) lives in the root `demolab.yaml`, *outside* the swapped dirs, so overwriting the engine can't touch the user's identity — that's the point. Drive it interactively:

1. **Fetch upstream read-only** — never touches the working tree:
   `git clone --depth 1 "${DEMOLAB_UPSTREAM:-https://github.com/eoinmurray/demolab.git}" /tmp/demolab-upstream`.
2. **Show what would change.** `diff -ru demolab-engine /tmp/demolab-upstream/demolab-engine`. Summarise in plain terms. If nothing differs, you're already current — stop.
3. **Swap the black box.** Overwrite the two upstream-owned dirs:
   `rsync -a --delete /tmp/demolab-upstream/demolab-engine/ demolab-engine/`. `demolab.yaml` and all content are untouched by construction — they aren't in `demolab-engine/`.
4. **Check the override surface.** If upstream added a new branding key (compare `demolab-engine/build/lib.typ`'s `default-brand` against the user's `demolab.yaml`), mention it — the engine defaults every key, so nothing breaks, but the user may want to set it.
5. **Reconcile the root/framework files** from the third tier above — diff each, present the upstream change, apply only what the user approves. CI and `pyproject.toml` especially: *merge*, never blind-overwrite (their deps live there).
6. **Verify.** `task build` and `task test`; open `task dev` and confirm branding + content survived and the new features work.
7. **Clean up.** Remove the temp clone. Commit describing what was pulled.

Notes: this is a copy, so upstream's git history doesn't come along — deliberate; the user's repo keeps its own single line of history. If a swap ever fights a local edit *inside* `demolab-engine/build/`, that edit was in the wrong place — the engine is a black box, so lift the customisation into `demolab.yaml` (or propose it upstream as a new config knob) and re-swap.
