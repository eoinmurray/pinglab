# Runbook: Update demolab from upstream

Refresh the demolab **engine** to the latest upstream while leaving everything that's *yours* — branding, content, deps — untouched.

**Triggers** — say any of these, or just `UPDATE`: **"update demolab"**, "update from upstream", "pull the latest demolab", "get the newest engine".

**The model: the engine is a black box; your stuff lives outside it.** Updating is a **vendor-copy**, not a merge. Three tiers:

- **Swap wholesale** (pure upstream — the user never edits these): the entire **`demolab-engine/`** — `build/` (the Typst engine), `runbooks/` (these runbooks), `guides/` (`RULES.md`, `GLOSSARY.md`, `HOUSESTYLE.md`, `STRUCTURE.md`, `SUPPORT.md`), and `scaffold/` (the `skeleton/` + `demo/` overlays and `demo-manifest.json`). Nothing user-owned lives in there, so it's a clean overwrite — a swap refreshes the scaffold/demo too, and since the user's content was copied *out* into the working tree, it's untouched.
- **Never touch** (100% the user's): `demolab.yaml` (their branding, if present), `HOUSESTYLE.local.md` (their house-style overrides, if present), `writings/`, `tools/`, `experiments/`, `artifacts/`, `temp/`.
- **Reconcile carefully** (framework, but often customised or pinned to root): `AGENTS.md`, `Taskfile.yml`, `.github/workflows/deploy.yml`, `pyproject.toml`, `README.md`. Diff these and apply changes *with the user's say-so* — don't clobber their added tasks, deps, or prose.

Branding (the wordmark + PDF titles) lives in the root `demolab.yaml`, *outside* the swapped dirs, so overwriting the engine can't touch the user's identity — that's the point. Drive it interactively:

1. **Fetch upstream read-only** — never touches the working tree:
   `git clone --depth 1 "${DEMOLAB_UPSTREAM:-https://github.com/eoinmurray/demolab.git}" /tmp/demolab-upstream`.
2. **Compare versions and read the changelog.** `cat demolab-engine/VERSION` (yours) vs `cat /tmp/demolab-upstream/demolab-engine/VERSION` (upstream). If they match, you're current — stop. Otherwise show the [`CHANGELOG.md`](../CHANGELOG.md) entries between the two versions so the user sees *what* they're getting. **If the major version differs, stop and get explicit confirmation** — a major bump means the contract, firewall, or `meta` schema changed in a way that may need edits to their content; walk them through it before touching anything.
3. **Show what would change.** `diff -ru demolab-engine /tmp/demolab-upstream/demolab-engine`. Summarise in plain terms.
4. **Swap the black box.** Overwrite the two upstream-owned dirs:
   `rsync -a --delete /tmp/demolab-upstream/demolab-engine/ demolab-engine/`. `demolab.yaml` and all content are untouched by construction — they aren't in `demolab-engine/`.
5. **Check the override surface.** If upstream added a new branding key (compare `demolab-engine/build/lib.typ`'s `default-brand` against the user's `demolab.yaml`), mention it — the engine defaults every key, so nothing breaks, but the user may want to set it.
6. **Reconcile the root/framework files** from the third tier above — diff each, present the upstream change, apply only what the user approves. CI and `pyproject.toml` especially: *merge*, never blind-overwrite (their deps live there).
7. **Verify.** `task build` and `task test`; open `task dev` and confirm branding + content survived and the new features work.
8. **Clean up.** Remove the temp clone. Commit describing what was pulled.

Notes: this is a copy, so upstream's git history doesn't come along — deliberate; the user's repo keeps its own single line of history. If a swap ever fights a local edit *inside* `demolab-engine/build/`, that edit was in the wrong place — the engine is a black box, so lift the customisation into `demolab.yaml` (or propose it upstream as a new config knob) and re-swap.
