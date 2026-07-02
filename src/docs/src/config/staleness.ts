/**
 * Build-time run-provenance + staleness for a notebook entry.
 *
 * Reads the run manifest (public/figures/notebooks/<slug>/_manifest.json) that
 * the runner stamps at launch, then asks git whether the code the run depended
 * on has changed since the commit the run was locked to. The result drives the
 * per-entry status bar (last run, commit lock, staleness pip).
 *
 * Everything degrades to state "unknown" rather than throwing: a missing
 * manifest, a shallow CI clone, a git that is not on PATH, or a locked commit
 * that is not in history all yield a bar that says so instead of failing the
 * build. (CI must still fetch full history — fetch-depth: 0 — for the diff to
 * mean anything; without it every entry reads "unknown".)
 */
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";

// Walk up from the build's working directory (src/docs during `astro build`)
// to the repo root, identified by the src/cli + src/notebooks pair. Anchoring
// on cwd rather than import.meta.url survives Vite bundling the config module
// (which relocates import.meta.url and silently broke root resolution).
function findRepoRoot(): string {
  let dir = process.cwd();
  for (let i = 0; i < 8; i++) {
    if (existsSync(join(dir, "src/cli")) && existsSync(join(dir, "src/notebooks"))) {
      return dir;
    }
    const up = dirname(dir);
    if (up === dir) break;
    dir = up;
  }
  return process.cwd();
}

const REPO_ROOT = findRepoRoot();

export type RunState = "fresh" | "stale" | "irreproducible" | "unknown";

export interface Provenance {
  state: RunState;
  runId?: string;
  runAt?: string; // ISO-8601
  sha?: string;
  dirty?: boolean;
  host?: string;
  commitsSince?: number; // populated when state === "stale"
  changedFiles?: string[]; // sample of dependency files that changed
  note?: string; // human reason for "unknown" / "irreproducible"
}

function git(args: string[]): string | null {
  try {
    return execFileSync("git", args, {
      cwd: REPO_ROOT,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
  } catch {
    return null;
  }
}

// The code a notebook run reproduces from: its own runner, the shared notebook
// helpers, and the oscilloscope CLI. A commit touching any of these since the
// locked sha makes the published figures potentially stale.
function deps(slug: string): string[] {
  return [`src/notebooks/${slug}.py`, "src/notebooks/helpers", "src/cli"];
}

function manifestPath(slug: string): string {
  return join(
    REPO_ROOT,
    "src/docs/public/figures/notebooks",
    slug,
    "_manifest.json",
  );
}

export function provenance(slug: string): Provenance {
  const mp = manifestPath(slug);
  if (!existsSync(mp)) {
    // Backfill for pre-manifest entries: the last commit that touched the
    // committed figures dir is a rough "last run" date. No commit lock.
    const iso = git([
      "log",
      "-1",
      "--format=%cI",
      "--",
      `src/docs/public/figures/notebooks/${slug}`,
    ]);
    return { state: "unknown", runAt: iso ?? undefined, note: "no run manifest" };
  }

  let m: any;
  try {
    m = JSON.parse(readFileSync(mp, "utf8"));
  } catch {
    return { state: "unknown", note: "unreadable manifest" };
  }

  const base: Provenance = {
    state: "unknown",
    runId: m.run_id,
    runAt: m.run_at,
    sha: m.git_sha,
    dirty: m.dirty === true,
    host: m.host,
  };

  // A dirty run cannot be reproduced from any commit — there is nothing to
  // diff against, so staleness is undefined; flag it rather than guess.
  if (m.dirty === true) {
    return { ...base, state: "irreproducible", note: "run had uncommitted changes" };
  }
  if (!m.git_sha || m.git_sha === "unknown") {
    return { ...base, note: "run recorded no commit" };
  }
  // Is the locked commit reachable in this checkout? (Fails on a shallow CI
  // clone or a rebased-away commit.) git cat-file -e exits 0 → "" here.
  if (git(["cat-file", "-e", `${m.git_sha}^{commit}`]) === null) {
    return { ...base, note: "locked commit not in history" };
  }

  const log = git([
    "log",
    "--format=%H",
    `${m.git_sha}..HEAD`,
    "--",
    ...deps(slug),
  ]);
  if (log === null) {
    return { ...base, note: "staleness unavailable" };
  }
  const commits = log ? log.split("\n").filter(Boolean) : [];
  if (commits.length === 0) {
    return { ...base, state: "fresh" };
  }
  const files = git([
    "diff",
    "--name-only",
    `${m.git_sha}..HEAD`,
    "--",
    ...deps(slug),
  ]);
  return {
    ...base,
    state: "stale",
    commitsSince: commits.length,
    changedFiles: files ? files.split("\n").filter(Boolean).slice(0, 8) : undefined,
  };
}
