export const REPO_OWNER = "eoinmurray";
export const REPO_NAME = "pinglab";
export const REPO_BRANCH = "main";
export const REPO_URL = `https://github.com/${REPO_OWNER}/${REPO_NAME}`;

export function blobUrl(path: string, opts: { sha?: string; line?: number; range?: [number, number] } = {}): string {
  const ref = (opts.sha ?? REPO_BRANCH).replace(/\s*\(dirty\)\s*$/i, "");
  const anchor = opts.range
    ? `#L${opts.range[0]}-L${opts.range[1]}`
    : opts.line != null
      ? `#L${opts.line}`
      : "";
  return `${REPO_URL}/blob/${ref}/${path}${anchor}`;
}

export function commitUrl(sha: string): string {
  const clean = sha.replace(/\s*\(dirty\)\s*$/i, "");
  return `${REPO_URL}/commit/${clean}`;
}

export function treeUrl(sha: string): string {
  const clean = sha.replace(/\s*\(dirty\)\s*$/i, "");
  return `${REPO_URL}/tree/${clean}`;
}

export function isDirty(sha: string | undefined | null): boolean {
  return typeof sha === "string" && /\(dirty\)/i.test(sha);
}

export function shortSha(sha: string | undefined | null): string {
  if (!sha) return "";
  return sha.replace(/\s*\(dirty\)\s*$/i, "").slice(0, 7);
}
