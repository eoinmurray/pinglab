// Notebook lifecycle status — the single source of truth for the `status:`
// frontmatter field. Notebooks move: draft → building → revising → final,
// and may move backward freely (a `final` entry can be reopened).
//
//   draft     ideas + context; prose-heavy, no trusted results, expect churn.
//   building  distilled to the core claim; code + plots landing (LLM's turn).
//   revising  results exist; under human review / changes in flight.
//   final     reviewed and approved — ready to send.

export const NOTEBOOK_STATUSES = ["draft", "building", "revising", "final"] as const;
export type NotebookStatus = (typeof NOTEBOOK_STATUSES)[number];

export const STATUS_META: Record<
  NotebookStatus,
  { label: string; description: string }
> = {
  draft: {
    label: "Draft",
    description: "Ideas and context — prose-heavy, no trusted results yet; expect churn.",
  },
  building: {
    label: "Building",
    description: "Distilled to the core claim; code and plots landing.",
  },
  revising: {
    label: "Revising",
    description: "Results exist; under review with changes in flight.",
  },
  final: {
    label: "Final",
    description: "Reviewed and approved — ready to send.",
  },
};

// Absent status defaults to draft — safe for new stubs (never over-claims
// approval). Published entries set `status: final` explicitly.
export const DEFAULT_STATUS: NotebookStatus = "draft";

export function normalizeStatus(raw: unknown): NotebookStatus {
  if (raw == null || raw === "") return DEFAULT_STATUS;
  if (
    typeof raw === "string" &&
    (NOTEBOOK_STATUSES as readonly string[]).includes(raw)
  ) {
    return raw as NotebookStatus;
  }
  // Build-time enforcement: an unknown status is a frontmatter error.
  throw new Error(
    `Unknown notebook status ${JSON.stringify(raw)}; expected one of ${NOTEBOOK_STATUSES.join(", ")}`,
  );
}
