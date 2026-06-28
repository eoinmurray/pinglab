// Central registry of collections. Collections are still *derived* from the
// `collection:` frontmatter on articles/notebooks — this file just attaches a
// display label and a one-line description to each known slug. A slug with no
// entry here still works; it falls back to a title-cased label and no
// description. Add or edit collections by editing this map.

export type CollectionMeta = {
  label?: string; // overrides the title-cased slug
  description?: string;
};

export const COLLECTIONS: Record<string, CollectionMeta> = {
  "gamma-gated-sparsity": {
    label: "Gamma-Gated Sparsity",
    description: "Training PING networks so the gamma rhythm gates excitatory spikes.",
  },
  literature: {
    label: "Literature Reviews",
    description: "Pedagogical readings of key papers.",
  },
  "weekly-feed": {
    label: "Weekly Feed",
    description: "A weekly triage of new papers worth reading.",
  },
  documentation: {
    description: "Reference docs for the codebase and its conventions.",
  },
  "ai-state": {
    label: "AI State",
    description: "The asynchronous–irregular balanced state, and where PING sits.",
  },
  videos: {
    description: "Short animations of PING dynamics.",
  },
  bayesian: {
    description: "Bayesian inference and methods.",
  },
  "ping-as-a-clock": {
    label: "PING as a Clock",
    description: "The gamma cycle read as a self-generated timing reference.",
  },
};

// Explicit display order for the homepage and the collections index. Slugs not
// listed here sort after these, alphabetically.
export const COLLECTION_ORDER = [
  "gamma-gated-sparsity",
  "ai-state",
  "ping-as-a-clock",
  "bayesian",
  "literature",
  "weekly-feed",
  "videos",
  "documentation",
];

export function collectionRank(slug: string): number {
  const i = COLLECTION_ORDER.indexOf(slug);
  return i === -1 ? COLLECTION_ORDER.length : i;
}

const ACRONYMS = new Set(["coba", "ping", "snn", "ai", "cli"]);

function titleCase(slug: string): string {
  return slug
    .split("-")
    .map((w) => (ACRONYMS.has(w) ? w.toUpperCase() : w.charAt(0).toUpperCase() + w.slice(1)))
    .join(" ");
}

export function collectionLabel(slug: string): string {
  return COLLECTIONS[slug]?.label ?? titleCase(slug);
}

export function collectionDescription(slug: string): string | undefined {
  return COLLECTIONS[slug]?.description;
}
