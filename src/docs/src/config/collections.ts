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
    description:
      "The core project: training a pyramidal–interneuron gamma (PING) spiking network end to end, and showing the architecture-generated rhythm gates the excitatory population to an order-of-magnitude-lower spike rate at matched accuracy. The manuscript, its literature review, and the notebooks behind every claim.",
  },
  literature: {
    description:
      "Close readings of key papers — pedagogical reviews that tie the external literature back to this project — plus the rolling paper feed.",
  },
  documentation: {
    description:
      "Reference docs for the codebase and its conventions: the model and synapses, training and gradient stabilisation, metrics, the parameter and unit system, the CLI, and the repo house rules.",
  },
  "ai-state": {
    label: "AI State",
    description:
      "The asynchronous–irregular balanced state: the Vreeswijk–Sompolinsky regime and where a PING network sits relative to it.",
  },
  videos: {
    description:
      "Animations of PING dynamics — spikes, drive, input rate, integration step, and E→I coupling — rendered as short videos.",
  },
  bayesian: {
    description: "Bayesian inference and methods — literature notes.",
  },
  "ping-as-a-clock": {
    label: "PING as a Clock",
    description:
      "The PING rhythm read as a timing reference — the gamma cycle as the clock that paces excitatory spiking.",
  },
};

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
