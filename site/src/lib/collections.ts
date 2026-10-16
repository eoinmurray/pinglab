import type { Post } from "./posts";

const collections = [
  {
    id: "gamma-gated-sparsity",
    label: "Gamma-gated sparsity",
    description:
      "Experiments on gamma rhythms, sparse neural activity, and learning in spiking networks.",
  },
  {
    id: "neuromodulation",
    label: "Neuromodulation",
    description:
      "Experiments on how changes to neural dynamics affect network evidence, accuracy, and activity.",
  },
  {
    id: "datasets",
    label: "Datasets",
    description:
      "Training banks, recordings, and other reusable experimental data, with their provenance and diagnostics.",
  },
  {
    id: "demo",
    label: "Uncategorized",
    description: "Articles and experiments without a subject collection.",
  },
  {
    id: "guides",
    label: "Guides",
    description: "Technical explanations of the models, methods, and tools used in Pinglab.",
  },
  {
    id: "animations",
    label: "Animations",
    description: "Experiments and guides with interactive PixiJS animations.",
  },
  {
    id: "paper-inspired",
    label: "Paper-inspired",
    description: "Experiments inspired by published research papers.",
  },
  {
    id: "literature",
    label: "Literature",
    description: "Research papers, reading notes, and discussions of published findings.",
  },
  {
    id: "weekly-feed",
    label: "Weekly feed",
    description: "Periodic summaries of research progress and recent work.",
  },
  {
    id: "notes",
    label: "Notes",
    description: "Practical project notes, infrastructure guidance, and operational workflows.",
  },
  {
    id: "slides",
    label: "Slides",
    description: "Talks and journal-club presentations on research and related ideas.",
  },
];

export function collectionMemberships(value: string | string[]): string[] {
  return typeof value === "string" ? [value] : value;
}

export function collectionLabel(id: string): string {
  return collections.find((collection) => collection.id === id)?.label ?? id;
}

export function compareCollections(a: string, b: string): number {
  const rank = (id: string) => {
    const index = collections.findIndex((collection) => collection.id === id);
    return index < 0 ? collections.length : index;
  };
  return rank(a) - rank(b) || a.localeCompare(b);
}

export function collectionId(collection: string): string {
  return `collection-${collection}`;
}
export function collectionUrl(collection: string): string {
  return `/collections/${encodeURIComponent(collection)}`;
}

export function collectionDescription(id: string): string {
  return (
    collections.find((collection) => collection.id === id)?.description ??
    `Articles in ${collectionLabel(id)}.`
  );
}

export function collectionSummary(posts: readonly Post[]): { count: number; latestDate?: string } {
  const members = [...new Map(posts.map((post) => [post.href, post])).values()];
  return {
    count: members.length,
    latestDate: members
      .map((post) => post.data.updated_at)
      .sort()
      .at(-1),
  };
}
