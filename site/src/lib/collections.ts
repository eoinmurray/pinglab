const collections = [
  { id: 'gamma-gated-sparsity', label: 'Gamma-gated sparsity' },
  { id: 'neuromodulation', label: 'Neuromodulation' },
  { id: 'demo', label: 'Uncategorized' },
  { id: 'documentation', label: 'Documentation' },
  { id: 'literature', label: 'Literature' },
  { id: 'weekly-feed', label: 'Weekly feed' },
  { id: 'notes', label: 'Notes' },
  { id: 'slides', label: 'Slides' },
];

export function collectionLabel(id: string): string {
  return collections.find(collection => collection.id === id)?.label ?? id;
}

export function compareCollections(a: string, b: string): number {
  const rank = (id: string) => {
    const index = collections.findIndex(collection => collection.id === id);
    return index < 0 ? collections.length : index;
  };
  return rank(a) - rank(b) || a.localeCompare(b);
}

export function collectionId(collection: string): string { return `collection-${collection}`; }
export function collectionUrl(collection: string): string {
  return `/collections/${encodeURIComponent(collection)}`;
}
