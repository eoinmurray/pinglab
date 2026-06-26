// Paper-feed sources. Add a new feed by appending an item to `sources`.
// Handlers for each `type` live in fetch.ts:
//   rss               — any RSS/Atom feed; needs `url`
//   arxiv             — arXiv category via the export API; needs `category`
//   pubmed            — NCBI E-utilities query; needs `query`
//   semantic_scholar  — "papers like these" recommendations; seeds from a .bib
// Everything is best-effort: a source that errors or times out is logged and
// skipped, never fatal.

export type Source =
  | { type: "arxiv"; name: string; category: string; max?: number }
  | { type: "rss"; name: string; url: string }
  | { type: "pubmed"; name: string; query: string; max?: number }
  | { type: "semantic_scholar"; name: string; seedsFromBib: string; max?: number };

// Prose description of the project — used (with `keywords`) to pre-rank
// candidates and handed to the summariser as the relevance yardstick.
export const projectProfile = `
Training pyramidal-interneuron gamma (PING) spiking networks end to end with
surrogate-gradient descent. Interests: gamma oscillations (30-80 Hz),
excitatory-inhibitory balance, conductance-based spiking neuron models,
surrogate-gradient / backprop-through-time training of SNNs, sparse and
low-rate cortical firing, the metabolic/energy cost of spikes, mean-field and
Hopf-bifurcation analyses of E-I loops, and rate-vs-timing coding.
`.trim();

export const keywords = [
  "gamma", "PING", "oscillation", "inhibition", "interneuron",
  "excitatory-inhibitory", "E/I balance", "spiking neural network",
  "surrogate gradient", "backpropagation through time", "sparse coding",
  "sparse firing", "energy", "metabolic", "conductance-based", "mean-field",
  "Hopf bifurcation", "parvalbumin", "synchrony", "rate coding",
];

// Only keep items published within this many days (RSS / arXiv / PubMed only;
// Semantic Scholar recommendations are by relevance, not date).
export const maxAgeDays = 14;

export const sources: Source[] = [
  { type: "arxiv", name: "arXiv q-bio.NC", category: "q-bio.NC", max: 60 },
  { type: "arxiv", name: "arXiv cs.NE", category: "cs.NE", max: 60 },

  { type: "rss", name: "bioRxiv Neuroscience", url: "https://connect.biorxiv.org/biorxiv_xml.php?subject=neuroscience" },
  { type: "rss", name: "eLife", url: "https://elifesciences.org/rss/recent.xml" },
  { type: "rss", name: "PLoS Comput Biol", url: "https://journals.plos.org/ploscompbiol/feed/atom" },
  { type: "rss", name: "J Neurosci", url: "https://www.jneurosci.org/rss/current.xml" },
  { type: "rss", name: "Nature Neuroscience", url: "https://www.nature.com/neuro.rss" },
  { type: "rss", name: "Nature Rev Neurosci", url: "https://www.nature.com/nrn.rss" },

  {
    type: "pubmed",
    name: "PubMed: gamma+E/I",
    query: "(gamma oscillation[Title/Abstract]) AND (inhibition[Title/Abstract] OR interneuron[Title/Abstract] OR spiking[Title/Abstract])",
    max: 40,
  },

  {
    type: "semantic_scholar",
    name: "S2 recommendations",
    seedsFromBib: "src/papers/paper001/references.bib",
    max: 40,
  },
];
