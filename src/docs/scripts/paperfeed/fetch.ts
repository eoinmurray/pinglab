#!/usr/bin/env bun
/**
 * Fetch candidate papers for the feed from every source in sources.ts.
 *
 * Pulls RSS/Atom feeds, arXiv categories, a PubMed query, and Semantic Scholar
 * "papers like these" recommendations (seeded from a .bib). Normalises, dedupes
 * against a running seen-list, pre-ranks by keyword overlap with the project
 * profile, and writes candidates.json for the summariser to turn into the feed
 * article (src/docs/src/pages/articles/ar061.mdx). Every source is best-effort:
 * one that errors or times out is logged and skipped, never fatal.
 *
 * Run:  bun run feed            (from src/docs)
 *   or  bun scripts/paperfeed/fetch.ts [--max-age-days N] [--no-seen]
 */
import { XMLParser } from "fast-xml-parser";
import { mkdir, readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { keywords, maxAgeDays, sources, type Source } from "./sources.ts";

const HERE = import.meta.dir;
const REPO = resolve(HERE, "../../../..");
const SEEN_JSON = join(HERE, "seen.json");
const CANDIDATES_JSON = join(HERE, "candidates.json");
const ARCHIVE_DIR = join(HERE, "archive");

const UA = {
  "User-Agent": "pinglab-paperfeed/1.0 (mailto:me@eoinmurray.info)",
  Accept: "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
};
const TIMEOUT = 30_000;

const parser = new XMLParser({ ignoreAttributes: false, attributeNamePrefix: "@_" });
const log = (m: string) => console.error(m);

type Candidate = {
  title: string;
  abstract: string;
  authors: string[];
  url: string;
  doi?: string;
  arxiv?: string;
  published?: string;
  source: string;
  recommended?: boolean;
  id?: string;
  score?: number;
};

// --------------------------------------------------------------------------- //
// Normalisation helpers
// --------------------------------------------------------------------------- //
const toArray = <T>(x: T | T[] | undefined | null): T[] =>
  Array.isArray(x) ? x : x == null ? [] : [x];

function txt(v: any): string {
  if (v == null) return "";
  if (typeof v === "string" || typeof v === "number") return String(v);
  if (typeof v === "object" && "#text" in v) return String(v["#text"]);
  return "";
}

const clean = (s: string): string =>
  s.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();

const normTitle = (t: string): string =>
  (t || "").toLowerCase().replace(/[^a-z0-9 ]/g, "").replace(/\s+/g, " ").trim();

function sha1short(s: string): string {
  return new Bun.CryptoHasher("sha1").update(s).digest("hex").slice(0, 16);
}

function makeId(c: Candidate): string {
  if (c.doi) return "doi:" + c.doi.toLowerCase().trim();
  if (c.arxiv) return "arxiv:" + c.arxiv.replace(/v\d+$/, "");
  return "title:" + sha1short(normTitle(c.title));
}

function extractDoi(...texts: string[]): string | undefined {
  for (const t of texts) {
    const m = t?.match(/10\.\d{4,9}\/[-._;()/:a-z0-9]+/i);
    if (m) return m[0].replace(/\.$/, "");
  }
  return undefined;
}

function isoDate(s: string): string | undefined {
  if (!s) return undefined;
  const d = new Date(s);
  return isNaN(d.getTime()) ? undefined : d.toISOString().slice(0, 10);
}

const atomLink = (link: any): string => {
  const arr = toArray(link);
  const alt = arr.find((l: any) => l?.["@_rel"] === "alternate") ?? arr[0];
  if (!alt) return "";
  return typeof alt === "string" ? alt : (alt["@_href"] ?? "");
};

async function httpFetch(url: string): Promise<Response> {
  return fetch(url, { headers: UA, signal: AbortSignal.timeout(TIMEOUT) });
}
async function httpText(url: string): Promise<string> {
  const r = await httpFetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.text();
}

// --------------------------------------------------------------------------- //
// Source handlers — each returns normalised candidates
// --------------------------------------------------------------------------- //
async function fetchRss(src: Extract<Source, { type: "rss" }>): Promise<Candidate[]> {
  const doc = parser.parse(await httpText(src.url));
  const out: Candidate[] = [];
  // RSS 2.0 (rss.channel.item) and RSS 1.0/RDF (rdf:RDF.item, used by bioRxiv
  // and nature.com) carry the same item fields; Atom (feed.entry) differs.
  const rssItems = [
    ...toArray<any>(doc?.rss?.channel?.item),
    ...toArray<any>(doc?.["rdf:RDF"]?.item),
  ];
  for (const it of rssItems) {
    const link = typeof it.link === "string" ? it.link : atomLink(it.link);
    out.push({
      title: clean(txt(it.title)),
      abstract: clean(txt(it.description) || txt(it["content:encoded"])),
      authors: toArray(it["dc:creator"]).map(txt).filter(Boolean),
      url: link,
      doi: extractDoi(txt(it.guid), link, txt(it["dc:identifier"]), txt(it["prism:doi"])),
      published: isoDate(txt(it.pubDate) || txt(it["dc:date"])),
      source: src.name,
    });
  }
  for (const e of toArray<any>(doc?.feed?.entry)) {
    const link = atomLink(e.link);
    out.push({
      title: clean(txt(e.title)),
      abstract: clean(txt(e.summary) || txt(e.content)),
      authors: toArray(e.author).map((a: any) => txt(a?.name)).filter(Boolean),
      url: link,
      doi: extractDoi(txt(e.id), link),
      published: isoDate(txt(e.published) || txt(e.updated)),
      source: src.name,
    });
  }
  return out;
}

async function fetchArxiv(src: Extract<Source, { type: "arxiv" }>): Promise<Candidate[]> {
  const url =
    "http://export.arxiv.org/api/query?search_query=cat:" +
    `${src.category}&sortBy=submittedDate&sortOrder=descending&max_results=${src.max ?? 50}`;
  const doc = parser.parse(await httpText(url));
  return toArray<any>(doc?.feed?.entry).map((e) => {
    const idUrl = txt(e.id);
    return {
      title: clean(txt(e.title)),
      abstract: clean(txt(e.summary)),
      authors: toArray(e.author).map((a: any) => txt(a?.name)).filter(Boolean),
      url: atomLink(e.link) || idUrl,
      doi: txt(e["arxiv:doi"]) || undefined,
      arxiv: idUrl.split("/abs/").pop() ?? "",
      published: isoDate(txt(e.published)),
      source: src.name,
    } as Candidate;
  });
}

async function fetchPubmed(src: Extract<Source, { type: "pubmed" }>): Promise<Candidate[]> {
  const base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils";
  const es = await (
    await httpFetch(
      `${base}/esearch.fcgi?db=pubmed&term=${encodeURIComponent(src.query)}` +
        `&retmax=${src.max ?? 40}&retmode=json&sort=most+recent`,
    )
  ).json();
  const ids: string[] = es?.esearchresult?.idlist ?? [];
  if (!ids.length) return [];
  const doc = parser.parse(
    await httpText(`${base}/efetch.fcgi?db=pubmed&id=${ids.join(",")}&retmode=xml`),
  );
  return toArray<any>(doc?.PubmedArticleSet?.PubmedArticle).map((art) => {
    const cit = art?.MedlineCitation;
    const article = cit?.Article;
    let doi: string | undefined;
    for (const eid of toArray<any>(article?.ELocationID)) {
      if (eid?.["@_EIdType"] === "doi") doi = txt(eid);
    }
    const year = txt(article?.Journal?.JournalIssue?.PubDate?.Year);
    const pmid = txt(cit?.PMID);
    return {
      title: clean(txt(article?.ArticleTitle)),
      abstract: clean(toArray(article?.Abstract?.AbstractText).map(txt).join(" ")),
      authors: toArray<any>(article?.AuthorList?.Author)
        .map((a) => [txt(a?.ForeName), txt(a?.LastName)].filter(Boolean).join(" "))
        .filter(Boolean),
      url: pmid ? `https://pubmed.ncbi.nlm.nih.gov/${pmid}/` : "",
      doi,
      published: year ? `${year}-01-01` : undefined,
      source: src.name,
    } as Candidate;
  });
}

async function fetchSemanticScholar(
  src: Extract<Source, { type: "semantic_scholar" }>,
): Promise<Candidate[]> {
  const bib = await readFile(join(REPO, src.seedsFromBib), "utf8");
  const seeds = [...bib.matchAll(/doi\s*=\s*\{([^}]+)\}/g)]
    .map((m) => "DOI:" + m[1].replace(/\\_/g, "_").trim())
    .slice(0, 90);
  if (!seeds.length) return [];
  const res = await fetch(
    "https://api.semanticscholar.org/recommendations/v1/papers" +
      `?fields=title,abstract,authors,year,externalIds,url&limit=${src.max ?? 40}`,
    {
      method: "POST",
      headers: { ...UA, "Content-Type": "application/json" },
      body: JSON.stringify({ positivePaperIds: seeds }),
      signal: AbortSignal.timeout(TIMEOUT),
    },
  );
  if (!res.ok) {
    log(`  [semantic_scholar] HTTP ${res.status}`);
    return [];
  }
  const data = await res.json();
  return toArray<any>(data?.recommendedPapers).map((p) => {
    const ext = p?.externalIds ?? {};
    return {
      title: clean(p?.title ?? ""),
      abstract: clean(p?.abstract ?? ""),
      authors: toArray(p?.authors).map((a: any) => a?.name).filter(Boolean),
      url: p?.url ?? "",
      doi: ext?.DOI,
      arxiv: ext?.ArXiv,
      published: p?.year ? `${p.year}-01-01` : undefined,
      source: src.name,
      recommended: true,
    } as Candidate;
  });
}

const HANDLERS = {
  rss: fetchRss,
  arxiv: fetchArxiv,
  pubmed: fetchPubmed,
  semantic_scholar: fetchSemanticScholar,
} as const;

// --------------------------------------------------------------------------- //
// Scoring + main
// --------------------------------------------------------------------------- //
function score(c: Candidate): number {
  const title = (c.title || "").toLowerCase();
  const abstract = (c.abstract || "").toLowerCase();
  const count = (h: string, n: string) => (n ? h.split(n).length - 1 : 0);
  let s = 0;
  for (const kw of keywords) {
    const k = kw.toLowerCase();
    s += 3 * count(title, k) + count(abstract, k);
  }
  if (c.recommended) s += 2;
  return s;
}

async function main() {
  const argv = Bun.argv.slice(2);
  const noSeen = argv.includes("--no-seen");
  const ageArg = argv.indexOf("--max-age-days");
  const maxAge = ageArg >= 0 ? Number(argv[ageArg + 1]) : maxAgeDays;
  const cutoff = new Date(Date.now() - maxAge * 86_400_000).toISOString().slice(0, 10);

  const seen: Record<string, string> =
    noSeen || !(await Bun.file(SEEN_JSON).exists())
      ? {}
      : await Bun.file(SEEN_JSON).json();

  const raw: Candidate[] = [];
  for (const src of sources) {
    const handler = HANDLERS[src.type] as (s: Source) => Promise<Candidate[]>;
    if (!handler) {
      log(`[skip] unknown source type ${(src as any).type}`);
      continue;
    }
    const t0 = performance.now();
    try {
      const items = await handler(src);
      log(`[ok] ${src.name.padEnd(24)} ${String(items.length).padStart(3)} items (${((performance.now() - t0) / 1000).toFixed(1)}s)`);
      raw.push(...items);
    } catch (e: any) {
      log(`[warn] ${src.name.padEnd(24)} ${e?.name}: ${e?.message}`);
    }
  }

  // Normalise ids, drop dupes (this run + seen), apply age filter.
  const byId = new Map<string, Candidate>();
  for (const c of raw) {
    if (!c.title) continue;
    c.id = makeId(c);
    if (!c.recommended && c.published && c.published < cutoff) continue;
    if (seen[c.id]) continue;
    const existing = byId.get(c.id);
    if (existing) {
      if (c.abstract.length > existing.abstract.length) byId.set(c.id, c);
      continue;
    }
    byId.set(c.id, c);
  }

  const cands = [...byId.values()];
  for (const c of cands) c.score = score(c);
  cands.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));

  const today = new Date().toISOString().slice(0, 10);
  const payload = { generated: today, n: cands.length, candidates: cands };
  await Bun.write(CANDIDATES_JSON, JSON.stringify(payload, null, 2) + "\n");
  await mkdir(ARCHIVE_DIR, { recursive: true });
  await Bun.write(join(ARCHIVE_DIR, `${today}.json`), JSON.stringify(payload, null, 2) + "\n");

  for (const c of cands) seen[c.id!] = today;
  await Bun.write(SEEN_JSON, JSON.stringify(seen, null, 2) + "\n");

  log(`\n${cands.length} new candidates -> ${CANDIDATES_JSON.replace(REPO + "/", "")}`);
  for (const c of cands.slice(0, 12)) {
    log(`  [${String(c.score).padStart(3)}] ${c.source.slice(0, 18).padEnd(18)} ${c.title.slice(0, 70)}`);
  }
}

main();
