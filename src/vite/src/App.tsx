import { useEffect, useState } from 'react'

interface PdfEntry {
  file: string
  title: string
  description: string
  date: string
}

function extractNumber(entry: PdfEntry): string {
  const m = (entry.title.match(/\.(\d+)-/) || entry.file.match(/\.(\d+)-/))
  return m ? m[1] : ''
}

function formatTitle(title: string): string {
  return title.replace(/^(study|report)\.\d+-?/, '').replace(/-/g, ' ')
}

function App() {
  const [pdfs, setPdfs] = useState<PdfEntry[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/manifest.json')
      .then((res) => {
        if (!res.ok) throw new Error('manifest.json not found — run: task typst:publish')
        return res.json()
      })
      .then(setPdfs)
      .catch((err) => setError(err.message))
  }, [])

  const byDateDesc = (a: PdfEntry, b: PdfEntry) => b.date.localeCompare(a.date)
  const reports = pdfs.filter((p) => p.file.startsWith('report.')).sort(byDateDesc)
  const studies = pdfs.filter((p) => p.file.startsWith('study.')).sort(byDateDesc)

  return (
    <div className="max-w-lg mx-auto px-5 py-14 sm:py-20">
      <header className="mb-12">
        <h1 className="text-sm font-normal text-[var(--color-text-primary)] mb-1">
          Pinglab
        </h1>
        <p className="text-xs text-[var(--color-text-secondary)]">
          All works in progress.
        </p>
      </header>

      {error && (
        <p className="text-xs text-[var(--color-text-secondary)] mb-8">
          {error}
        </p>
      )}

      {reports.length > 0 && (
        <Section label="reports" entries={reports} />
      )}

      {studies.length > 0 && (
        <Section label="studies" entries={studies} />
      )}
    </div>
  )
}

function Section({ label, entries }: { label: string; entries: PdfEntry[] }) {
  return (
    <section className="mb-10">
      <h2 className="text-[10px] text-[var(--color-text-tertiary)] uppercase tracking-[0.15em] mb-3">
        {label}
      </h2>
      <ul>
        {entries.map((entry) => {
          const num = extractNumber(entry)
          return (
            <li key={entry.file} className="border-t border-[var(--color-border-dim)]">
              <a
                href={`/${entry.file}`}
                className="group flex items-baseline gap-3 py-2 text-[12px] hover:text-[var(--color-accent)] transition-colors"
              >
                <span className="text-[var(--color-text-tertiary)] tabular-nums w-4 shrink-0 text-right">
                  {num}
                </span>
                <span className="text-[var(--color-text-primary)] group-hover:text-[var(--color-accent)] transition-colors truncate">
                  {formatTitle(entry.title)}
                </span>
                <span className="text-[var(--color-text-tertiary)] tabular-nums ml-auto shrink-0">
                  {entry.date}
                </span>
              </a>
            </li>
          )
        })}
      </ul>
    </section>
  )
}

export default App
