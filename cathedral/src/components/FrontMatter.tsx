import { formatDate } from "@/lib/format-date"
import { Presentation } from "lucide-react"
import { FileEntry } from "plugins/cathedral-plugin/src/lib"
import { Link } from "react-router-dom"

export function FrontMatter({
  title,
  date,
  description,
  slides,
}: {
  title?: string
  date?: string
  description?: string
  slides?: FileEntry | null
}){

  return (
    <div>
      {title && (
        <header className="not-prose flex flex-col gap-2 mb-8 pt-4">
          <h1 className="text-2xl md:text-3xl font-semibold tracking-tight text-foreground mb-3">
            {title}
          </h1>

          {/* Meta line */}
          <div className="flex flex-wrap items-center gap-3 text-muted-foreground">
            {date && (
              <time className="font-mono text-xs bg-muted px-2 py-0.5 rounded">
                {formatDate(new Date(date as string))}
              </time>
            )}
            {slides && (
              <Link
                to={`/${slides.path}`}
                className="font-mono text-xs px-2 py-0.5 rounded flex items-center gap-1"
              >
                <Presentation className="h-3.5 w-3.5" />
                <span>slides</span>
              </Link>
            )}
          </div>

          {description && (
            <div className="flex flex-wrap text-sm items-center gap-3 text-muted-foreground">
              {description}
            </div>
          )}
        </header>
      )}
    </div>
  )
}