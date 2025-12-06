
import { useState, useEffect } from "react";
import { findSlides } from "../../../plugins/cathedral-plugin/src/client";
import { DirectoryEntry, FileEntry } from "../../../plugins/cathedral-plugin/src/lib";
import { loadMDXContent, MDXModule } from "@/lib/mdx-content";
import { MDXProviderWrapper } from "./MDXProvider";
import { Spinner } from "../ui/spinner";
import { formatDate } from "@/lib/format-date";
import { Link } from "react-router-dom";
import { Presentation } from "lucide-react";


export function MDXFile({ file, directory }: { file: FileEntry; directory?: DirectoryEntry | null }) {
  const [mdxModule, setMdxModule] = useState<MDXModule | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  let slides: FileEntry | null = null;
  if (directory) {
    slides = findSlides(directory);
  }

  useEffect(() => {
    setLoading(true)
    setError(null)

    loadMDXContent(file.path)
      .then(mod => {
        if (mod) {
          setMdxModule(mod)
        } else {
          setError(`MDX file not found in build: ${file.path}`)
        }
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [file.path])

  if (loading) {
    return (
      <div className="py-12 text-muted-foreground flex items-center gap-3 font-mono text-sm">
        <Spinner />
        <span className="tracking-wide">loading...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="text-destructive p-6 border border-destructive/20 bg-destructive/5">
        <h3 className="font-mono text-sm tracking-wide mb-2">error loading mdx</h3>
        <pre className="text-xs overflow-x-auto font-mono opacity-70">{error}</pre>
      </div>
    )
  }

  if (!mdxModule) {
    return <div className="text-muted-foreground py-12 font-mono text-sm">no content available</div>
  }

  const MDXContent = mdxModule.default

  return (
    <article className="prose dark:prose-invert prose-headings:tracking-tight prose-p:leading-relaxed prose-a:text-primary prose-a:no-underline hover:prose-a:underline max-w-[var(--prose-width)] animate-fade-in">
      {/* Title */}
      {file.frontmatter?.title && (
        <header className="not-prose flex flex-col gap-2 mb-8 pt-4">
          <h1 className="text-2xl md:text-3xl font-semibold tracking-tight text-foreground mb-3">
            {file.frontmatter.title}
          </h1>

          {/* Meta line */}
          <div className="flex flex-wrap items-center gap-3 text-muted-foreground">
            {file.frontmatter?.date && (
              <time className="font-mono text-xs bg-muted px-2 py-0.5 rounded">
                {formatDate(new Date(file.frontmatter.date as string))}
              </time>
            )}
            {slides && (
              <Link
                to={`/${slides.path}`}
                className="font-mono text-xs px-2 py-0.5 rounded flex items-center gap-1 print:hidden"
              >
                <Presentation className="h-3.5 w-3.5" />
                <span>slides</span>
              </Link>
            )}
          </div>

          {file.frontmatter?.description && (
            <div className="flex flex-wrap text-sm items-center gap-3 text-muted-foreground">
              {file.frontmatter.description}
            </div>
          )}
        </header>
      )}

      <MDXProviderWrapper>
        <MDXContent />
      </MDXProviderWrapper>
    </article>
  )
}
