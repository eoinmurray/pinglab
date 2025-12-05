import { useState, useEffect } from "react";
import { useFileContent } from "../../plugins/cathedral-plugin/src/client";
import { FileEntry } from "../../plugins/cathedral-plugin/src/lib";
import { loadMDXContent, MDXModule } from "@/lib/mdx-content";
import { MDXProviderWrapper } from "./MDXProvider";
import { CodeEditor } from "./CodeEditor";
import { Spinner } from "./ui/spinner";
import { formatDate } from "@/lib/format-date";

function MDXFile({ file }: { file: FileEntry }) {
  const [mdxModule, setMdxModule] = useState<MDXModule | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

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
        <header className="not-prose mb-8 pt-4">
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
            {file.frontmatter?.description && (
              <p className="text-sm">
                {file.frontmatter.description}
              </p>
            )}
          </div>
        </header>
      )}

      <MDXProviderWrapper>
        <MDXContent />
      </MDXProviderWrapper>
    </article>
  )
}

function ImageFile({ file, blob }: { file: FileEntry; blob: Blob }) {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    const objectUrl = URL.createObjectURL(blob);
    setUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [blob]);

  if (!url) return null;
  return (
    <figure className="animate-fade-in">
      <img src={url} alt={file.name} className="max-w-full shadow-sm" />
      <figcaption className="mt-3 font-mono text-xs text-muted-foreground tracking-wide">
        {file.name}
      </figcaption>
    </figure>
  );
}

function OtherFile({ file, content, blob }: { file: FileEntry; content: string | null; blob: Blob | null }) {
  const fileType = file.name.split('.').pop();

  switch (fileType) {
    case 'svg':
    case 'png':
      if (blob) {
        return <ImageFile file={file} blob={blob} />;
      }
      return <p className="text-muted-foreground font-mono text-sm">image viewer not available</p>;

    case 'py':
    case 'yaml':
    case 'json':
    case 'ts':
    case 'tsx':
    case 'js':
    case 'jsx':
    case 'css':
    case 'html':
      return (
        <div className="animate-fade-in">
          <CodeEditor code={content!} fileType={fileType} />
        </div>
      );

    default:
      return (
        <div className="py-12 text-muted-foreground font-mono text-sm tracking-wide">
          no viewer for .{fileType} files
        </div>
      );
  }
}

export default function File({ file }: { file: FileEntry }) {
  const isMDX = file.name.endsWith('.mdx') || file.name.endsWith('.md')

  if (isMDX) {
    return (
      <div className="print:border-none">
        <MDXFile file={file} />
      </div>
    )
  }

  return <NonMDXFile file={file} />
}

function NonMDXFile({ file }: { file: FileEntry }) {
  const { content, blob, loading, error } = useFileContent(file.path);

  return (
    <div className="print:border-none">
      {!loading && !error && (
        <OtherFile file={file} content={content} blob={blob} />
      )}

      {loading && (
        <div className="py-12 text-muted-foreground flex items-center gap-3 font-mono text-sm">
          <Spinner />
          <span className="tracking-wide">loading file...</span>
        </div>
      )}

      {error && (
        <div className="py-8 text-destructive font-mono text-sm">
          error: {error.message}
        </div>
      )}
    </div>
  )
}
