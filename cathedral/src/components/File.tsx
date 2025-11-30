import { useState, useEffect } from "react";
import { useFileContent } from "../../plugins/cathedral-plugin/src/client";
import { FileEntry } from "../../plugins/cathedral-plugin/src/lib";
import { loadMDXContent, MDXModule } from "@/lib/mdx-content";
import { MDXProviderWrapper } from "./MDXProvider";
import { CodeEditor } from "./CodeEditor";
import { Spinner } from "./ui/spinner";

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
      <div className="p-4 text-sm text-muted-foreground flex items-center gap-2">
        <Spinner />
        Loading content...
      </div>
    )
  }

  if (error) {
    return (
      <div className="text-destructive p-4 border border-destructive/30 rounded-lg bg-destructive/5">
        <h3 className="font-semibold">Error Loading MDX</h3>
        <pre className="text-sm mt-2 overflow-x-auto font-mono">{error}</pre>
      </div>
    )
  }

  if (!mdxModule) {
    return <div className="text-muted-foreground p-4">No content available.</div>
  }

  const MDXContent = mdxModule.default

  return (
    <article className="flex gap-8 p-4">
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
  return <img src={url} alt={file.name} />;
}

function OtherFile({ file, content, blob }: { file: FileEntry; content: string | null; blob: Blob | null }) {
  const fileType = file.name.split('.').pop();

  switch (fileType) {
    case 'svg':
    case 'png':
      if (blob) {
        return <ImageFile file={file} blob={blob} />;
      }
      return <p>Image viewer not implemented yet.</p>;

    case 'py':
    case 'yaml':
    case 'json':
    case 'ts':
    case 'tsx':
    case 'js':
    case 'jsx':
    case 'css':
    case 'html':
      return <CodeEditor code={content!} fileType={fileType} />;

    default:
      return <div className="p-4 text-sm text-muted-foreground">No viewer available for this file type.</div>;
  }
}

export default function File({ file }: { file: FileEntry }) {
  const isMDX = file.name.endsWith('.mdx') || file.name.endsWith('.md')

  if (isMDX) {
    return (
      <div className="print:border-none rounded">
        <MDXFile file={file} />
      </div>
    )
  }

  return <NonMDXFile file={file} />
}

function NonMDXFile({ file }: { file: FileEntry }) {
  const { content, blob, loading, error } = useFileContent(file.path);

  return (
    <div className="print:border-none rounded">
      {!loading && !error && (
        <OtherFile file={file} content={content} blob={blob} />
      )}

      {loading && (
        <div className="p-4 text-sm text-muted-foreground flex items-center gap-2">
          <Spinner />
          Loading file content...
        </div>
      )}

      {error && (
        <div className="p-4 text-sm text-red-600">Error loading file content: {error.message}</div>
      )}
    </div>
  )
}
