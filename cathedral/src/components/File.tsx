import { fetchFileContent } from "../../plugins/cathedral-plugin/src/client";
import { FileEntry } from "../../plugins/cathedral-plugin/src/lib";
import { MDXRenderer } from "./MDXRenderer";
import { CodeEditor } from "./CodeEditor";
import { Slides } from "./Slides";
import { Spinner } from "./ui/spinner";


function RenderFile({ file, content, blob }: { file: FileEntry; content: string | null; blob: unknown | null; }) {
  const fileType = file.name.split('.').pop();

  if (file.name === "SLIDES.md" || file.name === "slides.md" || file.name === "Slides.md" ) {
    return (
      <Slides content={content!} />
    )
  }

  switch (fileType) {
    case 'svg':
    case 'png':
      if (blob instanceof Blob) {
        const url = URL.createObjectURL(blob);
        return <img src={url} alt="" />;
      }
      return <p>Image viewer not implemented yet.</p>;

    case 'md':
    case 'mdx':
      if (!content) {
        return <p>No content available.</p>;
      }
      return (
        <article className="flex gap-8 p-4">
          <MDXRenderer content={content} />
        </article>
      )

    case 'py':
    case 'yaml':
    case 'json':
      return <CodeEditor code={content!} fileType={fileType} />;

    default:
      return <div className="p-4 text-sm text-muted-foreground">No viewer available for this file type.</div>;
  }
}

export default function File({ file }: { file: FileEntry }) {
  const { content, blob, loading, error } = fetchFileContent(file.path);

  return (
    <div className="border print:border-none rounded">
      <div className="print:hidden border-b px-4 py-3 bg-sidebar rounded-t">
        <span className="text-sm text-muted-foreground">{file.name}</span>
      </div>
      {!loading && !error && (
        <RenderFile file={file} content={content} blob={blob} />
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