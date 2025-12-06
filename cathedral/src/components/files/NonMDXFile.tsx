
import { useFileContent } from "../../../plugins/cathedral-plugin/src/client";
import { FileEntry } from "../../../plugins/cathedral-plugin/src/lib";
import { Spinner } from "../ui/spinner";
import { OtherFile } from "./OtherFile";


export function NonMDXFile({ file }: { file: FileEntry }) {
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
