
import { FileEntry } from "../../../plugins/cathedral-plugin/src/lib";
import { CodeEditor } from "./CodeEditor";
import { ImageFile } from "./ImageFile";


export function OtherFile({ file, content, blob }: { file: FileEntry; content: string | null; blob: Blob | null }) {
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
