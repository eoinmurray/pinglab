
import { useState, useEffect } from "react";
import { FileEntry } from "../../../plugins/cathedral-plugin/src/lib";


export function ImageFile({ file, blob }: { file: FileEntry; blob: Blob }) {
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
