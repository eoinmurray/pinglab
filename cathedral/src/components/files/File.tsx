
import { DirectoryEntry, FileEntry } from "../../../plugins/cathedral-plugin/src/lib";
import { MDXFile } from "./MDXFile";
import { NonMDXFile } from "./NonMDXFile";

export default function File({ file, directory }: { file: FileEntry; directory?: DirectoryEntry | null }) {
  const isMDX = file.name.endsWith('.mdx') || file.name.endsWith('.md')

  if (isMDX) {
    return (
      <div className="print:border-none">
        <MDXFile file={file} directory={directory} />
      </div>
    )
  }

  return <NonMDXFile file={file} />
}

