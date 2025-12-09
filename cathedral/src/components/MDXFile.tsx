
import { useState, useEffect } from "react";
import { DirectoryEntry, FileEntry } from "../../plugins/cathedral-plugin/src/lib";
import { loadMDXContent, MDXModule } from "@/lib/mdx-content";
import { MDXProviderWrapper } from "./MDXProvider";
import { Spinner } from "./ui/spinner";


export function MDXFile({ file }: { file: FileEntry; directory?: DirectoryEntry | null }) {
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
    <MDXProviderWrapper>
      <MDXContent />
    </MDXProviderWrapper>
  )
}
