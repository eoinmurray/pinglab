
import { useParams } from "react-router-dom";
import { findSlides, isSimulationRunning, useDirectory } from "../../plugins/cathedral-plugin/src/client";
import Loading from "@/components/Loading";
import { MDXFile } from "@/components/MDXFile";
import { FileEntry } from "plugins/cathedral-plugin/src/lib";
import { FrontMatter } from "@/components/FrontMatter";
import { RunningBar } from "@/components/RunningBar";
import { Header } from "@/components/Header";


export function Post() {
  const { "path": path = "." } = useParams();
  const filePath = `${path}/README.mdx`
  const { directory, file, loading, error } = useDirectory(filePath)
  const isRunning = isSimulationRunning();

  let slides: FileEntry | null = null;
  if (directory) {
    slides = findSlides(directory);
  }

  if (loading) return <Loading />

  if (error) {
    return (
      <main className="min-h-screen bg-background container mx-auto max-w-4xl py-12">
        <p className="text-center text-red-600">{error.message}</p>
      </main>
    )
  }

  return (
    <div className="flex min-h-screen flex-col bg-background noise-overlay">
      <title>{file?.frontmatter?.title}</title>
      <RunningBar />
      <Header />
      <main className="flex-1 mx-auto w-full max-w-[var(--content-width)] px-[var(--page-padding)]">
        
        {isRunning && (
          <div className="sticky top-0 z-50 px-[var(--page-padding)] py-2 bg-red-500 text-primary-foreground font-mono text-xs text-center tracking-wide">
            <span className="inline-flex items-center gap-3">
              <span className="h-1.5 w-1.5 rounded-full bg-current animate-pulse" />
              <span className="uppercase tracking-widest">simulation running</span>
              <span className="text-primary-foreground/60">Page will auto-refresh on completion</span>
            </span>
          </div>
        )}

        {file && (
          <article className="mt-24 prose dark:prose-invert prose-headings:tracking-tight prose-p:leading-relaxed prose-a:text-primary prose-a:no-underline hover:prose-a:underline max-w-[var(--prose-width)] animate-fade-in">
            <FrontMatter
              title={file.frontmatter?.title}
              date={file.frontmatter?.date}
              description={file.frontmatter?.description}
              slides={slides}
            />
            <MDXFile file={file} directory={directory} />
          </article>
        )}
      </main>
    </div>    
  )
}
