
import { useParams } from "react-router-dom"
import { isSimulationRunning, useFileContent } from "../../plugins/cathedral-plugin/src/client";
import { Slides as RenderSlides } from "@/components/files/Slides";
import Loading from "@/components/Loading";


export function Slides() {
  const { "path": path = "." } = useParams();
  const filePath = `${path}/SLIDES.mdx`

  const isRunning = isSimulationRunning();

  const { content, loading, error } = useFileContent(filePath);
  if (loading) {
    return (
      <Loading />
    )
  }

  if (error) {
    return (
      <main className="min-h-screen bg-background container mx-auto max-w-4xl py-12">
        <p className="text-center text-red-600">{error}</p>
      </main>
    )
  }
  
  return (
    <>
      <title>{`Pinglab ${path}`}</title>
      {isRunning && (
        // this should stay red not another color
        <div className="sticky top-0 z-50 px-[var(--page-padding)] py-2 bg-red-500 text-primary-foreground font-mono text-xs text-center tracking-wide print:hidden">
          <span className="inline-flex items-center gap-3">
            <span className="h-1.5 w-1.5 rounded-full bg-current animate-pulse" />
            <span className="uppercase tracking-widest">simulation running</span>
            <span className="text-primary-foreground/60">Page will auto-refresh on completion</span>
          </span>
        </div>
      )}
      {content && <RenderSlides content={content} />}
    </>
  )
}
