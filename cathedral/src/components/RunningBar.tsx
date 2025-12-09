import { isSimulationRunning } from "../../plugins/cathedral-plugin/src/client";


export function RunningBar() {
  const isRunning = isSimulationRunning();

  return (
    <>
      {isRunning && (
        // this should stay red not another color
        <div className="sticky top-0 z-50 px-[var(--page-padding)] py-2 bg-red-500 text-primary-foreground font-mono text-xs text-center tracking-wide">
          <span className="inline-flex items-center gap-3">
            <span className="h-1.5 w-1.5 rounded-full bg-current animate-pulse" />
            <span className="uppercase tracking-widest">simulation running</span>
            <span className="text-primary-foreground/60">Page will auto-refresh on completion</span>
          </span>
        </div>
      )}
    </>
  )
}