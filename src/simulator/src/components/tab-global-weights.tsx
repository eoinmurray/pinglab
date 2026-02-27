import PlotWeightMatrix from '@/components/plot-weight-matrix'
import { PlayCircle } from 'lucide-react'
import { Skeleton } from '@/components/ui/skeleton'
import { useSimulationContext } from '@/context/simulation-context'

export default function TabGlobalWeights() {
  const { runData } = useSimulationContext()

  if (!runData) {
    return (
      <div className="relative flex h-full w-full min-h-0 flex-col gap-3 rounded-md border bg-background p-4">
        <Skeleton className="h-4 w-40" />
        <Skeleton className="h-full min-h-[180px] w-full" />
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center px-8">
          <div className="flex items-center gap-2 text-center text-sm text-muted-foreground">
            <PlayCircle className="size-4" />
            <span>Run Simulation to view plots</span>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full w-full min-h-0">
      <PlotWeightMatrix />
    </div>
  )
}
