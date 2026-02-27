import PlotWeightHistogram from '@/components/plot-weight-histogram'

export default function TabWeights() {
  return (
    <div className="flex h-full w-full flex-col gap-2 rounded-lg bg-card text-card-foreground">
      <div className="grid h-full min-h-0 grid-cols-4 gap-2">
        <div className="min-h-0">
          <PlotWeightHistogram title="EE" mode="EE" />
        </div>
        <div className="min-h-0">
          <PlotWeightHistogram title="EI" mode="EI" />
        </div>
        <div className="min-h-0">
          <PlotWeightHistogram title="IE" mode="IE" />
        </div>
        <div className="min-h-0">
          <PlotWeightHistogram title="II" mode="II" />
        </div>
      </div>
    </div>
  )
}
