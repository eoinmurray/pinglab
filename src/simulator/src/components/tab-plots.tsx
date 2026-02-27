import TabsPlotsBarTop from '@/components/tabs-plots-bar-top'
import PlotRaster from '@/components/plot-raster'
import PlotPsd from '@/components/plot-psd'
import PlotAutocorrelation from '@/components/plot-autocorrelation'
import PlotCrossCorrelation from '@/components/plot-cross-correlation'
import PlotInputTrace from '@/components/plot-input-trace'
import PlotInputSpikes from '@/components/plot-input-spikes'
import PlotDecodePath from '@/components/plot-decode-path'
import PlotVoltageTrace from '@/components/plot-voltage-trace'
import PlotPopulationRateTrace from '@/components/plot-population-rate-trace'
import { PlayCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ButtonGroup } from '@/components/ui/button-group'
import { Skeleton } from '@/components/ui/skeleton'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useEffect, useMemo } from 'react'
import { useSimulationContext } from '@/context/simulation-context'

function PlotSkeleton() {
  return (
    <div className="flex h-full min-h-0 flex-col gap-3 rounded-md border bg-background p-4">
      <Skeleton className="h-4 w-40" />
      <Skeleton className="h-4 w-24" />
      <Skeleton className="h-full min-h-[180px] w-full" />
    </div>
  )
}

function RasterSkeleton() {
  return (
    <div className="relative flex h-full min-h-0 flex-col gap-3 rounded-md border bg-background p-4">
      <Skeleton className="h-4 w-40" />
      <Skeleton className="h-full min-h-[180px] w-full" />
      <div className="pointer-events-none absolute inset-0 flex items-center justify-center px-8">
        <div className="flex items-center gap-2 text-center text-sm text-muted-foreground">
          <PlayCircle className="size-4" />
          <span>Run Simulation to view raster data</span>
        </div>
      </div>
    </div>
  )
}

export default function TabPlots() {
  const {
    runData,
    parameters,
    plotsSubTab,
    setPlotsSubTab,
    plotsLayer,
    setPlotsLayer,
  } = useSimulationContext()
  const fallbackLabels = useMemo(() => {
    const labels = ['L1']
    if (Math.round(parameters.config.N_I.value) > 0) labels.push('I')
    return labels
  }, [parameters.config.N_I.value])
  const correlationLayerLabels =
    runData?.layer_labels?.length ? runData.layer_labels : fallbackLabels
  const activeCorrelationLayer = correlationLayerLabels.includes(plotsLayer)
    ? plotsLayer
    : (correlationLayerLabels[0] ?? 'L1')
  const hasRunData = runData !== null
  const isPerformanceMode = parameters.performanceMode.value
  const showInputSpikesTab = parameters.graph.nodes
    .filter((node) => node.kind === 'input')
    .some(
      (node) =>
        (parameters.graph.inputPrograms[node.id]?.mode ??
          (node.type as string)) === 'external_spike_train'
    )
  const showDecodePathTab = showInputSpikesTab
  const showInputsTab = !showInputSpikesTab

  useEffect(() => {
    if (!showInputSpikesTab && plotsSubTab === 'input-spikes') {
      setPlotsSubTab('inputs')
    }
  }, [showInputSpikesTab, plotsSubTab, setPlotsSubTab])

  useEffect(() => {
    if (!showDecodePathTab && plotsSubTab === 'decode-path') {
      setPlotsSubTab('population-rate')
    }
  }, [showDecodePathTab, plotsSubTab, setPlotsSubTab])

  useEffect(() => {
    if (!showInputsTab && plotsSubTab === 'inputs') {
      setPlotsSubTab('input-spikes')
    }
  }, [showInputsTab, plotsSubTab, setPlotsSubTab])

  return (
    <div className="flex h-full w-full flex-col gap-2 rounded-lg text-card-foreground">
      <TabsPlotsBarTop />
      <div className={`grid min-h-0 flex-1 gap-2 ${isPerformanceMode ? 'grid-rows-1' : 'grid-rows-2'}`}>
        <div className="min-h-0">
          {hasRunData ? <PlotRaster /> : <RasterSkeleton />}
        </div>
        {isPerformanceMode ? null : (
          <div className="flex min-h-0 flex-col gap-2">
          <Tabs
            value={plotsSubTab}
            onValueChange={(value) =>
              setPlotsSubTab(
                value as
                  | 'inputs'
                  | 'input-spikes'
                  | 'decode-path'
                  | 'voltage'
                  | 'population-rate'
                  | 'psd'
                  | 'correlations'
              )
            }
            className="flex min-h-0 flex-1 flex-col"
          >
            <div className="flex items-center gap-2">
              <TabsList className="w-fit">
                {showInputsTab ? <TabsTrigger value="inputs">Inputs</TabsTrigger> : null}
                {showInputSpikesTab ? (
                  <TabsTrigger value="input-spikes">Input Spikes</TabsTrigger>
                ) : null}
                {showDecodePathTab ? (
                  <TabsTrigger value="decode-path">Decode Path</TabsTrigger>
                ) : null}
                <TabsTrigger value="voltage">Voltages</TabsTrigger>
                <TabsTrigger value="population-rate">Rates</TabsTrigger>
                {/* <TabsTrigger value="weights">Weights</TabsTrigger> */}
                <TabsTrigger value="psd">PSD</TabsTrigger>
                <TabsTrigger value="correlations">Correlations</TabsTrigger>
              </TabsList>
              <div className="max-w-[300px] overflow-x-auto">
                <ButtonGroup aria-label="Layer selection" className="gap-1">
                  {correlationLayerLabels.map((label) => (
                    <Button
                      key={label}
                      type="button"
                      size="xs"
                      variant={activeCorrelationLayer === label ? 'default' : 'outline'}
                      onClick={() => setPlotsLayer(label)}
                    >
                      {label}
                    </Button>
                  ))}
                </ButtonGroup>
              </div>
            </div>
            {showInputsTab ? (
              <TabsContent value="inputs" className="min-h-0 flex-1">
                {hasRunData ? <PlotInputTrace layerLabel={activeCorrelationLayer} /> : <PlotSkeleton />}
              </TabsContent>
            ) : null}
            <TabsContent value="input-spikes" className="min-h-0 flex-1">
              {hasRunData ? <PlotInputSpikes layerLabel={activeCorrelationLayer} /> : <PlotSkeleton />}
            </TabsContent>
            <TabsContent value="decode-path" className="min-h-0 flex-1">
              {hasRunData ? <PlotDecodePath layerLabel={activeCorrelationLayer} /> : <PlotSkeleton />}
            </TabsContent>
            <TabsContent value="psd" className="min-h-0 flex-1">
              {hasRunData ? <PlotPsd layerLabel={activeCorrelationLayer} /> : <PlotSkeleton />}
            </TabsContent>
            <TabsContent value="voltage" className="min-h-0 flex-1">
              {hasRunData ? <PlotVoltageTrace layerLabel={activeCorrelationLayer} /> : <PlotSkeleton />}
            </TabsContent>
            <TabsContent value="population-rate" className="min-h-0 flex-1">
              {hasRunData ? (
                <PlotPopulationRateTrace layerLabel={activeCorrelationLayer} />
              ) : (
                <PlotSkeleton />
              )}
            </TabsContent>
            {/* <TabsContent value="weights" className="min-h-0 flex-1">
              <TabWeights />
            </TabsContent> */}
            <TabsContent value="correlations" className="min-h-0 flex-1">
              <div className="flex h-full min-h-0 flex-col gap-2">
                <div className="grid h-full min-h-0 grid-cols-2 gap-2">
                  <div className="min-h-0">
                    {hasRunData ? (
                      <PlotAutocorrelation layerLabel={activeCorrelationLayer} />
                    ) : (
                      <PlotSkeleton />
                    )}
                  </div>
                  <div className="min-h-0">
                    {hasRunData ? (
                      <PlotCrossCorrelation layerLabel={activeCorrelationLayer} />
                    ) : (
                      <PlotSkeleton />
                    )}
                  </div>
                </div>
              </div>
            </TabsContent>
          </Tabs>
          </div>
        )}
      </div>
    </div>
  )
}
