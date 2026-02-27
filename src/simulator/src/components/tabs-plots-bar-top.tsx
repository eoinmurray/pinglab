import { useEffect, useMemo, useRef, useState } from 'react'
import { useSimulationContext } from '@/context/simulation-context'
import { Button } from '@/components/ui/button'
import { Kbd, KbdGroup } from '@/components/ui/kbd'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { CircleHelp, Play } from 'lucide-react'
import { buildRunRequest } from '@/lib/api/payload'
import { cn } from '@/lib/utils'

function formatMaybeNumber(value: unknown, digits: number): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return '--'
  }
  return value.toFixed(digits)
}

type MetricBadgeProps = {
  label: string
  value: string
  description: string
}

function MetricBadge({ label, value, description }: MetricBadgeProps) {
  return (
    <div className="inline-flex items-center gap-1 rounded-md border bg-background px-2 py-1 text-[11px] text-muted-foreground">
      <span className="font-medium text-foreground">{label}</span>
      <span>{value}</span>
      <Popover>
        <PopoverTrigger asChild>
          <button
            type="button"
            className="inline-flex items-center justify-center rounded-sm text-muted-foreground/80 transition-colors hover:text-foreground"
            aria-label={`${label} info`}
          >
            <CircleHelp className="size-3" />
          </button>
        </PopoverTrigger>
        <PopoverContent className="w-64 p-3 text-xs" align="end" side="bottom">
          <p className="leading-relaxed text-muted-foreground">{description}</p>
        </PopoverContent>
      </Popover>
    </div>
  )
}

export default function TabsPlotsBarTop() {
  const { parameters, runData, runLoading, runValidationError, runSimulationNow } =
    useSimulationContext()
  const [lastSuccessfulRunSignature, setLastSuccessfulRunSignature] = useState<string | null>(
    null
  )
  const pendingRunSignatureRef = useRef<string | null>(null)
  const previousRunLoadingRef = useRef(runLoading)
  const currentRunSignature = useMemo(
    () => JSON.stringify(buildRunRequest(parameters)),
    [parameters]
  )

  useEffect(() => {
    if (!previousRunLoadingRef.current && runLoading) {
      pendingRunSignatureRef.current = currentRunSignature
    }
    previousRunLoadingRef.current = runLoading
  }, [currentRunSignature, runLoading])

  useEffect(() => {
    if (!runData || !pendingRunSignatureRef.current) {
      return
    }
    setLastSuccessfulRunSignature(pendingRunSignatureRef.current)
    pendingRunSignatureRef.current = null
  }, [runData])

  const isRunStale =
    !lastSuccessfulRunSignature || lastSuccessfulRunSignature !== currentRunSignature
  const spikeTotals = useMemo(() => {
    if (!runData) {
      return { e: 0, i: 0 }
    }
    if (
      Number.isFinite(runData.total_e_spikes) &&
      Number.isFinite(runData.total_i_spikes)
    ) {
      return {
        e: Math.max(0, Math.round(runData.total_e_spikes)),
        i: Math.max(0, Math.round(runData.total_i_spikes)),
      }
    }
    const types = runData.spikes?.types ?? []
    if (types.length > 0) {
      let e = 0
      let i = 0
      for (const t of types) {
        if (t === 0) e += 1
        else if (t === 1) i += 1
      }
      return { e, i }
    }
    const ids = runData.spikes?.ids ?? []
    const nE = Math.max(0, Math.round(parameters.config.N_E.value))
    let e = 0
    let i = 0
    for (const id of ids) {
      if (id < nE) e += 1
      else i += 1
    }
    return { e, i }
  }, [parameters.config.N_E.value, runData])

  return (
    <div
      className="flex w-full items-center justify-start gap-3"
      data-testid="tabs-plots-bar-top"
    >
      <Button
        size="sm"
        onClick={runSimulationNow}
        disabled={runLoading || Boolean(runValidationError)}
        title={runValidationError ?? undefined}
        data-testid="run-simulation-button"
      >
        <Play className="size-4" />
        Run
        <KbdGroup aria-label="Run shortcut" className="gap-0.5">
          <Kbd className="h-4 min-w-8 border-primary-foreground/25 bg-transparent px-1 text-primary-foreground/90">
            Space
          </Kbd>
        </KbdGroup>
      </Button>
      <div
        className={cn(
          'text-xs font-medium',
          runValidationError
            ? 'text-red-600 dark:text-red-500'
            : isRunStale
            ? 'text-amber-600 dark:text-amber-500'
            : 'text-emerald-600 dark:text-emerald-500'
        )}
      >
        {runValidationError
          ? 'Invalid config'
          : runLoading
            ? 'Running...'
            : isRunStale
              ? 'Stale config'
              : 'Up to date'}
      </div>
      {runData ? (
        <div className="ml-auto flex flex-wrap items-center justify-end gap-2">
          <MetricBadge
            label="Total E spikes"
            value={String(spikeTotals.e)}
            description="Total spikes emitted by excitatory neurons in this run."
          />
          <MetricBadge
            label="Total I spikes"
            value={String(spikeTotals.i)}
            description="Total spikes emitted by inhibitory neurons in this run."
          />
          <MetricBadge
            label="PSD peak"
            value={`${formatMaybeNumber(runData.psd_peak_freq_hz, 2)} Hz`}
            description="Frequency where the E-population PSD is maximal (20-120 Hz window)."
          />
          <MetricBadge
            label="Q-factor"
            value={formatMaybeNumber(runData.psd_peak_q_factor, 2)}
            description="Oscillation narrowness: Q = peak frequency / half-power bandwidth. Rough guide: Q < 2 is broad/noisy (AI-like), Q ~2-5 is moderate rhythm, Q > 5 is narrow and strongly oscillatory."
          />
          <MetricBadge
            label="BW50"
            value={`${formatMaybeNumber(runData.psd_peak_bandwidth_hz, 2)} Hz`}
            description="Half-power bandwidth around the PSD peak. Smaller BW50 means a narrower, cleaner oscillation."
          />
        </div>
      ) : null}
    </div>
  )
}
