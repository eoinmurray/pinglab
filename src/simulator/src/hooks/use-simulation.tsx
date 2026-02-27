import { useEffect, useRef, useState } from 'react'
import { CircleCheckBig } from 'lucide-react'
import { Spinner } from '@/components/ui/spinner'
import type { ParametersState } from '@/hooks/use-parameters'
import { runSimulation } from '@/lib/api/client'
import { buildRunRequest } from '@/lib/api/payload'
import { getApiTargetInfo } from '@/lib/api/target'
import type { RunResponse, RunTimingHeaders } from '@/lib/api/types'
import { toast } from 'sonner'

type UseSimulationResult = {
  data: RunResponse | null
  loading: boolean
  error: string | null
  timings: RunTimingHeaders | null
}

export function useSimulation(
  parameters: ParametersState,
  runTrigger: number,
  useArrowTransport: boolean
): UseSimulationResult {
  const [data, setData] = useState<RunResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [timings, setTimings] = useState<RunTimingHeaders | null>(null)
  const requestControllerRef = useRef<AbortController | null>(null)
  const latestParametersRef = useRef(parameters)
  const lastHandledRunTriggerRef = useRef(0)

  useEffect(() => {
    latestParametersRef.current = parameters
  }, [parameters])

  useEffect(() => {
    if (runTrigger <= 0 || runTrigger === lastHandledRunTriggerRef.current) {
      return
    }
    lastHandledRunTriggerRef.current = runTrigger

    const controller = new AbortController()
    requestControllerRef.current?.abort()
    requestControllerRef.current = controller
    const payload = buildRunRequest(latestParametersRef.current)

    setLoading(true)
    setError(null)
    const targetLabel = getApiTargetInfo().target
    toast(
      targetLabel === 'local'
        ? 'Simulation running on Localhost'
        : 'Simulation running on Modal',
      {
      id: 'simulation-status',
      duration: Infinity,
      icon: <Spinner className="size-4" />,
      }
    )

    runSimulation(payload, controller.signal, useArrowTransport ? 'arrow' : 'json')
      .then(({ data: nextData, timings: nextTimings }) => {
        if (requestControllerRef.current !== controller) {
          return
        }
        toast.dismiss('simulation-running-warning')
        setData(nextData)
        setTimings(nextTimings)
        toast('Simulation Finished', {
          id: 'simulation-status',
          duration: 500,
          icon: <CircleCheckBig className="size-4 text-emerald-500" />,
        })
      })
      .catch((err: unknown) => {
        if (requestControllerRef.current !== controller) {
          return
        }
        if ((err as Error).name === 'AbortError') {
          return
        }
        toast.dismiss('simulation-running-warning')
        setError(
          err instanceof Error ? err.message : 'Simulation request failed'
        )
        toast('Simulation Failed', {
          id: 'simulation-status',
        })
      })
      .finally(() => {
        if (requestControllerRef.current === controller) {
          requestControllerRef.current = null
          setLoading(false)
        }
      })

    return () => {
      requestControllerRef.current?.abort()
    }
  }, [runTrigger, useArrowTransport])

  return { data, loading, error, timings }
}
