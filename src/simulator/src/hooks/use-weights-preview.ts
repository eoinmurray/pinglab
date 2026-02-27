import { useEffect, useRef, useState } from 'react'
import type { ParametersState } from '@/hooks/use-parameters'
import { fetchWeightsPreview } from '@/lib/api/client'
import { buildWeightsRequest } from '@/lib/api/payload'
import type { WeightsResponse } from '@/lib/api/types'

type UseWeightsPreviewResult = {
  data: WeightsResponse | null
  loading: boolean
  error: string | null
}

export function useWeightsPreview(
  parameters: ParametersState,
  enabled: boolean,
  useArrowTransport: boolean
): UseWeightsPreviewResult {
  const [data, setData] = useState<WeightsResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const requestControllerRef = useRef<AbortController | null>(null)

  useEffect(() => {
    if (!enabled) {
      requestControllerRef.current?.abort()
      setLoading(false)
      return
    }

    const debounceMs = 200
    const timeoutId = window.setTimeout(() => {
      const controller = new AbortController()
      requestControllerRef.current?.abort()
      requestControllerRef.current = controller

      setLoading(true)
      setError(null)

      fetchWeightsPreview(
        buildWeightsRequest(parameters),
        controller.signal,
        useArrowTransport ? 'arrow' : 'json'
      )
        .then((nextData) => {
          if (requestControllerRef.current !== controller) {
            return
          }
          setData(nextData)
        })
        .catch((err: unknown) => {
          if (requestControllerRef.current !== controller) {
            return
          }
          if ((err as Error).name === 'AbortError') {
            return
          }
          setError(
            err instanceof Error ? err.message : 'Weights request failed'
          )
        })
        .finally(() => {
          if (requestControllerRef.current === controller) {
            requestControllerRef.current = null
            setLoading(false)
          }
        })
    }, debounceMs)

    return () => {
      window.clearTimeout(timeoutId)
      requestControllerRef.current?.abort()
    }
  }, [enabled, parameters, useArrowTransport])

  return { data, loading, error }
}
