import type {
  RunRequest,
  RunResponse,
  RunTimingHeaders,
  WeightsRequest,
  WeightsResponse,
} from '@/lib/api/types'
import { tableFromIPC } from 'apache-arrow'

const apiBase = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim() || 'http://localhost:8000'
const normalizedApiBase = apiBase.replace(/\/+$/, '')
const runUrl =
  (import.meta.env.VITE_API_RUN_URL as string | undefined)?.trim() ||
  `${normalizedApiBase}/run`
const weightsUrl =
  (import.meta.env.VITE_API_WEIGHTS_URL as string | undefined)?.trim() ||
  `${normalizedApiBase}/weights`
const arrowMediaType = 'application/vnd.apache.arrow.stream'
type ResponseFormat = 'arrow' | 'json'

function buildAcceptHeader(format: ResponseFormat) {
  return format === 'json'
    ? 'application/json'
    : `${arrowMediaType}, application/json`
}

function parseNumberHeader(value: string | null) {
  if (!value) {
    return null
  }
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function parseTimingHeaders(headers: Headers): RunTimingHeaders {
  return {
    coreSimMs: parseNumberHeader(headers.get('X-Pinglab-Core-Sim-Ms')),
    inputPrepMs: parseNumberHeader(headers.get('X-Pinglab-Input-Prep-Ms')),
    weightsBuildMs: parseNumberHeader(headers.get('X-Pinglab-Weights-Build-Ms')),
    analysisMs: parseNumberHeader(headers.get('X-Pinglab-Analysis-Ms')),
    responseBuildMs: parseNumberHeader(headers.get('X-Pinglab-Response-Build-Ms')),
    serverComputeMs: parseNumberHeader(headers.get('X-Pinglab-Server-Compute-Ms')),
    serializeMs: parseNumberHeader(headers.get('X-Pinglab-Serialize-Ms')),
    responseBytes: parseNumberHeader(headers.get('X-Pinglab-Response-Bytes')),
  }
}

function deepNormalizeArrowValue(value: unknown): unknown {
  if (typeof value === 'bigint') {
    return Number(value)
  }
  if (value && typeof value === 'object') {
    const maybeWithToArray = value as { toArray?: () => unknown[] }
    const valueTag = Object.prototype.toString.call(value)
    // Arrow vectors survive minification via Symbol.toStringTag, unlike
    // constructor names. Restrict to vectors so row objects keep field keys.
    if (
      valueTag.includes('Vector') &&
      typeof maybeWithToArray.toArray === 'function' &&
      !Array.isArray(value) &&
      !ArrayBuffer.isView(value) &&
      !(value instanceof DataView)
    ) {
      const vectorArray = maybeWithToArray.toArray()
      return Array.from(vectorArray as ArrayLike<unknown>, (entry) =>
        deepNormalizeArrowValue(entry)
      )
    }
  }
  if (value instanceof DataView) {
    return Array.from(
      new Uint8Array(value.buffer, value.byteOffset, value.byteLength),
      (entry) => Number(entry)
    )
  }
  if (ArrayBuffer.isView(value)) {
    const typed = value as unknown as { length: number; [index: number]: number }
    return Array.from({ length: typed.length }, (_, index) => Number(typed[index]))
  }
  if (Array.isArray(value)) {
    return value.map((entry) => deepNormalizeArrowValue(entry))
  }
  if (value && typeof value === 'object') {
    const out: Record<string, unknown> = {}
    for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
      out[key] = deepNormalizeArrowValue(entry)
    }
    return out
  }
  return value
}

function parseArrowSingleRow<T>(buffer: ArrayBuffer): T {
  const table = tableFromIPC(new Uint8Array(buffer))
  const rows = table.toArray()
  if (!rows.length) {
    throw new Error('Arrow response had no rows')
  }
  return deepNormalizeArrowValue(rows[0]) as T
}

export async function runSimulation(
  payload: RunRequest,
  signal?: AbortSignal,
  responseFormat: ResponseFormat = 'arrow'
): Promise<{ data: RunResponse; timings: RunTimingHeaders }> {
  const response = await fetch(runUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: buildAcceptHeader(responseFormat),
    },
    body: JSON.stringify(payload),
    signal,
  })

  if (!response.ok) {
    throw new Error(`Simulation request failed (${response.status})`)
  }

  const contentType = response.headers.get('content-type') || ''
  const data = contentType.includes(arrowMediaType)
    ? parseArrowSingleRow<RunResponse>(await response.arrayBuffer())
    : ((await response.json()) as RunResponse)
  return { data, timings: parseTimingHeaders(response.headers) }
}

export async function fetchWeightsPreview(
  payload: WeightsRequest,
  signal?: AbortSignal,
  responseFormat: ResponseFormat = 'arrow'
): Promise<WeightsResponse> {
  const response = await fetch(weightsUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: buildAcceptHeader(responseFormat),
    },
    body: JSON.stringify(payload),
    signal,
  })

  if (!response.ok) {
    throw new Error(`Weights request failed (${response.status})`)
  }

  const contentType = response.headers.get('content-type') || ''
  if (contentType.includes(arrowMediaType)) {
    return parseArrowSingleRow<WeightsResponse>(await response.arrayBuffer())
  }
  return (await response.json()) as WeightsResponse
}
