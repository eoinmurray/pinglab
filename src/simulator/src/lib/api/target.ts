type ApiTarget = 'local' | 'modal'

type ApiUrls = {
  baseUrl: string
  runUrl: string
  weightsUrl: string
}

type ApiTargetInfo = {
  target: ApiTarget
  runUrl: string
  weightsUrl: string
  hostname: string
}

const DEFAULT_BASE_URL = 'http://localhost:8000'

export function getApiUrls(): ApiUrls {
  const configuredBaseUrl = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim()
  const configuredRunUrl = (import.meta.env.VITE_API_RUN_URL as string | undefined)?.trim()
  const configuredWeightsUrl = (import.meta.env.VITE_API_WEIGHTS_URL as string | undefined)?.trim()

  if (configuredRunUrl || configuredWeightsUrl) {
    const fallbackBase = configuredBaseUrl?.replace(/\/+$/, '') || DEFAULT_BASE_URL
    return {
      baseUrl: fallbackBase,
      runUrl: configuredRunUrl || `${fallbackBase}/run`,
      weightsUrl: configuredWeightsUrl || `${fallbackBase}/weights`,
    }
  }

  if (configuredBaseUrl) {
    const baseUrl = configuredBaseUrl.replace(/\/+$/, '')
    return {
      baseUrl,
      runUrl: `${baseUrl}/run`,
      weightsUrl: `${baseUrl}/weights`,
    }
  }

  return {
    baseUrl: DEFAULT_BASE_URL,
    runUrl: `${DEFAULT_BASE_URL}/run`,
    weightsUrl: `${DEFAULT_BASE_URL}/weights`,
  }
}

export function getApiTargetInfo(): ApiTargetInfo {
  const { runUrl, weightsUrl } = getApiUrls()
  try {
    const parsed = new URL(runUrl)
    const hostname = parsed.hostname.toLowerCase()
    return {
      target: hostname === 'localhost' || hostname === '127.0.0.1' ? 'local' : 'modal',
      runUrl,
      weightsUrl,
      hostname: parsed.hostname,
    }
  } catch {
    const isLocal = runUrl.includes('localhost') || runUrl.includes('127.0.0.1')
    return {
      target: isLocal ? 'local' : 'modal',
      runUrl,
      weightsUrl,
      hostname: isLocal ? 'localhost' : 'remote',
    }
  }
}
