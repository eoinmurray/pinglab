type FormatAxisTickOptions = {
  integer?: boolean
  compactThreshold?: number
  scientificLowerThreshold?: number
  maxFractionDigits?: number
  significantDigits?: number
}

const DEFAULT_OPTIONS: Required<FormatAxisTickOptions> = {
  integer: false,
  compactThreshold: 1e4,
  scientificLowerThreshold: 1e-3,
  maxFractionDigits: 3,
  significantDigits: 3,
}

const COMPACT_FORMATTER = new Intl.NumberFormat('en-US', {
  notation: 'compact',
  maximumFractionDigits: 1,
})

function trimNumericString(text: string) {
  const trimmed = text
    .replace(/(\.\d*?[1-9])0+$/u, '$1')
    .replace(/\.0+$/u, '')
    .replace(/e\+?/u, 'e')
  return trimmed === '-0' ? '0' : trimmed
}

export function formatAxisTick(value: number, options: FormatAxisTickOptions = {}) {
  const resolved = { ...DEFAULT_OPTIONS, ...options }
  if (!Number.isFinite(value)) {
    return '0'
  }

  const abs = Math.abs(value)
  if (abs < 1e-12) {
    return '0'
  }

  if (resolved.integer) {
    return Math.round(value).toString()
  }

  if (abs >= resolved.compactThreshold && abs < 1e12) {
    return COMPACT_FORMATTER.format(value)
  }

  if (abs >= 1) {
    const fractionDigits =
      abs >= 100 ? 0 : abs >= 10 ? Math.min(1, resolved.maxFractionDigits) : resolved.maxFractionDigits
    return trimNumericString(value.toFixed(fractionDigits))
  }

  if (abs >= resolved.scientificLowerThreshold) {
    return trimNumericString(value.toPrecision(resolved.significantDigits))
  }

  return trimNumericString(value.toExponential(Math.max(1, resolved.significantDigits - 1)))
}
