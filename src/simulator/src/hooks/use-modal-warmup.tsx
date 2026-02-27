import { useEffect, useState } from 'react'
import { getApiTargetInfo } from '@/lib/api/target'

const RETRY_DELAY_MS = 1500
const DEBUG_KEEP_MODAL_WARMUP_OVERLAY = false

type ModalWarmupState = {
  isModal: boolean
  isWarming: boolean
}

const warmReadyByUrl = new Map<string, boolean>()
const warmPromiseByUrl = new Map<string, Promise<void>>()

function isWarm(url: string) {
  return warmReadyByUrl.get(url) === true
}

async function waitForModalWarm(weightsUrl: string): Promise<void> {
  while (true) {
    try {
      const response = await fetch(weightsUrl, {
        method: 'GET',
        headers: { Accept: 'application/json' },
        cache: 'no-store',
      })
      if (response.ok) {
        warmReadyByUrl.set(weightsUrl, true)
        return
      }
    } catch {
      // Keep polling until Modal is reachable.
    }
    await new Promise<void>((resolve) => {
      window.setTimeout(resolve, RETRY_DELAY_MS)
    })
  }
}

function ensureModalWarm(weightsUrl: string) {
  if (isWarm(weightsUrl)) {
    return Promise.resolve()
  }
  const existing = warmPromiseByUrl.get(weightsUrl)
  if (existing) {
    return existing
  }
  const warmupPromise = waitForModalWarm(weightsUrl).finally(() => {
    warmPromiseByUrl.delete(weightsUrl)
  })
  warmPromiseByUrl.set(weightsUrl, warmupPromise)
  return warmupPromise
}

export function useModalWarmup(): ModalWarmupState {
  const { target, weightsUrl } = getApiTargetInfo()
  const isModal = target === 'modal'
  if (isModal && DEBUG_KEEP_MODAL_WARMUP_OVERLAY) {
    return { isModal: true, isWarming: true }
  }
  const [isWarming, setIsWarming] = useState(isModal && !isWarm(weightsUrl))

  useEffect(() => {
    if (!isModal) {
      setIsWarming(false)
      return
    }
    if (isWarm(weightsUrl)) {
      setIsWarming(false)
      return
    }

    setIsWarming(true)
    let active = true
    void ensureModalWarm(weightsUrl).then(() => {
      if (active) {
        setIsWarming(false)
      }
    })
    return () => {
      active = false
    }
  }, [isModal, weightsUrl])

  return { isModal, isWarming }
}
