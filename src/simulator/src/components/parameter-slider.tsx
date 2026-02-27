import { useEffect, useRef, useState } from 'react'
import { ChevronLeft, ChevronRight, Info } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import { Slider } from '@/components/ui/slider'

type ParameterProps = {
  label: string
  min?: number
  max?: number
  step?: number
  defaultValue?: number
  value?: number
  onValueChange?: (value: number) => void
  info: string
}

export default function Parameter({
  label,
  min = 0,
  max = 100,
  step = 1,
  defaultValue = 50,
  value,
  onValueChange,
  info,
}: ParameterProps) {
  const clampedDefaultValue = Math.min(max, Math.max(min, defaultValue))
  const [internalValue, setInternalValue] = useState(clampedDefaultValue)
  const [dragValue, setDragValue] = useState<number | null>(null)
  const isControlled = value !== undefined && onValueChange !== undefined
  const current = Math.min(max, Math.max(min, isControlled ? value : internalValue))
  const displayedValue = dragValue ?? current
  const currentRef = useRef(current)
  const holdTimeoutRef = useRef<number | null>(null)
  const holdIntervalRef = useRef<number | null>(null)
  const holdStepCountRef = useRef(0)
  const HOLD_DELAY_MS = 250
  const HOLD_INTERVAL_MS = 75
  const FAST_HOLD_INTERVAL_MS = 38
  const SPEEDUP_AFTER_STEPS = 10

  useEffect(() => {
    currentRef.current = current
  }, [current])

  const commitValue = (next: number) => {
    const clampedNext = Math.min(max, Math.max(min, next))
    currentRef.current = clampedNext
    setDragValue(null)

    if (isControlled) {
      onValueChange(clampedNext)
      return
    }

    setInternalValue(clampedNext)
  }

  const decrement = () => {
    commitValue(currentRef.current - step)
  }

  const increment = () => {
    commitValue(currentRef.current + step)
  }

  const stopHold = () => {
    if (holdTimeoutRef.current !== null) {
      window.clearTimeout(holdTimeoutRef.current)
      holdTimeoutRef.current = null
    }

    if (holdIntervalRef.current !== null) {
      window.clearInterval(holdIntervalRef.current)
      holdIntervalRef.current = null
    }

    holdStepCountRef.current = 0
  }

  const startHold = (action: () => void) => {
    stopHold()

    const runStep = () => {
      action()
      holdStepCountRef.current += 1

      if (holdStepCountRef.current === SPEEDUP_AFTER_STEPS) {
        if (holdIntervalRef.current !== null) {
          window.clearInterval(holdIntervalRef.current)
        }
        holdIntervalRef.current = window.setInterval(runStep, FAST_HOLD_INTERVAL_MS)
      }
    }

    runStep()

    holdTimeoutRef.current = window.setTimeout(() => {
      holdIntervalRef.current = window.setInterval(runStep, HOLD_INTERVAL_MS)
    }, HOLD_DELAY_MS)
  }

  useEffect(() => stopHold, [])

  return (
    <div className="space-y-1 px-1.5">
      <div className="flex items-center justify-between text-[11px]">
        <div className="flex items-center gap-1">
          <span className="font-medium leading-none text-foreground">{label}</span>
          <Popover>
            <PopoverTrigger asChild>
              <Button
                type="button"
                size="icon-xs"
                variant="ghost"
                className="size-5 rounded-sm text-muted-foreground [&_svg]:size-3"
                aria-label={`${label} info`}
              >
                <Info />
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-64 text-[11px]" side="right" align="start">
              {info}
            </PopoverContent>
          </Popover>
        </div>
        <div className="flex items-center gap-1">
          <Button
            type="button"
            size="icon-xs"
            variant="ghost"
            className="size-5 rounded-sm [&_svg]:size-3"
            onClick={(event) => {
              if (event.detail === 0) {
                decrement()
              }
            }}
            onPointerDown={() => startHold(decrement)}
            onPointerUp={stopHold}
            onPointerCancel={stopHold}
            onPointerLeave={stopHold}
            disabled={current <= min}
            aria-label={`Decrease ${label}`}
          >
            <ChevronLeft />
          </Button>
          <span className="min-w-10 text-center text-[10px] text-muted-foreground">
            {displayedValue}
          </span>
          <Button
            type="button"
            size="icon-xs"
            variant="ghost"
            className="size-5 rounded-sm [&_svg]:size-3"
            onClick={(event) => {
              if (event.detail === 0) {
                increment()
              }
            }}
            onPointerDown={() => startHold(increment)}
            onPointerUp={stopHold}
            onPointerCancel={stopHold}
            onPointerLeave={stopHold}
            disabled={current >= max}
            aria-label={`Increase ${label}`}
          >
            <ChevronRight />
          </Button>
        </div>
      </div>
      <Slider
        className="[&_[data-slot=slider-thumb]]:size-3 [&_[data-slot=slider-track]]:h-1"
        value={[displayedValue]}
        onValueChange={(next) => {
          const nextValue = Math.min(max, Math.max(min, next[0] ?? currentRef.current))
          setDragValue(nextValue)
        }}
        onValueCommit={(next) => commitValue(next[0] ?? currentRef.current)}
        min={min}
        max={max}
        step={step}
      />
    </div>
  )
}
