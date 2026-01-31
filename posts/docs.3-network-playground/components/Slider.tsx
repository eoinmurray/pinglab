import type { ChangeEvent } from "react";
import { useEffect, useRef } from "react";

type SliderProps = {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  precision?: number;
  onChange: (value: number) => void;
  showNudge?: boolean;
};

export default function Slider({
  label,
  value,
  min,
  max,
  step,
  precision = 5,
  onChange,
  showNudge = false,
}: SliderProps) {
  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    onChange(Number(event.target.value));
  };

  const valueRef = useRef(value);
  useEffect(() => {
    valueRef.current = value;
  }, [value]);

  const clamp = (next: number) => Math.min(max, Math.max(min, next));
  const nudge = (direction: -1 | 1) => {
    const next = clamp(valueRef.current + direction * step);
    onChange(Number(next.toFixed(precision)));
  };
  const holdTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const holdCount = useRef(0);

  const startHold = (direction: -1 | 1) => {
    nudge(direction);
    holdCount.current = 1;
    if (holdTimer.current) {
      clearInterval(holdTimer.current);
    }
    holdTimer.current = setInterval(() => {
      holdCount.current += 1;
      const repeats = holdCount.current;
      nudge(direction);
      if (repeats === 10 && holdTimer.current) {
        clearInterval(holdTimer.current);
        holdTimer.current = setInterval(() => nudge(direction), 90);
      }
    }, 180);
  };

  const stopHold = () => {
    if (holdTimer.current) {
      clearInterval(holdTimer.current);
      holdTimer.current = null;
    }
    holdCount.current = 0;
  };

  return (
    <label className="flex flex-col gap-2">
      <span className="flex items-center justify-between text-[11px] text-zinc-600 dark:text-zinc-400">
        <span>{label}</span>
        <span className="flex items-center gap-1">
          {showNudge ? (
            <button
              type="button"
              onMouseDown={() => startHold(-1)}
              onMouseUp={stopHold}
              onMouseLeave={stopHold}
              onTouchStart={() => startHold(-1)}
              onTouchEnd={stopHold}
              className="rounded border border-black/10 px-1 text-[10px] text-zinc-600 hover:bg-black/5 dark:border-zinc-800 dark:text-zinc-300 dark:hover:bg-white/10"
              aria-label={`Decrease ${label}`}
            >
              ◀
            </button>
          ) : null}
          <span className="tabular-nums text-black dark:text-zinc-100">
            {value.toFixed(precision)}
          </span>
          {showNudge ? (
            <button
              type="button"
              onMouseDown={() => startHold(1)}
              onMouseUp={stopHold}
              onMouseLeave={stopHold}
              onTouchStart={() => startHold(1)}
              onTouchEnd={stopHold}
              className="rounded border border-black/10 px-1 text-[10px] text-zinc-600 hover:bg-black/5 dark:border-zinc-800 dark:text-zinc-300 dark:hover:bg-white/10"
              aria-label={`Increase ${label}`}
            >
              ▶
            </button>
          ) : null}
        </span>
      </span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={handleChange} />
    </label>
  );
}
