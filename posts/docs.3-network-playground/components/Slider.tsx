import type { ChangeEvent } from "react";

type SliderProps = {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  precision?: number;
  onChange: (value: number) => void;
};

export default function Slider({
  label,
  value,
  min,
  max,
  step,
  precision = 5,
  onChange,
}: SliderProps) {
  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    onChange(Number(event.target.value));
  };

  return (
    <label className="flex flex-col gap-2">
      <span className="flex items-center justify-between text-[11px] text-zinc-600 dark:text-zinc-400">
        <span>{label}</span>
        <span className="tabular-nums text-black dark:text-zinc-100">
          {value.toFixed(precision)}
        </span>
      </span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={handleChange} />
    </label>
  );
}
