import type { ChangeEvent } from "react";

type SelectProps = {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
};

export default function Select({ label, value, options, onChange }: SelectProps) {
  const handleChange = (event: ChangeEvent<HTMLSelectElement>) => {
    onChange(event.target.value);
  };

  return (
    <label className="flex flex-col gap-2">
      <span className="text-[11px] text-zinc-600 dark:text-zinc-400">{label}</span>
      <select
        value={value}
        onChange={handleChange}
        className="rounded-md border border-black/10 bg-white px-2 py-1 text-xs text-black dark:border-zinc-800 dark:bg-black dark:text-zinc-100"
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}
