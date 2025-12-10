import { useFileContent } from "../../plugins/cathedral-plugin/src/client";
import { load } from "js-yaml";
import { useMemo } from "react";
import { cn } from "@/lib/utils";

type ParameterValue = string | number | boolean | null | ParameterValue[] | { [key: string]: ParameterValue };

/**
 * Extract a value from nested data using a jq-like path.
 * Supports:
 *   - .foo.bar  → nested keys
 *   - .foo[0]   → array index
 */
function extractPath(data: ParameterValue, path: string): ParameterValue | undefined {
  if (!path || path === ".") return data;

  const cleanPath = path.startsWith(".") ? path.slice(1) : path;
  if (!cleanPath) return data;

  const segments: Array<{ type: "key" | "index"; value: string | number }> = [];
  let current = "";
  let i = 0;

  while (i < cleanPath.length) {
    const char = cleanPath[i];

    if (char === ".") {
      if (current) {
        segments.push({ type: "key", value: current });
        current = "";
      }
      i++;
    } else if (char === "[") {
      if (current) {
        segments.push({ type: "key", value: current });
        current = "";
      }
      const closeIdx = cleanPath.indexOf("]", i);
      if (closeIdx === -1) return undefined;
      const inner = cleanPath.slice(i + 1, closeIdx);
      const idx = parseInt(inner, 10);
      if (isNaN(idx)) return undefined;
      segments.push({ type: "index", value: idx });
      i = closeIdx + 1;
    } else {
      current += char;
      i++;
    }
  }
  if (current) {
    segments.push({ type: "key", value: current });
  }

  let result: ParameterValue | undefined = data;

  for (const seg of segments) {
    if (result === null || result === undefined) return undefined;

    if (seg.type === "key") {
      if (typeof result !== "object" || Array.isArray(result)) return undefined;
      result = (result as Record<string, ParameterValue>)[seg.value as string];
    } else if (seg.type === "index") {
      if (!Array.isArray(result)) return undefined;
      result = result[seg.value as number];
    }
  }

  return result;
}

function formatNumber(value: number): string {
  if (Math.abs(value) < 0.0001 && value !== 0) return value.toExponential(1);
  if (Math.abs(value) >= 10000) return value.toExponential(1);
  if (Number.isInteger(value)) return value.toString();
  const str = value.toString();
  if (str.length > 6) return value.toPrecision(4);
  return str;
}

function formatValue(value: ParameterValue): string {
  if (value === null) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return formatNumber(value);
  if (Array.isArray(value)) return `[${value.length}]`;
  if (typeof value === "object") return "{...}";
  return String(value);
}

function getValueType(value: ParameterValue): "string" | "number" | "boolean" | "null" | "array" | "object" {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  if (typeof value === "object") return "object";
  if (typeof value === "boolean") return "boolean";
  if (typeof value === "number") return "number";
  return "string";
}

interface ParameterBadgeProps {
  /** Path to the YAML or JSON file */
  path: string;
  /** jq-like path to the value (e.g., ".base.N_E") */
  keyPath: string;
  /** Optional label override (defaults to last segment of keyPath) */
  label?: string;
  /** Optional unit suffix (e.g., "ms", "Hz") */
  unit?: string;
}

export function ParameterBadge({ path, keyPath, label, unit }: ParameterBadgeProps) {
  const { content, loading, error } = useFileContent(path);

  const { value, displayLabel } = useMemo(() => {
    if (!content) return { value: undefined, displayLabel: "" };

    let data: Record<string, ParameterValue> | null = null;

    if (path.endsWith(".yaml") || path.endsWith(".yml")) {
      try {
        data = load(content) as Record<string, ParameterValue>;
      } catch {
        return { value: undefined, displayLabel: "" };
      }
    }

    if (path.endsWith(".json")) {
      try {
        data = JSON.parse(content) as Record<string, ParameterValue>;
      } catch {
        return { value: undefined, displayLabel: "" };
      }
    }

    if (!data) return { value: undefined, displayLabel: "" };

    const extracted = extractPath(data, keyPath);

    // Derive label from keyPath if not provided
    const cleanPath = keyPath.startsWith(".") ? keyPath.slice(1) : keyPath;
    const parts = cleanPath.split(".");
    const derivedLabel = label || parts[parts.length - 1].replace(/\[\d+\]/g, "");

    return { value: extracted, displayLabel: derivedLabel };
  }, [content, path, keyPath, label]);

  if (loading) {
    return (
      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-muted/50 border border-border/50">
        <span className="w-2 h-2 border border-muted-foreground/40 border-t-transparent rounded-full animate-spin" />
      </span>
    );
  }

  if (error || value === undefined) {
    return (
      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-destructive/10 border border-destructive/30">
        <span className="text-[10px] font-mono text-destructive">—</span>
      </span>
    );
  }

  const type = getValueType(value);
  const formattedValue = formatValue(value);

  return (
    <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md font-mono text-[11px]">
      <span className="text-muted-foreground">{displayLabel}</span>
      <span className="text-muted-foreground/40">=</span>
      <span
        className={cn(
          "font-medium tabular-nums",
          type === "number" && "text-foreground",
          type === "string" && "text-amber-600 dark:text-amber-500",
          type === "boolean" && "text-cyan-600 dark:text-cyan-500",
          type === "null" && "text-muted-foreground/50",
          type === "array" && "text-purple-600 dark:text-purple-400",
          type === "object" && "text-purple-600 dark:text-purple-400"
        )}
      >
        {type === "string" ? `"${formattedValue}"` : formattedValue}
      </span>
      {unit && <span className="text-muted-foreground/60">{unit}</span>}
    </span>
  );
}
