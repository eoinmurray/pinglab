import { useFileContent } from "../../plugins/cathedral-plugin/src/client";
import { load } from "js-yaml";
import { useMemo, useState } from "react";
import { cn } from "@/lib/utils";

type ParameterValue = string | number | boolean | null | ParameterValue[] | { [key: string]: ParameterValue };

/**
 * Extract a value from nested data using a jq-like path.
 * Supports:
 *   - .foo.bar  → nested keys
 *   - .foo[0]   → array index
 *   - .foo[]    → all array elements (returns array)
 *   - .foo.bar, .foo.baz → multiple paths (comma-separated in keys array)
 */
function extractPath(data: ParameterValue, path: string): ParameterValue | undefined {
  if (!path || path === ".") return data;

  // Remove leading dot if present
  const cleanPath = path.startsWith(".") ? path.slice(1) : path;
  if (!cleanPath) return data;

  // Parse path segments: split on . but handle [] notation
  const segments: Array<{ type: "key" | "index" | "all"; value: string | number }> = [];
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
      // Find closing bracket
      const closeIdx = cleanPath.indexOf("]", i);
      if (closeIdx === -1) return undefined;
      const inner = cleanPath.slice(i + 1, closeIdx);
      if (inner === "") {
        segments.push({ type: "all", value: 0 });
      } else {
        const idx = parseInt(inner, 10);
        if (isNaN(idx)) return undefined;
        segments.push({ type: "index", value: idx });
      }
      i = closeIdx + 1;
    } else {
      current += char;
      i++;
    }
  }
  if (current) {
    segments.push({ type: "key", value: current });
  }

  // Traverse the data
  let result: ParameterValue | undefined = data;

  for (const seg of segments) {
    if (result === null || result === undefined) return undefined;

    if (seg.type === "key") {
      if (typeof result !== "object" || Array.isArray(result)) return undefined;
      result = (result as Record<string, ParameterValue>)[seg.value as string];
    } else if (seg.type === "index") {
      if (!Array.isArray(result)) return undefined;
      result = result[seg.value as number];
    } else if (seg.type === "all") {
      if (!Array.isArray(result)) return undefined;
      // Return the array itself for further processing
      result = result;
    }
  }

  return result;
}

/**
 * Build a filtered data object from an array of jq-like paths.
 * Each path extracts data and places it in the result under the final key name.
 */
function filterData(
  data: Record<string, ParameterValue>,
  keys: string[]
): Record<string, ParameterValue> {
  const result: Record<string, ParameterValue> = {};

  for (const keyPath of keys) {
    const extracted = extractPath(data, keyPath);
    if (extracted === undefined) continue;

    // Use the last segment as the key name, or the full path for nested
    const cleanPath = keyPath.startsWith(".") ? keyPath.slice(1) : keyPath;

    // For simple paths like .base.N_E, use "N_E" as key
    // For paths with [], preserve more context
    let keyName: string;
    if (cleanPath.includes("[")) {
      // Use full path but clean it up
      keyName = cleanPath.replace(/\[\]/g, "").replace(/\[(\d+)\]/g, "_$1");
    } else {
      // Use last segment
      const parts = cleanPath.split(".");
      keyName = parts[parts.length - 1];
    }

    result[keyName] = extracted;
  }

  return result;
}

function getValueType(value: ParameterValue): "string" | "number" | "boolean" | "null" | "array" | "object" {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  if (typeof value === "object") return "object";
  if (typeof value === "boolean") return "boolean";
  if (typeof value === "number") return "number";
  return "string";
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
  return String(value);
}

// Renders a flat section of key-value pairs in a dense grid
function ParameterGrid({ entries }: { entries: [string, ParameterValue][] }) {
  if (entries.length === 0) return null;

  return (
    <div className="grid grid-cols-[repeat(auto-fill,minmax(180px,1fr))] gap-x-6 gap-y-px">
      {entries.map(([key, value]) => {
        const type = getValueType(value);
        return (
          <div
            key={key}
            className="flex items-baseline justify-between gap-2 py-1 border-b border-border/30 last:border-b-0 group hover:bg-muted/30 -mx-1.5 px-1.5 rounded-sm transition-colors"
          >
            <span className="text-[11px] text-muted-foreground font-mono truncate">
              {key}
            </span>
            <span
              className={cn(
                "text-[11px] font-mono tabular-nums font-medium shrink-0",
                type === "number" && "text-foreground",
                type === "string" && "text-amber-600 dark:text-amber-500",
                type === "boolean" && "text-cyan-600 dark:text-cyan-500",
                type === "null" && "text-muted-foreground/50"
              )}
            >
              {type === "string" ? `"${formatValue(value)}"` : formatValue(value)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// Renders a nested section with its own header
function ParameterSection({
  name,
  data,
  depth = 0
}: {
  name: string;
  data: Record<string, ParameterValue>;
  depth?: number;
}) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  // Separate leaf values from nested objects
  const entries = Object.entries(data);
  const leafEntries = entries.filter(([, v]) => {
    const t = getValueType(v);
    return t !== "object" && t !== "array";
  });
  const nestedEntries = entries.filter(([, v]) => {
    const t = getValueType(v);
    return t === "object" || t === "array";
  });

  return (
    <div className={cn(depth === 0 && "mb-4 last:mb-0")}>
      {/* Section header */}
      <button
        onClick={() => setIsCollapsed(!isCollapsed)}
        className={cn(
          "flex items-center gap-2 w-full text-left group mb-1.5",
          depth === 0 && "pb-1 border-b border-border/50"
        )}
      >
        {/* Collapse indicator */}
        <span className={cn(
          "text-[10px] text-muted-foreground/60 transition-transform duration-150 select-none",
          isCollapsed && "-rotate-90"
        )}>
          {isCollapsed ? "+" : "-"}
        </span>

        {/* Section name */}
        <span className={cn(
          "font-mono text-[11px] uppercase tracking-widest",
          depth === 0
            ? "text-foreground/80 font-semibold"
            : "text-muted-foreground/70"
        )}>
          {name.replace(/_/g, " ")}
        </span>

        {/* Count badge */}
        <span className="text-[9px] font-mono text-muted-foreground/40 ml-auto">
          {entries.length}
        </span>
      </button>

      {/* Content */}
      {!isCollapsed && (
        <div className={cn(
          depth > 0 && "pl-3 ml-1 border-l border-border/40"
        )}>
          {/* Leaf values grid */}
          {leafEntries.length > 0 && (
            <div className={cn(nestedEntries.length > 0 && "mb-3")}>
              <ParameterGrid entries={leafEntries} />
            </div>
          )}

          {/* Nested sections */}
          {nestedEntries.map(([key, value]) => {
            const type = getValueType(value);
            if (type === "array") {
              // Handle arrays - show as indexed items
              const arr = value as ParameterValue[];
              return (
                <div key={key} className="mb-2 last:mb-0">
                  <div className="text-[10px] font-mono text-muted-foreground/60 uppercase tracking-wider mb-1">
                    {key} [{arr.length}]
                  </div>
                  <div className="pl-3 ml-1 border-l border-border/40">
                    {arr.map((item, i) => {
                      const itemType = getValueType(item);
                      if (itemType === "object") {
                        return (
                          <ParameterSection
                            key={i}
                            name={`${i}`}
                            data={item as Record<string, ParameterValue>}
                            depth={depth + 1}
                          />
                        );
                      }
                      return (
                        <div key={i} className="text-[11px] font-mono text-foreground py-0.5">
                          [{i}] {formatValue(item)}
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            }
            return (
              <ParameterSection
                key={key}
                name={key}
                data={value as Record<string, ParameterValue>}
                depth={depth + 1}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}

interface ParameterTableProps {
  /** Path to the YAML or JSON file */
  path: string;
  /**
   * Optional array of jq-like paths to filter which parameters to show.
   * Examples:
   *   - [".base.N_E", ".base.N_I"] → show only N_E and N_I from base
   *   - [".base"] → show entire base section
   *   - [".default_inputs", ".base.dt"] → show default_inputs section and dt from base
   */
  keys?: string[];
}

/**
 * Estimate the height contribution of a data structure.
 * Each leaf entry ~24px, section headers ~28px, nested depth adds padding.
 */
function estimateHeight(data: Record<string, ParameterValue>, depth = 0): number {
  const entries = Object.entries(data);
  let height = 0;

  for (const [, value] of entries) {
    const type = getValueType(value);
    if (type === "object") {
      // Section header + nested content
      height += 28 + estimateHeight(value as Record<string, ParameterValue>, depth + 1);
    } else if (type === "array") {
      const arr = value as ParameterValue[];
      height += 28; // Array header
      for (const item of arr) {
        if (getValueType(item) === "object") {
          height += 24 + estimateHeight(item as Record<string, ParameterValue>, depth + 1);
        } else {
          height += 24;
        }
      }
    } else {
      height += 24; // Leaf entry
    }
  }

  return height;
}

/**
 * Split entries into balanced columns based on estimated height.
 */
function splitIntoColumns<T extends [string, ParameterValue]>(
  entries: T[],
  data: Record<string, ParameterValue>,
  numColumns: number
): T[][] {
  if (numColumns <= 1) return [entries];

  // Calculate height for each entry
  const entryHeights = entries.map(([key, value]) => {
    const type = getValueType(value);
    if (type === "object") {
      return 28 + estimateHeight(value as Record<string, ParameterValue>);
    } else if (type === "array") {
      const arr = value as ParameterValue[];
      let h = 28;
      for (const item of arr) {
        if (getValueType(item) === "object") {
          h += 24 + estimateHeight(item as Record<string, ParameterValue>);
        } else {
          h += 24;
        }
      }
      return h;
    }
    return 24;
  });

  const totalHeight = entryHeights.reduce((a, b) => a + b, 0);
  const targetPerColumn = totalHeight / numColumns;

  const columns: T[][] = [];
  let currentColumn: T[] = [];
  let currentHeight = 0;

  for (let i = 0; i < entries.length; i++) {
    const entry = entries[i];
    const entryHeight = entryHeights[i];

    // Start new column if we've exceeded target and have more columns to fill
    if (currentHeight >= targetPerColumn && columns.length < numColumns - 1 && currentColumn.length > 0) {
      columns.push(currentColumn);
      currentColumn = [];
      currentHeight = 0;
    }

    currentColumn.push(entry);
    currentHeight += entryHeight;
  }

  if (currentColumn.length > 0) {
    columns.push(currentColumn);
  }

  return columns;
}

export function ParameterTable({ path, keys }: ParameterTableProps) {
  const { content, loading, error } = useFileContent(path);

  const parsed = useMemo(() => {
    if (!content) return null;

    let data: Record<string, ParameterValue> | null = null;

    if (path.endsWith(".yaml") || path.endsWith(".yml")) {
      try {
        data = load(content) as Record<string, ParameterValue>;
      } catch (e) {
        console.error("Error parsing YAML:", e);
        return null;
      }
    }

    if (path.endsWith(".json")) {
      try {
        data = JSON.parse(content) as Record<string, ParameterValue>;
      } catch (e) {
        console.error("Error parsing JSON:", e);
        return null;
      }
    }

    if (!data) return null;

    // Apply key filtering if specified
    if (keys && keys.length > 0) {
      return filterData(data, keys);
    }

    return data;
  }, [content, path, keys]);

  if (loading) {
    return (
      <div className="my-6 p-4 rounded border border-border/50 bg-card/30">
        <div className="flex items-center gap-2 text-muted-foreground/60">
          <div className="w-3 h-3 border border-current border-t-transparent rounded-full animate-spin" />
          <span className="text-[11px] font-mono">loading parameters...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="my-6 p-3 rounded border border-destructive/30 bg-destructive/5">
        <p className="text-[11px] font-mono text-destructive">{error}</p>
      </div>
    );
  }

  if (!parsed) {
    return (
      <div className="my-6 p-3 rounded border border-border/50 bg-card/30">
        <p className="text-[11px] font-mono text-muted-foreground">unable to parse config</p>
      </div>
    );
  }

  const entries = Object.entries(parsed);

  // Separate top-level leaves from nested sections
  const topLeaves = entries.filter(([, v]) => {
    const t = getValueType(v);
    return t !== "object" && t !== "array";
  });
  const topNested = entries.filter(([, v]) => {
    const t = getValueType(v);
    return t === "object" || t === "array";
  });

  // Estimate total height and determine if we need columns
  const estHeight = estimateHeight(parsed);
  const HEIGHT_THRESHOLD = 500;
  const numColumns = estHeight > HEIGHT_THRESHOLD ? Math.min(Math.ceil(estHeight / HEIGHT_THRESHOLD), 3) : 1;
  const useColumns = numColumns > 1 && topNested.length > 1;

  // Split nested sections into columns if needed
  const columns = useColumns
    ? splitIntoColumns(topNested as [string, ParameterValue][], parsed, numColumns)
    : [topNested];

  const filename = path.split("/").pop() || path;

  // Render a single nested entry (shared between column and non-column layouts)
  const renderNestedEntry = ([key, value]: [string, ParameterValue]) => {
    const type = getValueType(value);
    if (type === "array") {
      const arr = value as ParameterValue[];
      return (
        <div key={key} className="mb-4 last:mb-0">
          <div className="text-[11px] font-mono text-foreground/80 uppercase tracking-widest font-semibold mb-1.5 pb-1 border-b border-border/50">
            {key.replace(/_/g, " ")} [{arr.length}]
          </div>
          <div className="pl-3 ml-1 border-l border-border/40">
            {arr.map((item, i) => {
              const itemType = getValueType(item);
              if (itemType === "object") {
                return (
                  <ParameterSection
                    key={i}
                    name={`${i}`}
                    data={item as Record<string, ParameterValue>}
                    depth={1}
                  />
                );
              }
              return (
                <div key={i} className="text-[11px] font-mono text-foreground py-0.5">
                  [{i}] {formatValue(item)}
                </div>
              );
            })}
          </div>
        </div>
      );
    }
    return (
      <ParameterSection
        key={key}
        name={key}
        data={value as Record<string, ParameterValue>}
        depth={0}
      />
    );
  };

  return (
    <div className="my-6 not-prose">
      {/* Minimal header */}
      <div className="flex items-center gap-2 mb-2 px-1">
        <div className="w-1.5 h-1.5 rounded-full bg-primary/60" />
        <span className="text-[10px] font-mono text-muted-foreground/50 uppercase tracking-wider">
          {filename}
        </span>
      </div>

      {/* Main container */}
      <div className="rounded border border-border/60 bg-card/20 p-3 overflow-hidden">
        {/* Top-level leaf values */}
        {topLeaves.length > 0 && (
          <div className={cn(topNested.length > 0 && "mb-4 pb-3 border-b border-border/30")}>
            <ParameterGrid entries={topLeaves} />
          </div>
        )}

        {/* Nested sections - with or without columns */}
        {useColumns ? (
          <div
            className="grid gap-6"
            style={{ gridTemplateColumns: `repeat(${columns.length}, 1fr)` }}
          >
            {columns.map((columnEntries, colIndex) => (
              <div key={colIndex} className={cn(
                colIndex > 0 && "border-l border-border/30 pl-6"
              )}>
                {columnEntries.map(renderNestedEntry)}
              </div>
            ))}
          </div>
        ) : (
          topNested.map(renderNestedEntry)
        )}
      </div>
    </div>
  );
}
