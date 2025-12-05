import { useDirectory, useFileContent } from "../../plugins/cathedral-plugin/src/client";
import yaml from "js-yaml";
import { cn } from "@/lib/utils";
import { FileText } from "lucide-react";

function getByPath(obj: Record<string, unknown>, path: string): unknown {
  return path.split(".").reduce<unknown>((acc, key) => {
    if (acc && typeof acc === 'object' && key in acc) {
      return (acc as Record<string, unknown>)[key];
    }
    return undefined;
  }, obj);
}

function formatValue(value: unknown): string {
  if (value === null) return "null";
  if (value === undefined) return "—";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") {
    if (Math.abs(value) < 0.001 && value !== 0) return value.toExponential(2);
    if (Math.abs(value) >= 10000) return value.toExponential(2);
    if (Number.isInteger(value)) return value.toString();
    return value.toPrecision(4).replace(/\.?0+$/, '');
  }
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return `[${value.length}]`;
  if (typeof value === "object") return `{${Object.keys(value).length}}`;
  return String(value);
}

function getValueType(value: unknown): string {
  if (value === null) return "null";
  if (value === undefined) return "undefined";
  if (Array.isArray(value)) return "array";
  return typeof value;
}

type FlattenedEntry = {
  key: string;
  value: unknown;
  depth: number;
};

function flattenObject(obj: Record<string, unknown>, prefix = "", depth = 0): FlattenedEntry[] {
  const entries: FlattenedEntry[] = [];

  for (const [key, value] of Object.entries(obj)) {
    const fullKey = prefix ? `${prefix}.${key}` : key;

    if (value && typeof value === "object" && !Array.isArray(value)) {
      entries.push({ key: fullKey, value: `{${Object.keys(value).length}}`, depth });
      entries.push(...flattenObject(value as Record<string, unknown>, fullKey, depth + 1));
    } else {
      entries.push({ key: fullKey, value, depth });
    }
  }

  return entries;
}

export function ParameterTable({ path, fields }: { path: string; fields?: string[] }) {
  const { file } = useDirectory(path);
  const { content, loading, error } = useFileContent(path);

  if (loading) {
    return (
      <div className="not-prose my-8">
        <div className="border border-border/50">
          <div className="px-4 py-3 border-b border-border/30 bg-muted/20">
            <div className="h-3 w-32 bg-muted/50 animate-pulse" />
          </div>
          <div className="p-4 space-y-3">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-3 bg-muted/30 animate-pulse" style={{ width: `${50 + Math.random() * 40}%`, animationDelay: `${i * 100}ms` }} />
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="not-prose my-8 border border-destructive/20 bg-destructive/5 p-4">
        <p className="font-mono text-xs text-destructive tracking-wide">error loading parameters</p>
        <p className="text-xs text-destructive/60 mt-1 font-mono">{error.message}</p>
      </div>
    );
  }

  if (!file || !content) {
    return (
      <div className="not-prose my-8 border border-border/50 bg-muted/20 p-4">
        <p className="font-mono text-xs text-muted-foreground tracking-wide">file not found</p>
        <p className="text-xs text-muted-foreground/50 mt-1 font-mono">{path}</p>
      </div>
    );
  }

  let parsed: Record<string, unknown>;
  try {
    parsed = file.name.endsWith(".json")
      ? JSON.parse(content)
      : yaml.load(content) as Record<string, unknown>;
  } catch (e) {
    return (
      <div className="not-prose my-8 border border-destructive/20 bg-destructive/5 p-4">
        <p className="font-mono text-xs text-destructive tracking-wide">parse error</p>
        <p className="text-xs text-destructive/60 mt-1 font-mono">{(e as Error).message}</p>
      </div>
    );
  }

  let display: Record<string, unknown>;
  if (fields && fields.length > 0) {
    display = {};
    for (const field of fields) {
      const value = getByPath(parsed, field);
      if (value !== undefined) {
        display[field] = value;
      }
    }
  } else {
    display = parsed;
  }

  if (typeof display !== "object" || display === null || Object.keys(display).length === 0) {
    return (
      <div className="not-prose my-8 border border-border/50">
        <div className="px-4 py-3 border-b border-border/30 bg-muted/10 flex items-center gap-2">
          <FileText className="h-3 w-3 text-muted-foreground/50" />
          <span className="font-mono text-xs text-muted-foreground">{file.name}</span>
        </div>
        <div className="p-4">
          <span className="font-mono text-xs text-muted-foreground/60">
            {fields ? "no matching fields" : "empty"}
          </span>
        </div>
      </div>
    );
  }

  const entries = flattenObject(display);

  return (
    <div className="not-prose my-8 border border-border/50 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-2.5 border-b border-border/30 bg-muted/10 flex items-center gap-3">
        <FileText className="h-3 w-3 text-muted-foreground/50" />
        <span className="font-mono text-xs text-muted-foreground">{file.name}</span>
        {fields && fields.length > 0 && (
          <>
            <span className="text-border">→</span>
            <span className="font-mono text-xs text-primary/70">{fields.join(", ")}</span>
          </>
        )}
        <span className="ml-auto font-mono text-[10px] text-muted-foreground/40 tracking-wide">
          {Object.keys(display).length} params
        </span>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-border/30">
              <th className="text-left font-mono text-[10px] text-muted-foreground/60 uppercase tracking-widest px-4 py-2">
                key
              </th>
              <th className="text-left font-mono text-[10px] text-muted-foreground/60 uppercase tracking-widest px-4 py-2">
                value
              </th>
            </tr>
          </thead>
          <tbody>
            {entries.map((entry, index) => {
              const isSection = typeof entry.value === "string" && entry.value.startsWith("{");

              return (
                <tr
                  key={entry.key}
                  className={cn(
                    "border-b border-border/20 last:border-b-0",
                    "transition-colors hover:bg-muted/30",
                    isSection && "bg-muted/20"
                  )}
                >
                  <td className={cn(
                    "px-4 py-2 font-mono text-xs",
                    isSection ? "text-foreground/80" : "text-muted-foreground"
                  )}>
                    <span style={{ paddingLeft: `${entry.depth * 12}px` }}>
                      {entry.depth > 0 ? entry.key.split(".").pop() : entry.key}
                    </span>
                  </td>
                  <td className="px-4 py-2 font-mono text-xs">
                    {isSection ? (
                      <span className="text-muted-foreground/40">{entry.value as string}</span>
                    ) : (
                      <span className={cn(
                        getValueType(entry.value) === "number" && "text-primary",
                        getValueType(entry.value) === "boolean" && (entry.value ? "text-primary" : "text-muted-foreground/50"),
                        getValueType(entry.value) === "string" && "text-foreground",
                        getValueType(entry.value) === "null" && "text-muted-foreground/40 italic",
                        getValueType(entry.value) === "array" && "text-primary/70"
                      )}>
                        {formatValue(entry.value)}
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
