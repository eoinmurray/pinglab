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
    // Scientific notation for very small/large numbers
    if (Math.abs(value) < 0.001 && value !== 0) return value.toExponential(2);
    if (Math.abs(value) >= 10000) return value.toExponential(2);
    // Clean up floating point display
    if (Number.isInteger(value)) return value.toString();
    return value.toPrecision(4).replace(/\.?0+$/, '');
  }
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return `[${value.length} items]`;
  if (typeof value === "object") return `{${Object.keys(value).length} keys}`;
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

export function ParameterTable({ path, field }: { path: string; field?: string }) {
  const { file } = useDirectory(path);
  const { content, loading, error } = useFileContent(path);

  if (loading) {
    return (
      <div className="not-prose border border-border rounded-lg overflow-hidden">
        <div className="px-4 py-3 bg-muted/30 border-b border-border">
          <div className="h-4 w-32 bg-muted animate-pulse rounded" />
        </div>
        <div className="p-4 space-y-2">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-4 bg-muted/50 animate-pulse rounded" style={{ width: `${60 + Math.random() * 30}%` }} />
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="not-prose border border-destructive/30 rounded-lg bg-destructive/5 p-4">
        <p className="text-sm text-destructive font-medium">Failed to load parameters</p>
        <p className="text-xs text-destructive/70 mt-1 font-mono">{error.message}</p>
      </div>
    );
  }

  if (!file || !content) {
    return (
      <div className="not-prose border border-border rounded-lg bg-muted/30 p-4">
        <p className="text-sm text-muted-foreground">No file found at path</p>
        <p className="text-xs text-muted-foreground/70 mt-1 font-mono">{path}</p>
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
      <div className="not-prose border border-destructive/30 rounded-lg bg-destructive/5 p-4">
        <p className="text-sm text-destructive font-medium">Parse error</p>
        <p className="text-xs text-destructive/70 mt-1 font-mono">{(e as Error).message}</p>
      </div>
    );
  }

  const display = field ? getByPath(parsed, field) : parsed;

  // Handle non-object display values
  if (typeof display !== "object" || display === null) {
    return (
      <div className="not-prose border border-border rounded-lg overflow-hidden">
        <div className="px-4 py-2.5 bg-muted/30 border-b border-border flex items-center gap-2">
          <FileText className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="font-mono text-xs text-muted-foreground">{file.name}</span>
          {field && (
            <>
              <span className="text-muted-foreground/40">→</span>
              <span className="font-mono text-xs text-primary">{field}</span>
            </>
          )}
        </div>
        <div className="p-4">
          <span className={cn(
            "font-mono text-sm",
            getValueType(display) === "string" ? "text-foreground" : "text-primary"
          )}>
            {formatValue(display)}
          </span>
        </div>
      </div>
    );
  }

  const entries = flattenObject(display as Record<string, unknown>);

  return (
    <div className="not-prose border border-border rounded-lg overflow-hidden">
      {/* Header */}
      <div className="px-4 py-2.5 bg-muted/30 border-b border-border flex items-center gap-2">
        <FileText className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="font-mono text-xs text-muted-foreground">{file.name}</span>
        {field && (
          <>
            <span className="text-muted-foreground/40">→</span>
            <span className="font-mono text-xs text-primary">{field}</span>
          </>
        )}
        <span className="ml-auto text-xs text-muted-foreground/60">
          {Object.keys(display).length} parameters
        </span>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border bg-muted/20">
              <th className="text-left font-medium text-muted-foreground px-4 py-2 text-xs uppercase tracking-wider">
                Parameter
              </th>
              <th className="text-left font-medium text-muted-foreground px-4 py-2 text-xs uppercase tracking-wider">
                Value
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/50">
            {entries.map((entry, index) => {
              const isNested = entry.depth > 0;
              const isSection = typeof entry.value === "string" && entry.value.startsWith("{");

              return (
                <tr
                  key={entry.key}
                  className={cn(
                    "transition-colors",
                    index % 2 === 0 ? "bg-transparent" : "bg-muted/10",
                    "hover:bg-primary/5",
                    isSection && "bg-muted/20"
                  )}
                >
                  <td className={cn(
                    "px-4 py-2 font-mono text-xs",
                    isSection ? "text-foreground font-medium" : "text-muted-foreground"
                  )}>
                    <span style={{ paddingLeft: `${entry.depth * 12}px` }}>
                      {isNested ? entry.key.split(".").pop() : entry.key}
                    </span>
                  </td>
                  <td className="px-4 py-2 font-mono text-xs">
                    {isSection ? (
                      <span className="text-muted-foreground/60 italic">{entry.value as string}</span>
                    ) : (
                      <span className={cn(
                        getValueType(entry.value) === "number" && "text-primary",
                        getValueType(entry.value) === "boolean" && (entry.value ? "text-emerald-600 dark:text-emerald-400" : "text-muted-foreground"),
                        getValueType(entry.value) === "string" && "text-foreground",
                        getValueType(entry.value) === "null" && "text-muted-foreground/50 italic",
                        getValueType(entry.value) === "array" && "text-amber-600 dark:text-amber-400"
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
