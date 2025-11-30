import { useState, useEffect, useMemo, useCallback } from "react";
import { X, ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { FULLSCREEN_DATA_ATTR } from "@/lib/constants";
import { useTheme } from "next-themes";
import { useDirectory } from "../../plugins/cathedral-plugin/src/client";
import { FileEntry } from "../../plugins/cathedral-plugin/src/lib";
import { cathedralPluginConfig } from "../../cathedral-plugin.config";
import { minimatch } from "minimatch";
import { useParams } from "react-router-dom";

function filterPathsByTheme(paths: string[], theme: string | undefined): string[] {
  const pathGroups = new Map<string, { light?: string; dark?: string; original?: string }>();

  paths.forEach(path => {
    if (path.endsWith('_light.png')) {
      const baseName = path.replace('_light.png', '');
      const group = pathGroups.get(baseName) || {};
      group.light = path;
      pathGroups.set(baseName, group);
    } else if (path.endsWith('_dark.png')) {
      const baseName = path.replace('_dark.png', '');
      const group = pathGroups.get(baseName) || {};
      group.dark = path;
      pathGroups.set(baseName, group);
    } else {
      pathGroups.set(path, { original: path });
    }
  });

  const filtered: string[] = [];
  pathGroups.forEach((group, baseName) => {
    if (group.original) {
      filtered.push(group.original);
    } else {
      const isDark = theme === 'dark';
      const preferredPath = isDark ? group.dark : group.light;
      const fallbackPath = isDark ? group.light : group.dark;
      filtered.push(preferredPath || fallbackPath || baseName);
    }
  });

  return filtered;
}

export default function Gallery({
  path,
  relativePath,
  caption,
  globs = null,
  single = false,
  limit,
}: {
  path?: string,
  relativePath?: string,
  caption?: string,
  globs?: string[] | null,
  single?: boolean,
  limit?: number
}) {
  const { "*": paramPath = "." } = useParams();

  if (relativePath) {
    const basePath = paramPath === "." ? "" : paramPath;
    path = basePath + (basePath.endsWith("/") ? "" : "/") + relativePath;
  }

  const { directory } = useDirectory(path);

  const imageChildren = directory?.children
    .filter((child): child is FileEntry => {
      return !!child.name.match(/\.(png|jpeg|gif|svg|webp)$/i) && child.type === "file";
    });

  const paths = imageChildren?.map(child => child.path) || [];

  if (globs && globs.length > 0) {
    const matchedPaths = paths.filter(path => {
      return globs.some(glob => minimatch(path.split('/').pop() || '', glob));
    });
    paths.splice(0, paths.length, ...matchedPaths);
  }

  paths.sort((a, b) => {
    const nums = (s: string) =>
      (s.match(/\d+/g) || []).map(Number);

    const na = nums(a);
    const nb = nums(b);

    const len = Math.max(na.length, nb.length);
    for (let i = 0; i < len; i++) {
      const diff = (na[i] ?? 0) - (nb[i] ?? 0);
      if (diff !== 0) return diff;
    }

    return a.localeCompare(b);
  });

  const { resolvedTheme } = useTheme();
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  let filteredPaths = useMemo(() => filterPathsByTheme(paths, resolvedTheme), [paths, resolvedTheme]);

  if (limit) {
    filteredPaths = filteredPaths.slice(0, limit);
  }

  // Dynamic grid based on image count
  const gridClass = useMemo(() => {
    const count = filteredPaths.length;
    if (count === 1) return "grid-cols-1 max-w-md";
    if (count === 2) return "grid-cols-2 max-w-2xl";
    if (count <= 4) return "grid-cols-2 md:grid-cols-4";
    if (count <= 6) return "grid-cols-3 md:grid-cols-6";
    return "grid-cols-4 md:grid-cols-5 lg:grid-cols-6";
  }, [filteredPaths.length]);

  const goToPrevious = useCallback(() => {
    setSelectedIndex(prev => prev !== null && prev > 0 ? prev - 1 : prev);
  }, []);

  const goToNext = useCallback(() => {
    setSelectedIndex(prev => prev !== null && prev < filteredPaths.length - 1 ? prev + 1 : prev);
  }, [filteredPaths.length]);

  const close = useCallback(() => {
    setSelectedIndex(null);
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (selectedIndex === null) return;

      if (e.key === "Escape") {
        close();
      } else if (e.key === "ArrowLeft") {
        goToPrevious();
      } else if (e.key === "ArrowRight") {
        goToNext();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedIndex, close, goToPrevious, goToNext]);

  return (
    <>
      <div className="border rounded-lg overflow-hidden shadow-sm">
        <div className="border-b px-4 py-2.5 bg-muted/50">
          <span className="font-mono text-xs text-muted-foreground">
            {path}
            {globs && globs.length > 0 && ` · ${globs.join(', ')}`}
          </span>
        </div>
        <div
          className={cn(
            "w-full not-prose p-4",
            !single && "grid gap-3",
            !single && gridClass,
          )}
        >
          {single && filteredPaths.length > 0 && (
            <div
              className="w-[50%] aspect-square overflow-hidden cursor-pointer rounded-md ring-1 ring-border hover:ring-primary/50 transition-all duration-200 hover:shadow-md"
              onClick={() => setSelectedIndex(0)}
            >
              <img
                src={`${cathedralPluginConfig.contentPrefix}/${filteredPaths[0]}`}
                alt={filteredPaths[0]}
                className="w-full h-full object-cover hover:scale-105 transition-transform duration-300"
              />
            </div>
          )}

          {!single && filteredPaths.map((path, index) => (
            <div
              key={path}
              className="aspect-square overflow-hidden cursor-pointer rounded-md ring-1 ring-border hover:ring-primary/50 transition-all duration-200 hover:shadow-md"
              onClick={() => setSelectedIndex(index)}
            >
              <img
                src={`${cathedralPluginConfig.contentPrefix}/${path}`}
                alt={path}
                className="w-full h-full object-cover hover:scale-105 transition-transform duration-300"
              />
            </div>
          ))}
        </div>
        {caption && (
          <div className="text-muted-foreground text-sm p-4 border-t bg-muted/50">
            {caption}
          </div>
        )}
      </div>

      {selectedIndex !== null && (
        <div
          className="fixed inset-0 bg-background/95 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={close}
          {...{[FULLSCREEN_DATA_ATTR]: "true"}}
        >
          <button
            onClick={close}
            className={cn(
              "absolute top-4 right-4",
              "p-2 rounded-full",
              "bg-muted hover:bg-accent transition-colors",
              "text-foreground z-10"
            )}
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>

          <button
            onClick={(e) => {
              e.stopPropagation();
              goToPrevious();
            }}
            disabled={selectedIndex === 0}
            className={cn(
              "absolute left-4",
              "p-2 rounded-full",
              "bg-muted hover:bg-accent transition-colors",
              "text-foreground z-10",
              "disabled:opacity-30 disabled:cursor-not-allowed"
            )}
            aria-label="Previous image"
          >
            <ChevronLeft className="h-5 w-5" />
          </button>

          <button
            onClick={(e) => {
              e.stopPropagation();
              goToNext();
            }}
            disabled={selectedIndex === filteredPaths.length - 1}
            className={cn(
              "absolute right-4",
              "p-2 rounded-full",
              "bg-muted hover:bg-accent transition-colors",
              "text-foreground z-10",
              "disabled:opacity-30 disabled:cursor-not-allowed"
            )}
            aria-label="Next image"
          >
            <ChevronRight className="h-5 w-5" />
          </button>

          <img
            src={`${cathedralPluginConfig.contentPrefix}/${filteredPaths[selectedIndex]}`}
            alt={filteredPaths[selectedIndex]}
            className="max-h-[90vh] max-w-full object-contain rounded-lg shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          />

          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 font-mono text-sm text-muted-foreground bg-muted px-4 py-2 rounded-full">
            {selectedIndex + 1} / {filteredPaths.length}
          </div>
        </div>
      )}
    </>
  );
}
