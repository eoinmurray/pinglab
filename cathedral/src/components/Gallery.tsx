import { useState, useEffect, useMemo } from "react";
import { X, ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { useTheme } from "next-themes";
import { fetchDirectory } from "../../plugins/cathedral-plugin/src/client";
import { FileEntry } from "../../plugins/cathedral-plugin/src/lib";
import { cathedralPluginConfig } from "../../cathedral-plugin.config";
import { minimatch } from "minimatch";

function filterPathsByTheme(paths: string[], theme: string | undefined): string[] {
  // Group paths by base name (without _light or _dark suffix)
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
      // Keep original path that doesn't have light/dark variants
      pathGroups.set(path, { original: path });
    }
  });

  // Select appropriate variant based on theme
  const filtered: string[] = [];
  pathGroups.forEach((group, baseName) => {
    if (group.original) {
      // No variants, use original
      filtered.push(group.original);
    } else {
      // Has variants, choose based on theme
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
  caption,
  globs = null,
  single = false,
  } : {
  path: string,
  caption?: string,
  globs?: string[] | null,
  single?: boolean
}) {
  const { directory } = fetchDirectory(path)

  const imageChildren = directory?.children
    .filter((child): child is FileEntry => {
      return !!child.name.match(/\.(png|jpeg|gif|svg|webp)$/i) && child.type === "file";
    })

  const paths = imageChildren?.map(child => child.path) || [];

  if (globs && globs.length > 0) {
    const matchedPaths = paths.filter(path => {
      return globs.some(glob => minimatch(path.split('/').pop() || '', glob));
    });
    paths.splice(0, paths.length, ...matchedPaths);
  }

  paths.sort((a, b) => {
    const nums = (s: string) =>
      (s.match(/\d+/g) || []).map(Number); // extract all numbers in order

    const na = nums(a);
    const nb = nums(b);

    // lexicographic numeric comparison of number-arrays
    const len = Math.max(na.length, nb.length);
    for (let i = 0; i < len; i++) {
      const diff = (na[i] ?? 0) - (nb[i] ?? 0);
      if (diff !== 0) return diff;
    }

    // if the numbers are identical, fall back to plain string
    return a.localeCompare(b);
  });


  const { resolvedTheme } = useTheme();
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const filteredPaths = useMemo(() => filterPathsByTheme(paths, resolvedTheme), [paths, resolvedTheme]);
  
  const goToPrevious = () => {
    if (selectedIndex !== null && selectedIndex > 0) {
      setSelectedIndex(selectedIndex - 1);
    }
  };

  const goToNext = () => {
    if (selectedIndex !== null && selectedIndex < filteredPaths.length - 1) {
      setSelectedIndex(selectedIndex + 1);
    }
  };

  const close = () => {
    setSelectedIndex(null);
  };

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
  }, [selectedIndex]);

  return (
    <>
      <div className="border rounded">
        <div className="border-b px-3 py-2 bg-sidebar rounded-t">
          <span className="text-muted-foreground text-sm">
            Gallery for{' '}
          </span>
          <span className="text-muted-foreground text-xs font-mono">
            {path}
          </span>
          <span className="text-muted-foreground text-xs font-mono">
            {globs && globs.length > 0 && (
              <>
                {' '}|{' '}
                {globs.join(', ')}
              </>
            )}
          </span>

        </div>
        <div 
          className={
            cn(
              "w-full not-prose p-6",
              !single && "grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 ",
              "gap-2 sm:gap-3 md:gap-4"
          )}
        >
          {single && filteredPaths.length > 0 && (
            <div
              className="w-[50%] aspect-square overflow-hidden cursor-pointer rounded"
              onClick={() => setSelectedIndex(0)}
            >
              <img
                src={`${cathedralPluginConfig.contentPrefix}/${filteredPaths[0]}`}
                alt={filteredPaths[0]}
                className="w-full h-full object-cover hover:scale-105 transition-transform duration-200"
              />
            </div>
          )}

          {!single && filteredPaths.map((path, index) => (
            <div
              key={path}
              className="w-full aspect-square overflow-hidden cursor-pointer rounded"
              onClick={() => setSelectedIndex(index)}
            >
              <img
                src={`${cathedralPluginConfig.contentPrefix}/${path}`}
                alt={path}
                className="w-full h-full object-cover hover:scale-105 transition-transform duration-200"
              />
            </div>
          ))}
        </div>
        {caption && (
          <div className="mt-2 text-xs text-muted-foreground p-3 border-t rounded-b bg-sidebar">
            {caption}
          </div>
        )}
      </div>

      {selectedIndex !== null && (
        <div
          className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center p-2 sm:p-4"
          onClick={close}
          data-gallery-fullscreen="true"
        >
          <button
            onClick={close}
            className={cn(
              "absolute top-2 right-2 sm:top-4 sm:right-4",
              "p-1.5 sm:p-2 rounded-full",
              "bg-white/10 hover:bg-white/20 transition-colors",
              "text-white z-10"
            )}
            aria-label="Close"
          >
            <X className="h-5 w-5 sm:h-6 sm:w-6" />
          </button>

          <button
            onClick={(e) => {
              e.stopPropagation();
              goToPrevious();
            }}
            className={cn(
              "absolute left-2 sm:left-4",
              "p-1.5 sm:p-2 rounded-full",
              "bg-white/10 hover:bg-white/20 transition-colors",
              "text-white z-10"
            )}
            aria-label="Previous image"
          >
            <ChevronLeft className="h-5 w-5 sm:h-6 sm:w-6" />
          </button>

          <button
            onClick={(e) => {
              e.stopPropagation();
              goToNext();
            }}
            className={cn(
              "absolute right-2 sm:right-4",
              "p-1.5 sm:p-2 rounded-full",
              "bg-white/10 hover:bg-white/20 transition-colors",
              "text-white z-10"
            )}
            aria-label="Next image"
          >
            <ChevronRight className="h-5 w-5 sm:h-6 sm:w-6" />
          </button>

          <img
            src={`${cathedralPluginConfig.contentPrefix}/${filteredPaths[selectedIndex]}`}
            alt={filteredPaths[selectedIndex]}
            className="max-h-[85vh] sm:max-h-[90vh] max-w-full object-contain"
            onClick={(e) => e.stopPropagation()}
          />

          <div className="absolute bottom-2 sm:bottom-4 left-1/2 -translate-x-1/2 text-white text-xs sm:text-sm bg-black/50 px-3 py-1.5 sm:px-4 sm:py-2 rounded-full">
            {selectedIndex + 1} / {filteredPaths.length}
          </div>
        </div>
      )}
    </>
  );
}