import { useMemo, useState, useCallback } from "react";
import { createPortal } from "react-dom";
import { useTheme } from "next-themes";
import { useDirectory } from "../../../plugins/cathedral-plugin/src/client";
import { FileEntry } from "../../../plugins/cathedral-plugin/src/lib";
import { cathedralPluginConfig } from "../../../cathedral-plugin.config";
import { minimatch } from "minimatch";
import { useParams } from "react-router-dom";
import { cn } from "@/lib/utils";
import { X, ChevronLeft, ChevronRight, Expand } from "lucide-react";
import { FULLSCREEN_DATA_ATTR } from "@/lib/constants";
import { useKeyBindings } from "@/hooks/useKeyBindings";

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

function getImageLabel(path: string): string {
  const filename = path.split('/').pop() || path;
  return filename
    .replace(/\.(png|jpg|jpeg|gif|svg|webp)$/i, '')
    .replace(/[_-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export default function SlideGallery({
  path,
  relativePath,
  globs = null,
  single = false,
  limit,
  title,
  caption,
}: {
  path?: string,
  relativePath?: string,
  globs?: string[] | null,
  single?: boolean,
  limit?: number,
  title?: string,
  caption?: string,
}) {
  const { "*": paramPath = "." } = useParams();
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

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
    const matchedPaths = paths.filter(p => {
      return globs.some(glob => minimatch(p.split('/').pop() || '', glob));
    });
    paths.splice(0, paths.length, ...matchedPaths);
  }

  paths.sort((a, b) => {
    const nums = (s: string) => (s.match(/\d+/g) || []).map(Number);
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
  let filteredPaths = useMemo(() => filterPathsByTheme(paths, resolvedTheme), [paths, resolvedTheme]);

  if (limit) {
    filteredPaths = filteredPaths.slice(0, limit);
  }

  const gridCols = useMemo(() => {
    const count = filteredPaths.length;
    if (count === 1) return "grid-cols-1";
    if (count === 2) return "md:grid-cols-2 sm:grid-cols-1";
    if (count <= 4) return "md:grid-cols-3 sm:grid-cols-1";
    return "grid-cols-3";
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

  useKeyBindings(
    [
      { key: "Escape", action: close },
      { key: "ArrowLeft", action: goToPrevious },
      { key: "ArrowRight", action: goToNext },
    ],
    { enabled: () => selectedIndex !== null }
  );

  if (!directory) {
    return (
      <div className="not-prose h-[50vh] flex items-center justify-center">
        <div className="text-muted-foreground font-mono text-xs">loading...</div>
      </div>
    );
  }

  if (filteredPaths.length === 0) {
    return (
      <div className="not-prose h-[50vh] flex items-center justify-center">
        <div className="text-muted-foreground font-mono text-xs">no images</div>
      </div>
    );
  }

  // Single image - maximize to fill available space
  if (single) {
    return (
      <>
        <figure className="not-prose w-full h-[90vh] flex flex-col items-center justify-center gap-4">
          {title && (
            <h2 className="text-2xl md:text-3xl font-semibold tracking-tight text-foreground text-center max-w-4xl px-4">
              {title}
            </h2>
          )}
          <div
            className="relative group cursor-pointer max-h-[calc(100%-6rem)] flex items-center justify-center overflow-hidden"
            onClick={() => setSelectedIndex(0)}
          >
            <img
              src={`${cathedralPluginConfig.contentPrefix}/${filteredPaths[0]}`}
              alt={caption || title || ""}
              className="max-h-full max-w-full w-auto h-auto object-contain transition-transform duration-700 ease-out-expo group-hover:scale-[1.02]"
            />
            {/* Subtle vignette on hover */}
            <div className="absolute inset-0 bg-gradient-to-t from-foreground/0 via-transparent to-foreground/0 group-hover:from-foreground/10 transition-all duration-500 pointer-events-none" />
            {/* Expand icon */}
            <div className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
              <div className="bg-background/80 backdrop-blur-sm p-1.5 shadow-sm">
                <Expand className="h-3 w-3 text-foreground" />
              </div>
            </div>
          </div>
          {caption && (
            <figcaption className="text-xl md:text-2xl text-muted-foreground text-center max-w-3xl px-4">
              {caption}
            </figcaption>
          )}
        </figure>

        {/* Fullscreen Lightbox */}
        {selectedIndex !== null && createPortal(
          <div
            className="fixed inset-0 z-[9999] bg-background"
            onClick={close}
            {...{[FULLSCREEN_DATA_ATTR]: "true"}}
          >
            {/* Top bar */}
            <div className="fixed top-0 left-0 right-0 z-10 flex items-center justify-between px-4 py-3 bg-background/80 backdrop-blur-sm">
              <div className="font-mono text-xs text-muted-foreground tabular-nums">
                {String(selectedIndex + 1).padStart(2, '0')} / {String(filteredPaths.length).padStart(2, '0')}
              </div>
              <button
                onClick={close}
                className="p-2 text-muted-foreground hover:text-foreground transition-colors"
                aria-label="Close"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* Main image */}
            <div
              className="fixed inset-0 flex items-center justify-center p-16"
              onClick={(e) => e.stopPropagation()}
            >
              <img
                src={`${cathedralPluginConfig.contentPrefix}/${filteredPaths[selectedIndex]}`}
                alt={getImageLabel(filteredPaths[selectedIndex])}
                className="max-w-full max-h-full object-contain"
              />
            </div>

            {/* Caption */}
            <div className="fixed bottom-0 left-0 right-0 z-10 p-4 text-center bg-background/80 backdrop-blur-sm">
              <span className="font-mono text-xs text-muted-foreground">
                {getImageLabel(filteredPaths[selectedIndex])}
              </span>
            </div>
          </div>,
          document.body
        )}
      </>
    );
  }

  // Grid of images - maximize to fill screen
  return (
    <>
      <figure className="not-prose w-full h-[100vh] lg:px-12 flex flex-col">
        {title && (
          <h2 className="text-2xl md:text-3xl font-semibold tracking-tight text-foreground text-center py-4 max-w-4xl mx-auto px-4">
            {title}
          </h2>
        )}
        <div className={cn("grid flex-1 w-full min-h-0", gridCols, "gap-2")}>
          {filteredPaths.map((imgPath, index) => (
            <div
              key={imgPath}
              className="relative group flex items-center justify-center overflow-hidden min-h-0 cursor-pointer"
              onClick={() => setSelectedIndex(index)}
            >
              <img
                src={`${cathedralPluginConfig.contentPrefix}/${imgPath}`}
                alt=""
                className="max-h-full max-w-full w-auto h-auto object-contain transition-transform duration-700 ease-out-expo group-hover:scale-[1.02]"
              />
              {/* Subtle vignette on hover */}
              <div className="absolute inset-0 bg-gradient-to-t from-foreground/0 via-transparent to-foreground/0 group-hover:from-foreground/10 transition-all duration-500 pointer-events-none" />
              {/* Index - terminal style */}
              <div className="absolute bottom-3 left-3 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                <span className="font-mono text-[10px] text-white/90 bg-foreground/60 backdrop-blur-sm px-1.5 py-0.5 tracking-wider">
                  {String(index + 1).padStart(2, '0')}
                </span>
              </div>
              {/* Expand icon */}
              <div className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                <div className="bg-background/80 backdrop-blur-sm p-1.5 shadow-sm">
                  <Expand className="h-3 w-3 text-foreground" />
                </div>
              </div>
            </div>
          ))}
        </div>
        {caption && (
          <figcaption className="text-xl md:text-2xl text-muted-foreground text-center py-6 max-w-4xl mx-auto px-4">
            {caption}
          </figcaption>
        )}
      </figure>

      {/* Fullscreen Lightbox */}
      {selectedIndex !== null && createPortal(
        <div
          className="fixed inset-0 z-[9999] bg-background"
          onClick={close}
          {...{[FULLSCREEN_DATA_ATTR]: "true"}}
        >
          {/* Top bar */}
          <div className="fixed top-0 left-0 right-0 z-10 flex items-center justify-between px-4 py-3 bg-background/80 backdrop-blur-sm">
            <div className="font-mono text-xs text-muted-foreground tabular-nums">
              {String(selectedIndex + 1).padStart(2, '0')} / {String(filteredPaths.length).padStart(2, '0')}
            </div>
            <button
              onClick={close}
              className="p-2 text-muted-foreground hover:text-foreground transition-colors"
              aria-label="Close"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {/* Navigation: Previous */}
          {selectedIndex > 0 && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                goToPrevious();
              }}
              className="fixed left-4 top-1/2 -translate-y-1/2 z-10 p-2 text-muted-foreground hover:text-foreground transition-colors"
              aria-label="Previous image"
            >
              <ChevronLeft className="h-8 w-8" />
            </button>
          )}

          {/* Navigation: Next */}
          {selectedIndex < filteredPaths.length - 1 && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                goToNext();
              }}
              className="fixed right-4 top-1/2 -translate-y-1/2 z-10 p-2 text-muted-foreground hover:text-foreground transition-colors"
              aria-label="Next image"
            >
              <ChevronRight className="h-8 w-8" />
            </button>
          )}

          {/* Main image */}
          <div
            className="fixed inset-0 flex items-center justify-center p-16"
            onClick={(e) => e.stopPropagation()}
          >
            <img
              src={`${cathedralPluginConfig.contentPrefix}/${filteredPaths[selectedIndex]}`}
              alt={getImageLabel(filteredPaths[selectedIndex])}
              className="max-w-full max-h-full object-contain"
            />
          </div>

          {/* Caption */}
          <div className="fixed bottom-0 left-0 right-0 z-10 p-4 text-center bg-background/80 backdrop-blur-sm">
            <span className="font-mono text-xs text-muted-foreground">
              {getImageLabel(filteredPaths[selectedIndex])}
            </span>
          </div>
        </div>,
        document.body
      )}
    </>
  );
}
