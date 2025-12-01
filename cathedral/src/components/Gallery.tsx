import { useState, useEffect, useMemo, useCallback, ImgHTMLAttributes } from "react";
import { X, ChevronLeft, ChevronRight, Image, ZoomIn } from "lucide-react";
import { cn } from "@/lib/utils";
import { FULLSCREEN_DATA_ATTR } from "@/lib/constants";
import { useTheme } from "next-themes";
import { useDirectory } from "../../plugins/cathedral-plugin/src/client";
import { FileEntry } from "../../plugins/cathedral-plugin/src/lib";
import { cathedralPluginConfig } from "../../cathedral-plugin.config";
import { minimatch } from "minimatch";
import { useParams } from "react-router-dom";

// Image component with loading state
function LoadingImage({
  className,
  wrapperClassName,
  ...props
}: ImgHTMLAttributes<HTMLImageElement> & { wrapperClassName?: string }) {
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);

  return (
    <div className={cn("relative", wrapperClassName)}>
      {/* Loading skeleton */}
      {isLoading && !hasError && (
        <div className="absolute inset-0 bg-muted/50 animate-pulse flex items-center justify-center">
          <Image className="h-6 w-6 text-muted-foreground/30" />
        </div>
      )}
      {/* Error state */}
      {hasError && (
        <div className="absolute inset-0 bg-muted/30 flex items-center justify-center">
          <div className="text-center">
            <Image className="h-6 w-6 text-muted-foreground/50 mx-auto" />
            <span className="text-xs text-muted-foreground/50 mt-1 block">Failed</span>
          </div>
        </div>
      )}
      <img
        {...props}
        className={cn(
          className,
          isLoading && "opacity-0",
          hasError && "opacity-0"
        )}
        onLoad={(e) => {
          setIsLoading(false);
          props.onLoad?.(e);
        }}
        onError={(e) => {
          setIsLoading(false);
          setHasError(true);
          props.onError?.(e);
        }}
      />
    </div>
  );
}

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
  // Remove extension and clean up
  return filename
    .replace(/\.(png|jpg|jpeg|gif|svg|webp)$/i, '')
    .replace(/[_-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
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
    if (count === 1) return "grid-cols-1 max-w-lg";
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

  // Loading state
  if (!directory) {
    return (
      <div className="not-prose border border-border rounded-lg overflow-hidden">
        <div className="px-4 py-2.5 bg-muted/30 border-b border-border">
          <div className="h-4 w-48 bg-muted animate-pulse rounded" />
        </div>
        <div className="p-4 grid grid-cols-4 gap-3">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="aspect-square bg-muted/50 animate-pulse rounded-md" />
          ))}
        </div>
      </div>
    );
  }

  // Empty state
  if (filteredPaths.length === 0) {
    return (
      <div className="not-prose border border-border rounded-lg overflow-hidden">
        <div className="px-4 py-2.5 bg-muted/30 border-b border-border flex items-center gap-2">
          <Image className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="font-mono text-xs text-muted-foreground">{path}</span>
        </div>
        <div className="p-8 text-center">
          <p className="text-sm text-muted-foreground">No images found</p>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="not-prose rounded-lg overflow-hidden">
        {/* <div className="px-4 py-2.5 flex items-center gap-2">
          <Image className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="font-mono text-xs text-muted-foreground">
            {title ? title : path === "." ? "root" : ""}
          </span>
          <span className="ml-auto flex items-center gap-3 text-xs text-muted-foreground/60">
            <span className="flex items-center gap-1">
              <ZoomIn className="h-3 w-3" />
              <span>Click to expand</span>
            </span>
          </span>
        </div> */}

        {/* Grid */}
        <div
          className={cn(
            "w-full p-8",
            !single && "grid gap-3",
            !single && gridClass,
          )}
        >
          {single && filteredPaths.length > 0 && (
            <figure
              className="group relative w-full max-w-lg cursor-pointer"
              onClick={() => setSelectedIndex(0)}
            >
              <div className="relative overflow-hidden rounded-md border border-border bg-muted/20">
                <LoadingImage
                  src={`${cathedralPluginConfig.contentPrefix}/${filteredPaths[0]}`}
                  alt={getImageLabel(filteredPaths[0])}
                  className="w-full h-auto object-contain transition-transform duration-300 group-hover:scale-[1.02]"
                  wrapperClassName="w-full"
                />
                {/* Hover overlay */}
                <div className="absolute inset-0 bg-primary/5 opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-center justify-center pointer-events-none">
                  <div className="bg-background/90 backdrop-blur-sm rounded-full p-2 shadow-sm">
                    <ZoomIn className="h-4 w-4 text-primary" />
                  </div>
                </div>
              </div>
              <figcaption className="mt-2 text-xs text-muted-foreground font-mono">
                {getImageLabel(filteredPaths[0])}
              </figcaption>
            </figure>
          )}

          {!single && filteredPaths.map((imgPath, index) => (
            <figure
              key={imgPath}
              className="group relative cursor-pointer"
              onClick={() => setSelectedIndex(index)}
            >
              <div className="relative aspect-square overflow-hidden rounded-md border border-border bg-muted/20 transition-all duration-200 group-hover:border-primary/30 group-hover:shadow-md">
                <LoadingImage
                  src={`${cathedralPluginConfig.contentPrefix}/${imgPath}`}
                  alt={getImageLabel(imgPath)}
                  className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                  wrapperClassName="w-full h-full"
                />
                {/* Hover overlay */}
                <div className="absolute inset-0 bg-gradient-to-t from-black/40 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none" />
                {/* Index badge */}
                <div className="absolute bottom-2 left-2 bg-background/80 backdrop-blur-sm rounded px-1.5 py-0.5 text-[10px] font-mono text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none">
                  {index + 1}
                </div>
              </div>
            </figure>
          ))}
        </div>

        {/* Caption */}
        {caption && (
          <div className="not-prose px-4 py-3">
            <div className="not-prose mb-0 text-sm text-muted-foreground leading-relaxed">
              {caption}
            </div>
          </div>
        )}
      </div>

      {/* Fullscreen Lightbox */}
      {selectedIndex !== null && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center"
          onClick={close}
          {...{[FULLSCREEN_DATA_ATTR]: "true"}}
        >
          {/* Backdrop */}
          <div className="absolute inset-0 bg-background/98 backdrop-blur-md" />

          {/* Close button */}
          <button
            onClick={close}
            className={cn(
              "absolute top-4 right-4 z-20",
              "p-2.5 rounded-full",
              "bg-muted/80 hover:bg-muted border border-border",
              "text-foreground transition-all duration-200",
              "hover:scale-105 active:scale-95"
            )}
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>

          {/* Navigation: Previous */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              goToPrevious();
            }}
            disabled={selectedIndex === 0}
            className={cn(
              "absolute left-4 z-20",
              "p-2.5 rounded-full",
              "bg-muted/80 hover:bg-muted border border-border",
              "text-foreground transition-all duration-200",
              "hover:scale-105 active:scale-95",
              "disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:scale-100"
            )}
            aria-label="Previous image"
          >
            <ChevronLeft className="h-5 w-5" />
          </button>

          {/* Navigation: Next */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              goToNext();
            }}
            disabled={selectedIndex === filteredPaths.length - 1}
            className={cn(
              "absolute right-4 z-20",
              "p-2.5 rounded-full",
              "bg-muted/80 hover:bg-muted border border-border",
              "text-foreground transition-all duration-200",
              "hover:scale-105 active:scale-95",
              "disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:scale-100"
            )}
            aria-label="Next image"
          >
            <ChevronRight className="h-5 w-5" />
          </button>

          {/* Main image */}
          <figure
            className="relative z-10 max-w-[90vw] max-h-[85vh] flex flex-col items-center"
            onClick={(e) => e.stopPropagation()}
          >
            <LoadingImage
              src={`${cathedralPluginConfig.contentPrefix}/${filteredPaths[selectedIndex]}`}
              alt={getImageLabel(filteredPaths[selectedIndex])}
              className="max-h-[80vh] max-w-full object-contain rounded-lg shadow-2xl border border-border/50"
              wrapperClassName="max-h-[80vh] max-w-full flex items-center justify-center min-w-[200px] min-h-[200px]"
            />
            <figcaption className="mt-4 text-sm text-muted-foreground font-mono text-center">
              {getImageLabel(filteredPaths[selectedIndex])}
            </figcaption>
          </figure>

          {/* Counter */}
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 flex items-center gap-3">
            <div className="bg-muted/80 backdrop-blur-sm border border-border rounded-full px-4 py-2 flex items-center gap-2">
              <span className="font-mono text-sm text-foreground tabular-nums">
                {selectedIndex + 1}
              </span>
              <span className="text-muted-foreground/50">/</span>
              <span className="font-mono text-sm text-muted-foreground tabular-nums">
                {filteredPaths.length}
              </span>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
