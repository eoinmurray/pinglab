import { useState, useMemo, useCallback, ImgHTMLAttributes } from "react";
import { X, ChevronLeft, ChevronRight, Image, Expand } from "lucide-react";
import { cn } from "@/lib/utils";
import { FULLSCREEN_DATA_ATTR } from "@/lib/constants";
import { useTheme } from "next-themes";
import { useDirectory } from "../../plugins/cathedral-plugin/src/client";
import { FileEntry } from "../../plugins/cathedral-plugin/src/lib";
import { cathedralPluginConfig } from "../../cathedral-plugin.config";
import { minimatch } from "minimatch";
import { useParams } from "react-router-dom";
import { useKeyBindings } from "@/hooks/useKeyBindings";

function LoadingImage({
  className,
  wrapperClassName,
  ...props
}: ImgHTMLAttributes<HTMLImageElement> & { wrapperClassName?: string }) {
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);

  return (
    <div className={cn("relative", wrapperClassName)}>
      {/* Loading skeleton - subtle pulse */}
      {isLoading && !hasError && (
        <div className="absolute inset-0 bg-muted/30 animate-pulse flex items-center justify-center">
          <div className="w-8 h-8 border border-border/50 rounded-sm" />
        </div>
      )}
      {/* Error state */}
      {hasError && (
        <div className="absolute inset-0 bg-muted/20 flex items-center justify-center">
          <div className="text-center">
            <Image className="h-5 w-5 text-muted-foreground/40 mx-auto" />
            <span className="text-xs text-muted-foreground/40 mt-1.5 block font-mono">failed</span>
          </div>
        </div>
      )}
      <img
        {...props}
        className={cn(
          className,
          "transition-opacity duration-500 ease-out-expo",
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

  // Museum-style grid: generous spacing, square aspect ratio for plots
  const gridConfig = useMemo(() => {
    const count = filteredPaths.length;
    if (count === 1) return { cols: "grid-cols-1", maxWidth: "max-w-md", gap: "gap-0" };
    if (count === 2) return { cols: "grid-cols-2", maxWidth: "max-w-[var(--gallery-width)]", gap: "gap-6 md:gap-8" };
    if (count === 3) return { cols: "grid-cols-2 md:grid-cols-3", maxWidth: "max-w-[var(--gallery-width)]", gap: "gap-5 md:gap-6" };
    if (count === 4) return { cols: "grid-cols-2", maxWidth: "max-w-[var(--gallery-width)]", gap: "gap-5 md:gap-6" };
    if (count <= 6) return { cols: "grid-cols-2 md:grid-cols-3", maxWidth: "max-w-[var(--gallery-width)]", gap: "gap-4 md:gap-5" };
    return { cols: "grid-cols-2 md:grid-cols-3 lg:grid-cols-4", maxWidth: "max-w-[var(--gallery-width)]", gap: "gap-4" };
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

  // Loading state - minimal skeleton
  if (!directory) {
    return (
      <div className="not-prose py-12">
        <div className="grid grid-cols-2 md:grid-cols-3 gap-6 max-w-[var(--gallery-width)] mx-auto">
          {[...Array(3)].map((_, i) => (
            <div
              key={i}
              className="aspect-[4/3] bg-muted/30 animate-pulse"
              style={{ animationDelay: `${i * 100}ms` }}
            />
          ))}
        </div>
      </div>
    );
  }

  // Empty state
  if (filteredPaths.length === 0) {
    return (
      <div className="not-prose py-16 text-center">
        <div className="inline-flex items-center gap-3 text-muted-foreground/60">
          <Image className="h-4 w-4" />
          <span className="font-mono text-sm tracking-wide">no images found</span>
        </div>
      </div>
    );
  }

  return (
    <>
      {/* Gallery Container - breaks out of content width for full impact */}
      <div className="not-prose relative -mx-[var(--page-padding)] md:-mx-[calc((var(--gallery-width)-var(--content-width))/2+var(--page-padding))] px-[var(--page-padding)] py-8 md:py-12">
        {/* Subtle background for gallery section */}
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-muted/20 to-transparent pointer-events-none" />

        {/* Grid */}
        <div
          className={cn(
            "relative mx-auto stagger-children",
            !single && "grid",
            !single && gridConfig.cols,
            !single && gridConfig.maxWidth,
            !single && gridConfig.gap,
          )}
        >
          {single && filteredPaths.length > 0 && (
            <figure
              className="group relative max-w-2xl mx-auto cursor-pointer"
              onClick={() => setSelectedIndex(0)}
            >
              <div className="relative overflow-hidden bg-card shadow-sm hover:shadow-lg transition-shadow duration-500 ease-out-expo">
                <LoadingImage
                  src={`${cathedralPluginConfig.contentPrefix}/${filteredPaths[0]}`}
                  alt={getImageLabel(filteredPaths[0])}
                  className="w-full h-auto object-contain"
                  wrapperClassName="w-full"
                />
                {/* Hover overlay - minimal */}
                <div className="absolute inset-0 bg-foreground/0 group-hover:bg-foreground/5 transition-colors duration-300 flex items-center justify-center pointer-events-none">
                  <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-300 bg-background/90 backdrop-blur-sm px-3 py-1.5 shadow-sm">
                    <Expand className="h-3.5 w-3.5 text-foreground" />
                  </div>
                </div>
              </div>
              <figcaption className="mt-4 text-center">
                <span className="font-mono text-xs text-muted-foreground tracking-wide">
                  {getImageLabel(filteredPaths[0])}
                </span>
              </figcaption>
            </figure>
          )}

          {!single && filteredPaths.map((imgPath, index) => (
            <figure
              key={imgPath}
              className="group relative cursor-pointer"
              onClick={() => setSelectedIndex(index)}
            >
              <div className="relative aspect-square overflow-hidden bg-card shadow-sm hover:shadow-md transition-all duration-500 ease-out-expo">
                <LoadingImage
                  src={`${cathedralPluginConfig.contentPrefix}/${imgPath}`}
                  alt={getImageLabel(imgPath)}
                  className="w-full h-full object-contain transition-transform duration-700 ease-out-expo group-hover:scale-[1.02]"
                  wrapperClassName="w-full h-full"
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
            </figure>
          ))}
        </div>

        {/* Caption */}
        {caption && (
          <div className="mt-8 max-w-[var(--prose-width)] mx-auto text-center">
            <p className="text-sm text-muted-foreground italic">
              {caption}
            </p>
          </div>
        )}
      </div>

      {/* Fullscreen Lightbox - cinematic experience */}
      {selectedIndex !== null && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center"
          onClick={close}
          {...{[FULLSCREEN_DATA_ATTR]: "true"}}
        >
          {/* Backdrop - deep, immersive */}
          <div className="absolute inset-0 bg-background/98 backdrop-blur-xl animate-fade-in-slow" />

          {/* Close button - minimal, top-right */}
          <button
            onClick={close}
            className={cn(
              "absolute top-6 right-6 z-20",
              "p-3 transition-all duration-300",
              "text-muted-foreground hover:text-foreground",
              "hover:bg-muted/50"
            )}
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>

          {/* Navigation: Previous */}
          {selectedIndex > 0 && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                goToPrevious();
              }}
              className={cn(
                "absolute left-6 z-20",
                "p-3 transition-all duration-300",
                "text-muted-foreground hover:text-foreground",
                "hover:bg-muted/50"
              )}
              aria-label="Previous image"
            >
              <ChevronLeft className="h-6 w-6" />
            </button>
          )}

          {/* Navigation: Next */}
          {selectedIndex < filteredPaths.length - 1 && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                goToNext();
              }}
              className={cn(
                "absolute right-6 z-20",
                "p-3 transition-all duration-300",
                "text-muted-foreground hover:text-foreground",
                "hover:bg-muted/50"
              )}
              aria-label="Next image"
            >
              <ChevronRight className="h-6 w-6" />
            </button>
          )}

          {/* Main image - centered, maximum impact */}
          <figure
            className="relative z-10 max-w-[92vw] max-h-[88vh] flex flex-col items-center animate-slide-up"
            onClick={(e) => e.stopPropagation()}
          >
            <LoadingImage
              src={`${cathedralPluginConfig.contentPrefix}/${filteredPaths[selectedIndex]}`}
              alt={getImageLabel(filteredPaths[selectedIndex])}
              className="max-h-[82vh] max-w-full object-contain shadow-2xl"
              wrapperClassName="max-h-[82vh] max-w-full flex items-center justify-center min-w-[200px] min-h-[200px]"
            />
            <figcaption className="mt-6 text-center">
              <span className="font-mono text-sm text-muted-foreground tracking-wide">
                {getImageLabel(filteredPaths[selectedIndex])}
              </span>
            </figcaption>
          </figure>

          {/* Counter - bottom, minimal */}
          <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-20">
            <div className="flex items-center gap-2 font-mono text-xs text-muted-foreground tracking-widest">
              <span className="tabular-nums">{String(selectedIndex + 1).padStart(2, '0')}</span>
              <span className="text-border">/</span>
              <span className="tabular-nums">{String(filteredPaths.length).padStart(2, '0')}</span>
            </div>
          </div>

          {/* Keyboard hint */}
          <div className="absolute bottom-6 right-6 z-20 hidden md:flex items-center gap-3 text-muted-foreground/50">
            <span className="font-mono text-[10px] tracking-wider">ESC to close</span>
          </div>
        </div>
      )}
    </>
  );
}
