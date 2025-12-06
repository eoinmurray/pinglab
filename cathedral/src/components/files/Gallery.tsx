import { useState, useMemo, useCallback, ImgHTMLAttributes, ReactNode } from "react";
import { Image, Expand } from "lucide-react";
import { cn } from "@/lib/utils";
import { useTheme } from "next-themes";
import { useDirectory } from "../../../plugins/cathedral-plugin/src/client";
import { FileEntry } from "../../../plugins/cathedral-plugin/src/lib";
import { cathedralPluginConfig } from "../../../cathedral-plugin.config";
import { minimatch } from "minimatch";
import { useParams } from "react-router-dom";
import { useKeyBindings } from "@/hooks/useKeyBindings";
import { Lightbox, LightboxImage } from "@/components/Lightbox";
import katex from "katex";

function renderMathInText(text: string): ReactNode {
  // Match $...$ for inline math and $$...$$ for display math
  const parts: ReactNode[] = [];
  let lastIndex = 0;
  // Match display math ($$...$$) first, then inline math ($...$)
  const regex = /\$\$([^$]+)\$\$|\$([^$]+)\$/g;
  let match;
  let key = 0;

  while ((match = regex.exec(text)) !== null) {
    // Add text before this match
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }

    const isDisplay = match[1] !== undefined;
    const mathContent = match[1] || match[2];

    try {
      const html = katex.renderToString(mathContent, {
        displayMode: isDisplay,
        throwOnError: false,
      });
      parts.push(
        <span
          key={key++}
          dangerouslySetInnerHTML={{ __html: html }}
        />
      );
    } catch {
      // If KaTeX fails, just show the original text
      parts.push(match[0]);
    }

    lastIndex = match.index + match[0].length;
  }

  // Add remaining text
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts.length === 1 && typeof parts[0] === 'string' ? parts[0] : <>{parts}</>;
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

function sortPathsNumerically(paths: string[]): void {
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
}

function getImageUrl(path: string): string {
  return `${cathedralPluginConfig.contentPrefix}/${path}`;
}

// --- Hooks ---

function useGalleryImages({
  path,
  relativePath,
  globs = null,
  limit,
}: {
  path?: string;
  relativePath?: string;
  globs?: string[] | null;
  limit?: number;
}) {
  const { "*": paramPath = "." } = useParams();
  const { resolvedTheme } = useTheme();

  let resolvedPath = path;
  if (relativePath) {
    const basePath = paramPath === "." ? "" : paramPath;
    resolvedPath = basePath + (basePath.endsWith("/") ? "" : "/") + relativePath;
  }

  const { directory } = useDirectory(resolvedPath);

  const paths = useMemo(() => {
    if (!directory) return [];

    const imageChildren = directory.children.filter((child): child is FileEntry => {
      return !!child.name.match(/\.(png|jpeg|gif|svg|webp)$/i) && child.type === "file";
    });

    let imagePaths = imageChildren.map(child => child.path);

    if (globs && globs.length > 0) {
      imagePaths = imagePaths.filter(p => {
        return globs.some(glob => minimatch(p.split('/').pop() || '', glob));
      });
    }

    sortPathsNumerically(imagePaths);
    let filtered = filterPathsByTheme(imagePaths, resolvedTheme);

    if (limit) {
      filtered = filtered.slice(0, limit);
    }

    return filtered;
  }, [directory, globs, resolvedTheme, limit]);

  return {
    paths,
    isLoading: !directory,
    isEmpty: directory !== undefined && paths.length === 0,
  };
}

function useLightbox(totalImages: number) {
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  const open = useCallback((index: number) => {
    setSelectedIndex(index);
  }, []);

  const close = useCallback(() => {
    setSelectedIndex(null);
  }, []);

  const goToPrevious = useCallback(() => {
    setSelectedIndex(prev => prev !== null && prev > 0 ? prev - 1 : prev);
  }, []);

  const goToNext = useCallback(() => {
    setSelectedIndex(prev => prev !== null && prev < totalImages - 1 ? prev + 1 : prev);
  }, [totalImages]);

  useKeyBindings(
    [
      { key: "Escape", action: close },
      { key: "ArrowLeft", action: goToPrevious },
      { key: "ArrowRight", action: goToNext },
    ],
    { enabled: () => selectedIndex !== null }
  );

  return {
    selectedIndex,
    open,
    close,
    goToPrevious,
    goToNext,
    isOpen: selectedIndex !== null,
  };
}

// --- Components ---

function LoadingImage({
  className,
  wrapperClassName,
  ...props
}: ImgHTMLAttributes<HTMLImageElement> & { wrapperClassName?: string }) {
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);

  return (
    <div className={cn("relative", wrapperClassName)}>
      {isLoading && !hasError && (
        <div className="absolute inset-0 bg-muted/30 animate-pulse flex items-center justify-center print:hidden">
          <div className="w-8 h-8 border border-border/50 rounded-sm" />
        </div>
      )}
      {hasError && (
        <div className="absolute inset-0 bg-muted/20 flex items-center justify-center print:hidden">
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
          "transition-opacity duration-500 ease-out-expo print:opacity-100 print:transition-none",
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

function FigureHeader({ title, subtitle }: { title?: string; subtitle?: string }) {
  if (!title && !subtitle) return null;

  return (
    <div className="max-w-lg mx-auto mb-4">
      {title && (
        <h3 className="text-sm md:text-base font-medium tracking-tight text-foreground text-left">
          {renderMathInText(title)}
        </h3>
      )}
      {subtitle && (
        <p className="text-sm text-muted-foreground leading-relaxed text-left mt-1">
          {renderMathInText(subtitle)}
        </p>
      )}
    </div>
  );
}

function FigureCaption({ caption, label }: { caption?: string; label?: string }) {
  if (!caption && !label) return null;

  return (
    <div className="mx-auto mx-w-md">
      <p className="text-sm text-muted-foreground leading-relaxed text-left">
        {label && <span className="font-medium text-foreground">{label}</span>}
        {label && caption && <span className="mx-1">—</span>}
        {caption && renderMathInText(caption)}
      </p>
    </div>
  );
}

export default function Gallery({
  path,
  relativePath,
  caption,
  captionLabel,
  title,
  subtitle,
  globs = null,
  single = false,
  limit,
}: {
  path?: string;
  relativePath?: string;
  caption?: string;
  captionLabel?: string;
  title?: string;
  subtitle?: string;
  globs?: string[] | null;
  single?: boolean;
  limit?: number;
}) {
  const { paths, isLoading, isEmpty } = useGalleryImages({
    path,
    relativePath,
    globs,
    limit,
  });

  const lightbox = useLightbox(paths.length);

  const images: LightboxImage[] = useMemo(() =>
    paths.map(p => ({ src: getImageUrl(p), label: getImageLabel(p) })),
    [paths]
  );

  const gridConfig = useMemo(() => {
    const count = paths.length;
    if (count === 1) return { cols: "grid-cols-1", maxWidth: "max-w-md", gap: "gap-0" };
    if (count === 2) return { cols: "grid-cols-2", maxWidth: "max-w-[var(--gallery-width)]", gap: "gap-6 md:gap-8" };
    if (count === 3) return { cols: "grid-cols-2 md:grid-cols-3", maxWidth: "max-w-[var(--gallery-width)]", gap: "gap-5 md:gap-6" };
    if (count === 4) return { cols: "grid-cols-2", maxWidth: "max-w-[var(--gallery-width)]", gap: "gap-5 md:gap-6" };
    if (count <= 6) return { cols: "grid-cols-2 md:grid-cols-3", maxWidth: "max-w-[var(--gallery-width)]", gap: "gap-4 md:gap-5" };
    return { cols: "grid-cols-2 md:grid-cols-3 lg:grid-cols-4", maxWidth: "max-w-[var(--gallery-width)]", gap: "gap-4" };
  }, [paths.length]);

  if (isLoading) {
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

  if (isEmpty) {
    return (
      <div className="not-prose py-16 text-center">
        <div className="inline-flex items-center gap-3 text-muted-foreground/60">
          <Image className="h-4 w-4" />
          <span className="font-mono text-sm tracking-wide">no image(s) found</span>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="not-prose relative -mx-[var(--page-padding)] md:-mx-[calc((var(--gallery-width)-var(--content-width))/2+var(--page-padding))] px-[var(--page-padding)] py-8 md:py-12 print:mx-0 print:px-0 print:py-4">
        <div className="absolute inset-0 pointer-events-none" />

        <FigureHeader title={title} subtitle={subtitle} />

        <div
          className={cn(
            "relative mx-auto stagger-children",
            !single && "grid",
            !single && gridConfig.cols,
            !single && gridConfig.maxWidth,
            !single && gridConfig.gap,
          )}
        >
          {single && images.length > 0 && (
            <figure
              className="group relative mx-auto cursor-pointer max-w-md"
              onClick={() => lightbox.open(0)}
            >
              <div className="relative overflow-hidden bg-card duration-500 ease-out-expo">
                <LoadingImage
                  src={images[0].src}
                  alt={images[0].label}
                  className="w-full h-auto object-contain"
                  wrapperClassName="w-full flex items-center justify-center"
                />
                <div className="absolute inset-0 bg-foreground/0 group-hover:bg-foreground/5 transition-colors duration-300 flex items-center justify-center pointer-events-none print:hidden">
                  <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-300 bg-background/90 backdrop-blur-sm px-3 py-1.5 shadow-sm">
                    <Expand className="h-3.5 w-3.5 text-foreground" />
                  </div>
                </div>
              </div>
              <figcaption className="mt-4 text-center">
                <span className="font-mono text-xs text-muted-foreground tracking-wide">
                  {images[0].label}
                </span>
              </figcaption>
            </figure>
          )}

          {!single && images.map((img, index) => (
            <figure
              key={paths[index]}
              className="group relative cursor-pointer"
              onClick={() => lightbox.open(index)}
            >
              <div className="relative aspect-square overflow-hidden bg-card transition-all duration-500 ease-out-expo">
                <LoadingImage
                  src={img.src}
                  alt={img.label}
                  className="w-full h-full object-contain transition-transform duration-700 ease-out-expo group-hover:scale-[1.02]"
                  wrapperClassName="w-full h-full"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-foreground/0 via-transparent to-foreground/0 group-hover:from-foreground/10 transition-all duration-500 pointer-events-none print:hidden" />

                <div className="absolute bottom-3 left-3 opacity-0 group-hover:opacity-100 transition-opacity duration-300 print:hidden">
                  <span className="font-mono text-[10px] text-white/90 bg-foreground/60 backdrop-blur-sm px-1.5 py-0.5 tracking-wider">
                    {String(index + 1).padStart(2, '0')}
                  </span>
                </div>

                <div className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity duration-300 print:hidden">
                  <div className="bg-background/80 backdrop-blur-sm p-1.5 shadow-sm">
                    <Expand className="h-3 w-3 text-foreground" />
                  </div>
                </div>
              </div>
            </figure>
          ))}
        </div>

        <div className="max-w-lg mx-auto mt-6">
          <FigureCaption caption={caption} label={captionLabel} />
        </div>
      </div>

      {lightbox.isOpen && lightbox.selectedIndex !== null && (
        <Lightbox
          images={images}
          selectedIndex={lightbox.selectedIndex}
          onClose={lightbox.close}
          onPrevious={lightbox.goToPrevious}
          onNext={lightbox.goToNext}
        />
      )}
    </>
  );
}
