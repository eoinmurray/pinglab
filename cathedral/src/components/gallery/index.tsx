import { useMemo } from "react";
import { Image, Expand } from "lucide-react";
import { cn } from "@/lib/utils";
import { cathedralPluginConfig } from "../../../cathedral-plugin.config";
import { Lightbox, LightboxImage } from "@/components/gallery/components/Lightbox";
import { useGalleryImages } from "./hooks/use-gallery-images";
import { useLightbox } from "./hooks/use-lightbox";
import { LoadingImage } from "./components/LoadingImage";
import { FigureHeader } from "./components/FigureHeader";
import { FigureCaption } from "./components/FigureCaption";

function getImageLabel(path: string): string {
  const filename = path.split('/').pop() || path;
  return filename
    .replace(/\.(png|jpg|jpeg|gif|svg|webp)$/i, '')
    .replace(/[_-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function getImageUrl(path: string): string {
  return `${cathedralPluginConfig.contentPrefix}/${path}`;
}

export default function Gallery({
  path,
  globs = null, 
  caption,
  captionLabel,
  title,
  subtitle,
  limit,
}: {
  path?: string;
  globs?: string[] | null;
  caption?: string;
  captionLabel?: string;
  title?: string;
  subtitle?: string;
  limit?: number;
}) {
  const { paths, isLoading, isEmpty } = useGalleryImages({
    path,
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
    const rows = count <= 3 ? 1 : count <= 6 ? 2 : Math.ceil(count / 4);
    // Calculate height per image based on rows (leaving room for header/caption)
    const imgH = rows === 1 ? "h-[45vh]" : rows === 2 ? "h-[28vh]" : "h-[20vh]";

    if (count === 1) return { cols: "grid-cols-1", maxWidth: "max-w-md", gap: "gap-0", imgH };
    if (count === 2) return { cols: "grid-cols-2", maxWidth: "max-w-[var(--gallery-width)]", gap: "gap-2 sm:gap-4 md:gap-6", imgH };
    if (count === 3) return { cols: "grid-cols-3", maxWidth: "max-w-[var(--gallery-width)]", gap: "gap-2 sm:gap-4 md:gap-5", imgH };
    if (count === 4) return { cols: "grid-cols-2", maxWidth: "max-w-[var(--gallery-width)]", gap: "gap-2 sm:gap-3 md:gap-4", imgH };
    if (count <= 6) return { cols: "grid-cols-3", maxWidth: "max-w-[var(--gallery-width)]", gap: "gap-2 sm:gap-3", imgH };
    return { cols: "grid-cols-3 md:grid-cols-4", maxWidth: "max-w-[var(--gallery-width)]", gap: "gap-1 sm:gap-2", imgH };
  }, [paths.length]);

  if (isLoading) {
    return (
      <div className="not-prose py-4 md:py-6">
        <div className="grid grid-cols-3 gap-2 sm:gap-4 max-w-[var(--gallery-width)] mx-auto">
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
      <div className="not-prose relative -mx-[var(--page-padding)] md:-mx-[calc((var(--gallery-width)-var(--content-width))/2+var(--page-padding))] px-[var(--page-padding)] py-4 md:py-6">
        <div className="absolute inset-0 pointer-events-none" />

        <FigureHeader title={title} subtitle={subtitle} />

        <div
          className={cn(
            "relative mx-auto stagger-children",
            "grid",
            gridConfig.cols,
            gridConfig.maxWidth,
            gridConfig.gap,
          )}
        >
          {images.map((img, index) => (
            <figure
              key={paths[index]}
              className="group relative cursor-pointer"
              onClick={() => lightbox.open(index)}
            >
              <div className={cn("relative overflow-hidden transition-all duration-500 ease-out-expo", gridConfig.imgH)}>
                <LoadingImage
                  src={img.src}
                  alt={img.label}
                  className="w-full h-full object-contain transition-transform duration-700 ease-out-expo group-hover:scale-[1.02]"
                  wrapperClassName="w-full h-full"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-foreground/0 via-transparent to-foreground/0 group-hover:from-foreground/10 transition-all duration-500 pointer-events-none" />

                <div className="absolute bottom-3 left-3 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                  <span className="font-mono text-[10px] text-white/90 bg-foreground/60 backdrop-blur-sm px-1.5 py-0.5 tracking-wider">
                    {String(index + 1).padStart(2, '0')}
                  </span>
                </div>

                <div className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                  <div className="bg-background/80 backdrop-blur-sm p-1.5 shadow-sm">
                    <Expand className="h-3 w-3 text-foreground" />
                  </div>
                </div>
              </div>
            </figure>
          ))}
        </div>

        <div className="max-w-lg mx-auto mt-3">
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
