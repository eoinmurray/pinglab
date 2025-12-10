import { useMemo } from "react";
import { Image } from "lucide-react";
import { cathedralPluginConfig } from "../../../cathedral-plugin.config";
import { Lightbox, LightboxImage } from "@/components/gallery/components/Lightbox";
import { useGalleryImages } from "./hooks/use-gallery-images";
import { useLightbox } from "./hooks/use-lightbox";
import { LoadingImage } from "./components/LoadingImage";
import { FigureHeader } from "./components/FigureHeader";
import { FigureCaption } from "./components/FigureCaption";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
} from "@/components/ui/carousel"

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
  limit = null,
  page = 0,
}: {
  path?: string;
  globs?: string[] | null;
  caption?: string;
  captionLabel?: string;
  title?: string;
  subtitle?: string;
  limit?: number | null;
  page?: number;
}) {
  const { paths, isLoading, isEmpty } = useGalleryImages({
    path,
    globs,
    limit,
    page: page,
  });

  const lightbox = useLightbox(paths.length);

  const images: LightboxImage[] = useMemo(() =>
    paths.map(p => ({ src: getImageUrl(p), label: getImageLabel(p) })),
    [paths]
  );

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
      <div className="rounded-lg not-prose flex flex-col gap-0 relative p-4 -mx-[calc((var(--gallery-width)-var(--content-width))/2+var(--page-padding))]">
        <FigureHeader title={title} subtitle={subtitle} />

        <Carousel>
          <CarouselContent>
            {images.map((img, index) => (
              <CarouselItem 
                key={index} 
                className="mx-auto md:basis-1/2 lg:basis-1/3 cursor-pointer transition-transform duration-700 ease-out-expo hover:scale-[1.02]"
                // wrapperClassName="h-full"
                onClick={() => lightbox.open(index)}
              >
                <LoadingImage
                  src={img.src}
                  alt={img.label}
                />
              </CarouselItem>
            ))}
          </CarouselContent>

          {images.length > 3 && (
            <>
              <CarouselPrevious />
              <CarouselNext />
            </>
          )}          
        </Carousel>

        <FigureCaption caption={caption} label={captionLabel} />
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
