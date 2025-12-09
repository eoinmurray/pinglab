import { createPortal } from "react-dom";
import { X, ChevronLeft, ChevronRight } from "lucide-react";
import { FULLSCREEN_DATA_ATTR } from "@/lib/constants";

export interface LightboxImage {
  src: string;
  label: string;
}

export interface LightboxProps {
  images: LightboxImage[];
  selectedIndex: number;
  onClose: () => void;
  onPrevious: () => void;
  onNext: () => void;
  showNavigation?: boolean;
}

/**
 * Fullscreen lightbox component for viewing images
 */
export function Lightbox({
  images,
  selectedIndex,
  onClose,
  onPrevious,
  onNext,
  showNavigation = true,
}: LightboxProps) {
  const current = images[selectedIndex];

  return createPortal(
    <div
      className="fixed inset-0 z-[9999] bg-background"
      onClick={onClose}
      {...{ [FULLSCREEN_DATA_ATTR]: "true" }}
      style={{ top: 0, left: 0, right: 0, bottom: 0 }}
    >
      {/* Top bar */}
      <div
        className="fixed top-0 left-0 right-0 z-10 flex items-center justify-between px-4 py-3 bg-background/80 backdrop-blur-sm"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="font-mono text-xs text-muted-foreground tabular-nums">
          {String(selectedIndex + 1).padStart(2, '0')} / {String(images.length).padStart(2, '0')}
        </div>
        <button
          onClick={onClose}
          className="p-2 text-muted-foreground hover:text-foreground transition-colors"
          aria-label="Close"
        >
          <X className="h-5 w-5" />
        </button>
      </div>

      {/* Navigation: Previous */}
      {showNavigation && selectedIndex > 0 && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onPrevious();
          }}
          className="fixed left-4 top-1/2 -translate-y-1/2 z-10 p-2 text-muted-foreground hover:text-foreground transition-colors"
          aria-label="Previous image"
        >
          <ChevronLeft className="h-8 w-8" />
        </button>
      )}

      {/* Navigation: Next */}
      {showNavigation && selectedIndex < images.length - 1 && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onNext();
          }}
          className="fixed right-4 top-1/2 -translate-y-1/2 z-10 p-2 text-muted-foreground hover:text-foreground transition-colors"
          aria-label="Next image"
        >
          <ChevronRight className="h-8 w-8" />
        </button>
      )}

      {/* Main image */}
      <div className="fixed inset-0 flex items-center justify-center p-16">
        <img
          src={current.src}
          alt={current.label}
          className="max-w-full max-h-full object-contain"
          onClick={(e) => e.stopPropagation()}
        />
      </div>

      {/* Caption */}
      <div
        className="fixed bottom-0 left-0 right-0 z-10 p-4 text-center bg-background/80 backdrop-blur-sm"
        onClick={(e) => e.stopPropagation()}
      >
        <span className="font-mono text-xs text-muted-foreground">
          {current.label}
        </span>
      </div>
    </div>,
    document.body
  );
}
