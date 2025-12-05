import { useState, useCallback } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { RuntimeMDX } from "./RuntimeMDX";
import { FULLSCREEN_DATA_ATTR } from "@/lib/constants";
import { ChevronLeft, ChevronRight, X } from "lucide-react";
import { useKeyBindings } from "@/hooks/useKeyBindings";
import { cn } from "@/lib/utils";

export function Slides({ content }: { content: string }) {
  const navigate = useNavigate();
  const location = useLocation();
  const [currentSlide, setCurrentSlide] = useState(0);

  // Strip frontmatter if present (starts with --- and ends with ---)
  let processedContent = content;
  const frontmatterMatch = content.match(/^---\n[\s\S]*?\n---\n/);
  if (frontmatterMatch) {
    processedContent = content.slice(frontmatterMatch[0].length);
  }

  const slides = processedContent
    .split(/^---$/m)
    .map((slide) => slide.trim())
    .filter((slide) => slide.length > 0);

  const totalSlides = slides.length;

  // Navigation handlers
  const goToNextSlide = useCallback(() => {
    setCurrentSlide((prev) => Math.min(prev + 1, totalSlides - 1));
  }, [totalSlides]);

  const goToPreviousSlide = useCallback(() => {
    setCurrentSlide((prev) => Math.max(prev - 1, 0));
  }, []);

  const goToFirstSlide = useCallback(() => {
    setCurrentSlide(0);
  }, []);

  // Go to parent directory
  const goToParentDir = useCallback(() => {
    const pathParts = location.pathname.split('/').filter(Boolean);
    if (pathParts.length > 0) {
      pathParts.pop();
      const parentPath = pathParts.length > 0 ? `/${pathParts.join('/')}` : '/';
      navigate(parentPath);
    } else {
      navigate('/');
    }
  }, [location.pathname, navigate]);

  const goToLastSlide = useCallback(() => {
    setCurrentSlide(totalSlides - 1);
  }, [totalSlides]);

  // Check if slides should handle keyboard
  const slidesCanHandleKeys = useCallback(() => {
    const fullscreenElements = document.querySelectorAll(`[${FULLSCREEN_DATA_ATTR}="true"]`);
    if (fullscreenElements.length > 1) return false;
    if (fullscreenElements.length === 1 && !fullscreenElements[0].classList.contains("slides-container")) return false;
    return true;
  }, []);

  useKeyBindings(
    [
      { key: ["ArrowRight", " ", "PageDown"], action: goToNextSlide },
      { key: ["ArrowLeft", "PageUp"], action: goToPreviousSlide },
      { key: "Home", action: goToFirstSlide },
      { key: "End", action: goToLastSlide },
      { key: "Escape", action: goToParentDir },
    ],
    { enabled: slidesCanHandleKeys }
  );

  if (totalSlides === 0) {
    return (
      <div className="flex items-center justify-center p-12 text-muted-foreground font-mono text-sm">
        no slides found — use "---" to separate slides
      </div>
    );
  }

  return (
    <div
      className="slides-container fixed inset-0 bg-background flex flex-col noise-overlay"
      {...{[FULLSCREEN_DATA_ATTR]: "true"}}
    >
      {/* Minimal top bar */}
      <div className="flex-shrink-0 flex items-center justify-between px-6 py-4 border-b border-border/20">
        {/* Slide counter - left */}
        <div className="font-mono text-xs text-muted-foreground tracking-widest tabular-nums">
          <span>{String(currentSlide + 1).padStart(2, '0')}</span>
          <span className="text-border mx-2">/</span>
          <span>{String(totalSlides).padStart(2, '0')}</span>
        </div>

        {/* Navigation controls - right */}
        <div className="flex items-center gap-1">
          <button
            onClick={goToPreviousSlide}
            disabled={currentSlide === 0}
            className={cn(
              "p-2 transition-colors duration-200",
              currentSlide === 0
                ? "text-muted-foreground/30 cursor-not-allowed"
                : "text-muted-foreground hover:text-foreground"
            )}
            title="Previous slide"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          <button
            onClick={goToNextSlide}
            disabled={currentSlide === totalSlides - 1}
            className={cn(
              "p-2 transition-colors duration-200",
              currentSlide === totalSlides - 1
                ? "text-muted-foreground/30 cursor-not-allowed"
                : "text-muted-foreground hover:text-foreground"
            )}
            title="Next slide"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
          <div className="w-px h-4 bg-border/30 mx-2" />
          <button
            onClick={goToParentDir}
            className="p-2 text-muted-foreground hover:text-foreground transition-colors duration-200"
            title="Exit (Esc)"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Main slide content - centered, generous padding */}
      <div className="flex-1 min-h-0 overflow-auto">
        <div className="min-h-full flex items-center justify-center p-12 md:p-20">
          <div className="max-w-4xl w-full animate-fade-in" key={currentSlide}>
            <article className="prose dark:prose-invert prose-headings:tracking-tight prose-lg md:prose-xl max-w-none">
              <RuntimeMDX content={slides[currentSlide]} />
            </article>
          </div>
        </div>
      </div>

      {/* Progress bar - bottom */}
      <div className="flex-shrink-0 h-0.5 bg-border/20">
        <div
          className="h-full bg-primary/50 transition-all duration-300 ease-out-expo"
          style={{ width: `${((currentSlide + 1) / totalSlides) * 100}%` }}
        />
      </div>

      {/* Keyboard hints - hidden on mobile */}
      <div className="absolute bottom-4 right-6 hidden md:flex items-center gap-4 text-muted-foreground/30 font-mono text-[10px] tracking-wider">
        <span>← → navigate</span>
        <span>ESC exit</span>
      </div>
    </div>
  );
}
