import { useCallback, useRef } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { RuntimeMDX } from "./RuntimeMDX";
import { FULLSCREEN_DATA_ATTR } from "@/lib/constants";
import { ChevronDown, ChevronUp, X } from "lucide-react";
import { useKeyBindings } from "@/hooks/useKeyBindings";
import { cn } from "@/lib/utils";

export function Slides({ content }: { content: string }) {
  const navigate = useNavigate();
  const location = useLocation();
  const containerRef = useRef<HTMLDivElement>(null);

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

  // Navigation handlers - scroll to next/prev slide
  const goToNextSlide = useCallback(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    const viewportHeight = container.clientHeight;
    const currentScroll = container.scrollTop;
    const currentSlide = Math.round(currentScroll / viewportHeight);
    const nextSlide = Math.min(currentSlide + 1, totalSlides - 1);
    container.scrollTo({ top: nextSlide * viewportHeight, behavior: "smooth" });
  }, [totalSlides]);

  const goToPreviousSlide = useCallback(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    const viewportHeight = container.clientHeight;
    const currentScroll = container.scrollTop;
    const currentSlide = Math.round(currentScroll / viewportHeight);
    const prevSlide = Math.max(currentSlide - 1, 0);
    container.scrollTo({ top: prevSlide * viewportHeight, behavior: "smooth" });
  }, []);

  const goToFirstSlide = useCallback(() => {
    if (!containerRef.current) return;
    containerRef.current.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const goToLastSlide = useCallback(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    const viewportHeight = container.clientHeight;
    container.scrollTo({ top: (totalSlides - 1) * viewportHeight, behavior: "smooth" });
  }, [totalSlides]);

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

  // Check if slides should handle keyboard
  const slidesCanHandleKeys = useCallback(() => {
    const fullscreenElements = document.querySelectorAll(`[${FULLSCREEN_DATA_ATTR}="true"]`);
    if (fullscreenElements.length > 1) return false;
    if (fullscreenElements.length === 1 && !fullscreenElements[0].classList.contains("slides-container")) return false;
    return true;
  }, []);

  useKeyBindings(
    [
      { key: ["ArrowRight", "ArrowDown", " ", "PageDown"], action: goToNextSlide },
      { key: ["ArrowLeft", "ArrowUp", "PageUp"], action: goToPreviousSlide },
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
      {/* Fixed navigation controls */}
      <div className="absolute top-4 right-6 z-10 flex items-center gap-1">
        <button
          onClick={goToPreviousSlide}
          className="p-2 transition-colors duration-200 flex items-center gap-2 text-muted-foreground hover:text-foreground"
          title="Previous slide"
        >
          <ChevronUp className="h-6 w-6" />
        </button>
        <button
          onClick={goToNextSlide}
          className="p-2 transition-colors duration-200 flex items-center gap-2 text-muted-foreground hover:text-foreground"
          title="Next slide"
        >
          <ChevronDown className="h-6 w-6" />
        </button>
      </div>

      {/* Scrollable slide container */}
      <div
        ref={containerRef}
        className="flex-1 overflow-y-auto"
      >
        {slides.map((slide, index) => (
          <div key={index}>
            <div className="min-h-screen w-full flex items-center justify-center text-lg md:text-2xl">
              <RuntimeMDX content={slide} />
            </div>
            {index < slides.length - 1 && (
              <div className="w-full h-px bg-border" />
            )}
          </div>
        ))}
      </div>

      {/* Keyboard hints - hidden on mobile */}
      <div className="absolute bottom-4 right-6 hidden md:flex items-center gap-4 text-muted-foreground/30 font-mono text-[10px] tracking-wider">
        <span>↑ ↓ navigate</span>
        <span>ESC exit</span>
      </div>
    </div>
  );
}
