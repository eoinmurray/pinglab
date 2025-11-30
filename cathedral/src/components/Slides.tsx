import { useState, useEffect, useCallback } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { RuntimeMDX } from "./RuntimeMDX";
import { Button } from "./ui/button";
import { FULLSCREEN_DATA_ATTR, isFullscreenActive } from "@/lib/constants";
import {
  ChevronLeft,
  ChevronRight,
  X,
} from "lucide-react";

export function Slides({ content }: { content: string }) {
  const navigate = useNavigate();
  const location = useLocation();
  const [currentSlide, setCurrentSlide] = useState(0);

  const slides = content
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
      pathParts.pop(); // Remove current file/directory
      const parentPath = pathParts.length > 0 ? `/${pathParts.join('/')}` : '/';
      navigate(parentPath);
    } else {
      navigate('/');
    }
  }, [location.pathname, navigate]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't handle keyboard events if another fullscreen component is active
      if (isFullscreenActive()) return;

      // Allow default browser behavior for modifier key combinations (Cmd/Ctrl + key)
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      switch (e.key) {
        case "ArrowRight":
        case " ":
        case "PageDown":
          e.preventDefault();
          goToNextSlide();
          break;
        case "ArrowLeft":
        case "PageUp":
          e.preventDefault();
          goToPreviousSlide();
          break;
        case "Home":
          e.preventDefault();
          goToFirstSlide();
          break;
        case "End":
          e.preventDefault();
          setCurrentSlide(totalSlides - 1);
          break;
        case "Escape":
          e.preventDefault();
          goToParentDir();
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    goToNextSlide,
    goToPreviousSlide,
    goToFirstSlide,
    goToParentDir,
    totalSlides,
  ]);

  if (totalSlides === 0) {
    return (
      <div className="flex items-center justify-center p-8 text-muted-foreground">
        No slides found. Use "---" to separate slides.
      </div>
    );
  }

  // Always render in fullscreen presentation mode
  return (
      <div className="slides-container fixed inset-0 z-50 bg-background flex flex-col" {...{[FULLSCREEN_DATA_ATTR]: "true"}}>
        {/* Controls */}
        <div className="slide-controls flex-shrink-0 flex items-center justify-between gap-4 bg-muted/50 backdrop-blur-sm border-b px-3 py-2">
          {/* Left controls */}
          <div />

          {/* Center - slide counter */}
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium tabular-nums">
              {currentSlide + 1} / {totalSlides}
            </span>
          </div>

          {/* Right controls */}
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={goToPreviousSlide}
              disabled={currentSlide === 0}
              title="Previous slide (←)"
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={goToNextSlide}
              disabled={currentSlide === totalSlides - 1}
              title="Next slide (→)"
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={goToParentDir}
              title="Go to parent directory (Esc)"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Main slide content */}
        <div className="slide-content flex-1 min-h-0 overflow-auto p-16">
          <div className="max-w-4xl mx-auto h-full flex items-center justify-center">
            <div className="w-full">
              <RuntimeMDX content={slides[currentSlide]} />
            </div>
          </div>
        </div>
      </div>
  );
}
