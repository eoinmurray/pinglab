import { useCallback, useEffect, useRef, useState } from "react";
import { FULLSCREEN_DATA_ATTR } from "@/lib/constants";
import { useDirectory, useFileContent } from "../../plugins/cathedral-plugin/src/client";
import { useParams, useSearchParams } from "react-router-dom"
import Loading from "@/components/Loading";
import { FrontMatter } from "@/components/FrontMatter";
import { RuntimeMDX } from "@/components/RuntimeMDX";
import { RunningBar } from "@/components/RunningBar";
import { Header } from "@/components/Header";

export function parseSlideContent(content: string): string[] {
  // Strip frontmatter if present (starts with --- and ends with ---)
  let processedContent = content;
  const frontmatterMatch = content.match(/^---\n[\s\S]*?\n---\n/);
  if (frontmatterMatch) {
    processedContent = content.slice(frontmatterMatch[0].length);
  }

  return processedContent
    .split(/^---$/m)
    .map((slide) => slide.trim())
    .filter((slide) => slide.length > 0);
}


export function SlidesPage() {
  const { "path": path = "." } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const filePath = `${path}/SLIDES.mdx`
  const { file } = useDirectory(filePath)
  const { content, loading, error } = useFileContent(filePath);
  const slides = content ? parseSlideContent(content) : [];

  // Total slides = 1 (title) + content slides
  const totalSlides = slides.length + 1;

  // Get initial slide from query param
  const initialSlide = Math.max(0, Math.min(
    parseInt(searchParams.get("slide") || "0", 10),
    totalSlides - 1
  ));

  const [currentSlide, setCurrentSlide] = useState(initialSlide);
  const slideRefs = useRef<(HTMLDivElement | null)[]>([]);

  // Scroll to slide and update query param
  const goToSlide = useCallback((index: number) => {
    const clampedIndex = Math.max(0, Math.min(index, totalSlides - 1));
    const target = slideRefs.current[clampedIndex];
    if (target) {
      const targetTop = target.getBoundingClientRect().top + window.scrollY;
      window.scrollTo({ top: targetTop, behavior: "smooth" });
    }
  }, [totalSlides]);

  const goToPrevious = useCallback(() => {
    goToSlide(currentSlide - 1);
  }, [currentSlide, goToSlide]);

  const goToNext = useCallback(() => {
    goToSlide(currentSlide + 1);
  }, [currentSlide, goToSlide]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "ArrowUp" || e.key === "k") {
        e.preventDefault();
        goToPrevious();
      } else if (e.key === "ArrowDown" || e.key === "j") {
        e.preventDefault();
        goToNext();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [goToPrevious, goToNext]);

  // Update query param on scroll (delayed to avoid interference on load)
  useEffect(() => {
    let observer: IntersectionObserver | null = null;

    const timeoutId = setTimeout(() => {
      observer = new IntersectionObserver(
        (entries) => {
          for (const entry of entries) {
            if (entry.isIntersecting) {
              const index = slideRefs.current.findIndex((ref) => ref === entry.target);
              if (index !== -1) {
                setCurrentSlide(index);
                setSearchParams(index > 0 ? { slide: String(index) } : {}, { replace: true });
              }
            }
          }
        },
        { threshold: 0.5 }
      );

      slideRefs.current.forEach((ref) => {
        if (ref) observer!.observe(ref);
      });
    }, 100);

    return () => {
      clearTimeout(timeoutId);
      observer?.disconnect();
    };
  }, [slides.length, setSearchParams]);

  if (loading) {
    return (
      <Loading />
    )
  }

  if (error) {
    return (
      <main className="min-h-screen bg-background container mx-auto max-w-4xl py-12">
        <p className="text-center text-red-600">{error}</p>
      </main>
    )
  }

  if (slides.length === 0) {
    return (
      <div className="flex items-center justify-center p-12 text-muted-foreground font-mono text-sm">
        no slides found — use "---" to separate slides
      </div>
    );
  }

  return (
    <main className="slides-container">
      <title>{file?.frontmatter?.title}</title>
      <RunningBar />
      <Header
        slideControls={{
          current: currentSlide,
          total: totalSlides,
          onPrevious: goToPrevious,
          onNext: goToNext,
        }}
      />
      <div {...{[FULLSCREEN_DATA_ATTR]: "true"}}>
        <div
          ref={(el) => { slideRefs.current[0] = el; }}
          className="slide-page max-w-xl min-h-[50vh] sm:min-h-[70vh] md:min-h-screen flex items-center justify-center py-8 sm:py-12 md:py-16 px-4 mx-auto"
        >
          {file && (
            <FrontMatter
              title={file.frontmatter?.title}
              date={file.frontmatter?.date}
              description={file.frontmatter?.description}
            />
          )}
        </div>
        <hr />
        {slides.map((content, index) => (
          <div key={index}>
            <div
              ref={(el) => { slideRefs.current[index + 1] = el; }}
              className="slide-page min-h-[50vh] sm:min-h-[70vh] md:min-h-screen flex items-center justify-center py-8 sm:py-12 md:py-16 px-4 mx-auto"
            >
              {content && <RuntimeMDX content={content} size="md" />}
            </div>
            {index < slides.length - 1 && <hr />}
          </div>
        ))}
      </div>
    </main>
  )
}
