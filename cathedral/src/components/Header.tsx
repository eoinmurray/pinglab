import { Link } from "react-router-dom";
import { ModeToggle } from "./ModeToggle";
import { SiGithub } from "@icons-pack/react-simple-icons";
import { ChevronUp, ChevronDown } from "lucide-react";

interface HeaderProps {
  slideControls?: {
    current: number;
    total: number;
    onPrevious: () => void;
    onNext: () => void;
  };
}

export function Header({ slideControls }: HeaderProps = {}) {
  return (
    <header className="fixed top-0 left-0 right-0 z-40">
      <div className="mx-auto w-full px-[var(--page-padding)] flex items-center gap-8 py-4">
        <nav className="flex items-center gap-1">
          <Link
            to="/"
            className="rounded-lg font-mono py-1.5 text-sm font-medium text-muted-foreground hover:underline"
          >
            pl
          </Link>
        </nav>

        <div className="flex-1" />

        {/* Slide navigation controls */}
        {slideControls && (
          <nav className="flex items-center gap-1">
            <button
              onClick={slideControls.onPrevious}
              className="p-1.5 text-muted-foreground/70 hover:text-foreground transition-colors duration-200"
              title="Previous slide (↑)"
            >
              <ChevronUp className="h-4 w-4" />
            </button>
            <span className="font-mono text-xs text-muted-foreground/70 tabular-nums min-w-[3ch] text-center">
              {slideControls.current + 1}/{slideControls.total}
            </span>
            <button
              onClick={slideControls.onNext}
              className="p-1.5 text-muted-foreground/70 hover:text-foreground transition-colors duration-200"
              title="Next slide (↓)"
            >
              <ChevronDown className="h-4 w-4" />
            </button>
          </nav>
        )}

        {/* Navigation */}
        <nav className="flex items-center gap-2">
          <Link
            to="https://github.com/eoinmurray/pinglab"
            target="_blank"
            className="text-muted-foreground/70 hover:text-foreground transition-colors duration-300"
            aria-label="GitHub"
          >
            <SiGithub className="h-4 w-4" />
          </Link>
          <ModeToggle />
        </nav>
      </div>
    </header>
  );
}
