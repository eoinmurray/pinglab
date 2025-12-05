import { Link } from "react-router-dom";
import { findSlides } from "../../plugins/cathedral-plugin/src/client";
import { DirectoryEntry } from "../../plugins/cathedral-plugin/src/lib";
import { Presentation } from "lucide-react";

export default function PageHeader({ directory }: { directory: DirectoryEntry }) {
  const slides = findSlides(directory);

  if (!slides) return null;

  return (
    <div className="flex items-center justify-end py-4 mb-4 border-b border-border/30">
      <Link
        to={`/${slides.path}`}
        className="group inline-flex items-center gap-2 px-3 py-1.5 text-sm font-mono tracking-wide text-muted-foreground hover:text-foreground border border-border/50 hover:border-border transition-all duration-300 hover:bg-muted/30"
      >
        <Presentation className="h-3.5 w-3.5" />
        <span>slides</span>
      </Link>
    </div>
  );
}
