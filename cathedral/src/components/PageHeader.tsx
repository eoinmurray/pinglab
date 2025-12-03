
import { Link } from "react-router-dom";
import { findReadme, findSlides } from "../../plugins/cathedral-plugin/src/client";
import { DirectoryEntry } from "../../plugins/cathedral-plugin/src/lib";
import { Button } from "./ui/button";
import { formatDate } from "@/lib/format-date";
import { ArrowBigLeft, Fullscreen } from "lucide-react";

export default function PageHeader({ directory }: { directory: DirectoryEntry }) {
  const slides = findSlides(directory);
  const readme = findReadme(directory);
  const frontmatter = readme?.frontmatter;

  return (
    <div className="flex mb-4 items-center gap-2 text-sm text-muted-foreground justify-between">
      <div className="flex gap-2 items-center">
        <Link to="/" className="flex gap-2 items-center hover:underline">
          <ArrowBigLeft className="w-3 h-3" />
          Back
        </Link>
      </div>

      <div className="flex-1" />

      <span className="flex-shrink-0">
        {frontmatter?.date ? formatDate(new Date(frontmatter.date as string)) : ""}
      </span>

      {slides && (
        <Button
          variant="outline"
          size="sm"
        >
          <Link to={`/${slides.path}`} className="flex items-center gap-2">
            <Fullscreen />
            View Slides
          </Link>
        </Button>
      )}
    </div>
  )
}
