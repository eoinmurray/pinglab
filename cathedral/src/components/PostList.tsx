import { Link, useNavigate, useParams } from "react-router-dom";
import { useState, useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
import { isFullscreenActive } from "@/lib/constants";
import { DirectoryEntry } from "../../plugins/cathedral-plugin/src/lib";
import { findReadme } from "../../plugins/cathedral-plugin/src/client";
import { formatDate } from "@/lib/format-date";

export default function PostList({ directory }: { directory: DirectoryEntry }) {
  const folders = directory.children.filter((c): c is DirectoryEntry => c.type === "directory");
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const navigate = useNavigate();
  const { "*": currentPath } = useParams();
  const itemRefs = useRef<(HTMLAnchorElement | null)[]>([]);

  // Reset selected index when directory changes
  useEffect(() => {
    setSelectedIndex(null);
    itemRefs.current = [];
  }, [directory.path]);

  // Scroll selected item into view
  useEffect(() => {
    if (selectedIndex !== null && itemRefs.current[selectedIndex]) {
      itemRefs.current[selectedIndex]?.scrollIntoView({
        block: "nearest",
        behavior: "smooth",
      });
    }
  }, [selectedIndex]);

  // Keyboard navigation
  useEffect(() => {
    if (folders.length === 0) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't handle keyboard events if any fullscreen component is active
      if (isFullscreenActive()) return;

      switch (e.key) {
        case "ArrowDown":
          if (e.metaKey) return;
          e.preventDefault();
          setSelectedIndex((prev) => prev === null ? 0 : (prev + 1) % folders.length);
          break;

        case "ArrowUp":
          if (e.metaKey) return;
          e.preventDefault();
          setSelectedIndex((prev) => prev === null ? 0 : (prev - 1 + folders.length) % folders.length);
          break;

        case "Enter":
        case "ArrowRight":
          e.preventDefault();
          if (selectedIndex !== null && folders[selectedIndex]) {
            navigate(`/${folders[selectedIndex].path}`);
          }
          break;

        case "ArrowLeft":
          e.preventDefault();
          // Navigate to parent directory
          if (currentPath) {
            const pathParts = currentPath.split("/");
            pathParts.pop();
            const parentPath = pathParts.join("/");
            navigate(parentPath ? `/${parentPath}` : "/");
          }
          break;

        case "Escape":
          e.preventDefault();
          setSelectedIndex(null);
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [folders, selectedIndex, navigate, currentPath]);

  if (folders.length === 0) {
    return (
      <div className="py-16 text-center">
        <p className="text-muted-foreground text-sm">No posts yet.</p>
      </div>
    );
  }

  return (
    <div>
      {folders.map((folder, index) => {
        const readme = findReadme(folder);
        const frontmatter = readme?.frontmatter;
        const title = (frontmatter?.title as string) || folder.name;
        const description = frontmatter?.description as string | undefined;
        const date = frontmatter?.date ? new Date(frontmatter.date as string) : null;
        const isSelected = selectedIndex === index;

        return (
          <Link
            key={folder.path}
            to={`/${folder.path}`}
            ref={(el) => (itemRefs.current[index] = el)}
            className={cn(
              "group block py-4",
              "transition-all duration-200",
              isSelected && "bg-accent -mx-4 px-4 rounded-lg"
            )}
            style={{
              animationDelay: `${index * 50}ms`,
            }}
          >
            <article className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between sm:gap-8">
              {/* Main content */}
              <div className="flex-1 min-w-0 space-y-1.5">
                <h3 className={cn(
                  "font-medium text-foreground",
                  "group-hover:text-primary transition-colors duration-200",
                  "flex items-center gap-2"
                )}>
                  <span className="truncate">{title}</span>
                </h3>

                {description && (
                  <p className="text-sm text-muted-foreground line-clamp-2 leading-relaxed">
                    {description}
                  </p>
                )}
              </div>

              {/* Date - right aligned on desktop */}
              {date && (
                <time
                  dateTime={date.toISOString()}
                  className={cn(
                    "text-xs font-mono text-muted-foreground tabular-nums",
                    "sm:text-right flex-shrink-0",
                    "mt-1 sm:mt-0.5"
                  )}
                >
                  {formatDate(date)}
                </time>
              )}
            </article>
          </Link>
        );
      })}
    </div>
  );
}
