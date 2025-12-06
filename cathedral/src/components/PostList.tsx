import { Link, useNavigate, useParams } from "react-router-dom";
import { useState, useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
import { isFullscreenActive } from "@/lib/constants";
import { DirectoryEntry } from "../../plugins/cathedral-plugin/src/lib";
import { findReadme, findSlides } from "../../plugins/cathedral-plugin/src/client";
import { formatDate } from "@/lib/format-date";
import { ArrowRight, Presentation } from "lucide-react";

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
      <div className="py-24 text-center">
        <p className="text-muted-foreground font-mono text-sm tracking-wide">no entries</p>
      </div>
    );
  }

  let posts = folders.map((folder) => {
    const readme = findReadme(folder);
    const slides = findSlides(folder);
    return {
      ...folder,
      readme,
      slides,
    }
  })

  posts = posts.filter((post) => {
    return post.readme?.frontmatter?.visibility !== "hidden";
  });

  posts = posts.sort((a, b) => {
    let aDate = a.readme?.frontmatter?.date ? new Date(a.readme.frontmatter.date as string) : null;
    let bDate = b.readme?.frontmatter?.date ? new Date(b.readme.frontmatter.date as string) : null;

    if (!aDate && a.slides) {
      aDate = a.slides.frontmatter?.date ? new Date(a.slides.frontmatter.date as string) : null;
    }
    if (!bDate && b.slides) {
      bDate = b.slides.frontmatter?.date ? new Date(b.slides.frontmatter.date as string) : null;
    }

    if (aDate && bDate) {
      return bDate.getTime() - aDate.getTime();
    } else if (aDate) {
      return -1;
    } else if (bDate) {
      return 1;
    } else {
      return a.name.localeCompare(b.name);
    }
  });

  const postsGroupedByMonthAndYear: { [key: string]: typeof posts } = {};
  posts.forEach((post) => {
    let date = post.readme?.frontmatter?.date ? new Date(post.readme.frontmatter.date as string) : null;
    if (!date && post.slides) {
      date = post.slides.frontmatter?.date ? new Date(post.slides.frontmatter.date as string) : null;
    }
    const monthYear = date ? `${date.getFullYear()}-${date.getMonth() + 1}` : "unknown";
    if (!postsGroupedByMonthAndYear[monthYear]) {
      postsGroupedByMonthAndYear[monthYear] = [];
    }
    postsGroupedByMonthAndYear[monthYear].push(post);
  });

  return (
    <div className="space-y-8">
      {Object.entries(postsGroupedByMonthAndYear)
        .sort(([a], [b]) => {
          if (a === "unknown") return 1;
          if (b === "unknown") return -1;
          return b.localeCompare(a);
        })
        .map(([monthYear, monthPosts]) => {
          const [year, month] = monthYear.split("-");
          const displayDate = monthYear === "unknown" 
            ? "Unknown Date" 
            : new Date(parseInt(year), parseInt(month) - 1).toLocaleDateString("en-US", { 
                year: "numeric", 
                month: "long" 
              });

          return (
            <div key={monthYear}>
              <h2 className="text-xs font-mono uppercase tracking-wider text-muted-foreground mb-3">
                {displayDate}
              </h2>
              <div className="space-y-1">
                {monthPosts.map((post) => {
                  const index = posts.indexOf(post);
                  let frontmatter = post.readme?.frontmatter;

                  if (!post.readme && post.slides) {
                    frontmatter = post.slides.frontmatter;
                  }

                  const title = (frontmatter?.title as string) || post.name;
                  const description = frontmatter?.description as string | undefined;
                  const date = frontmatter?.date ? new Date(frontmatter.date as string) : null;
                  const isSelected = selectedIndex === index;
                  const isSlidesOnly = post.slides && !post.readme;

                  return (
                    <Link
                      key={post.path}
                      to={(post.slides && !post.readme) ? `/${post.slides.path}` : `/${post.path}`}
                      ref={(el) => (itemRefs.current[index] = el)}
                      className={cn(
                        "group block py-3 px-3 -mx-3 rounded-md",
                        "transition-colors duration-150",
                        "hover:bg-accent",
                        isSelected && "bg-accent ring-1 ring-ring"
                      )}
                    >
                      <article className="flex items-start gap-4">
                        {/* Date - left side, fixed width */}
                        <time
                          dateTime={date?.toISOString()}
                          className="font-mono text-xs text-muted-foreground tabular-nums w-20 flex-shrink-0 pt-0.5"
                        >
                          {date ? formatDate(date) : <span className="text-muted-foreground/30">—</span>}
                        </time>

                        {/* Main content */}
                        <div className="flex-1 min-w-0">
                          <h3 className={cn(
                            "text-sm font-medium text-foreground",
                            "group-hover:text-primary transition-colors duration-200",
                            "flex items-center gap-2"
                          )}>
                            {isSlidesOnly && (
                              <span className="inline-flex items-center gap-1 px-1.5 py-0.5 bg-primary/10 text-primary rounded text-[10px] font-mono uppercase tracking-wider">
                                <Presentation className="h-2.5 w-2.5" />
                                slides
                              </span>
                            )}
                            <span>{title}</span>
                            <ArrowRight className="h-3 w-3 opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all duration-200 text-primary" />
                          </h3>

                          {description && (
                            <p className="text-sm text-muted-foreground line-clamp-1 mt-0.5">
                              {description}
                            </p>
                          )}
                        </div>
                      </article>
                    </Link>
                  );
                })}
              </div>
            </div>
          );
        })}
    </div>
  );
}
