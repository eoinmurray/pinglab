
import { Link, useNavigate, useParams } from "react-router-dom";
import { Folder, FileText, ChevronDown, ChevronRight } from "lucide-react";

import { cn } from "@/lib/utils";
import { DirectoryEntry, FileEntry } from "../../plugins/cathedral-plugin/src/lib";
import { cathedralPluginConfig } from "../../cathedral-plugin.config";
import { Breadcrumbs } from "./Breadcrumbs";
import { useState, useEffect, useRef } from "react";
import { findReadme } from "../../plugins/cathedral-plugin/src/client";
import { formatDate } from "@/lib/format-date";


function formatFileSize(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

const BROWSER_OPEN_KEY = "cathedral-browser-open";

export default function Browser({ directory, defaultOpen = true, alwaysOpen = false }: { directory: DirectoryEntry, defaultOpen?: boolean, alwaysOpen?: boolean }) {
  const [isOpen, setIsOpen] = useState(() => {
    if (alwaysOpen) return true;
    const stored = localStorage.getItem(BROWSER_OPEN_KEY);
    return stored !== null ? stored === "true" : defaultOpen;
  });

  // Force open when alwaysOpen prop is true
  useEffect(() => {
    if (alwaysOpen && !isOpen) {
      setIsOpen(true);
    }
  }, [alwaysOpen, isOpen]);

  // Persist open state to localStorage (only when not forced open)
  useEffect(() => {
    if (!alwaysOpen) {
      localStorage.setItem(BROWSER_OPEN_KEY, String(isOpen));
    }
  }, [isOpen, alwaysOpen]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const navigate = useNavigate();
  const { "*": currentPath } = useParams();
  const itemRefs = useRef<(HTMLAnchorElement | null)[]>([]);

  const folders = directory.children.filter((c): c is DirectoryEntry => c.type === "directory");
  const files = directory.children.filter((c): c is FileEntry => c.type === "file");
  const allItems = [...folders, ...files];

  // Reset selected index when directory changes
  useEffect(() => {
    setSelectedIndex(null);
    itemRefs.current = [];
  }, [directory.path]);

  // Scroll selected item into view
  useEffect(() => {
    if (isOpen && selectedIndex !== null && itemRefs.current[selectedIndex]) {
      itemRefs.current[selectedIndex]?.scrollIntoView({
        block: "nearest",
        behavior: "smooth",
      });
    }
  }, [selectedIndex, isOpen]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Check if any fullscreen component is active (Gallery or Slides)
      const isFullscreenActive = document.querySelector('.fixed.inset-0.bg-black\\/90') ||
                                 document.querySelector('.slides-container.fixed.inset-0');

      // Toggle browser with .
      if (e.key === "." && !e.metaKey && !e.ctrlKey && !e.altKey && !e.shiftKey) {
        // Don't toggle browser if fullscreen is active
        if (isFullscreenActive) return;
        e.preventDefault();
        setIsOpen(!isOpen);
        return;
      }

      // Navigate to homepage with ,
      if (e.key === "," && !e.metaKey && !e.ctrlKey && !e.altKey && !e.shiftKey) {
        // Don't navigate if fullscreen is active
        if (isFullscreenActive) return;
        e.preventDefault();
        navigate("/");
        return;
      }

      if (!isOpen || allItems.length === 0) return;

      switch (e.key) {
        case "ArrowDown":
          // Allow cmd-down to work normally
          if (e.metaKey) return;
          // Don't intercept if fullscreen is active
          if (isFullscreenActive) return;
          e.preventDefault();
          setSelectedIndex((prev) => prev === null ? 0 : (prev + 1) % allItems.length);
          break;

        case "ArrowUp":
          // Allow cmd-up to work normally
          if (e.metaKey) return;
          // Don't intercept if fullscreen is active
          if (isFullscreenActive) return;
          e.preventDefault();
          setSelectedIndex((prev) => prev === null ? 0 : (prev - 1 + allItems.length) % allItems.length);
          break;

        case "Enter":
        case "ArrowRight":
          // Don't intercept if fullscreen is active
          if (isFullscreenActive) {
            return;
          }
          e.preventDefault();
          if (selectedIndex !== null && allItems[selectedIndex]) {
            navigate(`/${allItems[selectedIndex].path}`);
          }
          break;

        case "ArrowLeft":
          // Don't intercept if fullscreen is active
          if (isFullscreenActive) {
            return;
          }
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
          // Don't intercept if fullscreen is active (let the fullscreen component handle it)
          if (isFullscreenActive) return;
          e.preventDefault();
          setSelectedIndex(null);
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, allItems, selectedIndex, navigate, currentPath, setIsOpen]);

  return (
    <div className="border rounded-lg overflow-hidden shadow-sm print:hidden">
      <div className="px-4 py-3 flex bg-muted/50 justify-between items-center gap-4 border-b">
        <Breadcrumbs />
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="p-1 rounded hover:bg-accent transition-colors text-muted-foreground hover:text-foreground"
          aria-label={isOpen ? "Collapse browser" : "Expand browser"}
        >
          {isOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
        </button>
      </div>

      {isOpen && directory.children.length === 0 && (
        <p className="text-muted-foreground text-center py-8 text-sm">No content found.</p>
      )}
      
      {isOpen && (
        <div className="space-y-1 px-2 py-3">
          {folders.map((folder, index) => {
            const isSelected = selectedIndex === index;
            
            const readme = findReadme(folder);
            const frontmatter = readme?.frontmatter;

            return (
              <Link
                key={folder.path}
                to={`/${folder.path}`}
                ref={(el) => (itemRefs.current[index] = el)}
                className={cn(
                  "flex items-center gap-3 px-3 py-2.5 rounded-md",
                  "hover:bg-accent transition-all duration-150",
                  "text-sm group",
                  isSelected && "bg-accent ring-1 ring-primary/20",
                )}
              >
                <Folder className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                <span className={cn(
                  "flex-1 truncate font-medium group-hover:text-foreground",
                  isSelected && "text-foreground"
                )}>
                  {folder.name}
                </span>
                <span className="text-xs text-muted-foreground flex-shrink-0">
                  {/* {folder.children.length} item{folder.children.length !== 1 ? "s" : ""} */}
                  {frontmatter?.date ? formatDate(new Date(frontmatter.date as string)) : ""}
                </span>
              </Link>
            );
          })}

          {files.map((file, index) => {
            const fileIndex = folders.length + index;
            const isSelected = selectedIndex === fileIndex;
            const isOwnPage = currentPath === file.path;

            return (
              <Link
                key={file.path}
                to={file.path.endsWith('.pdf') ? `${cathedralPluginConfig.contentPrefix}/${file.path}` : `/${file.path}`}
                target={file.path.endsWith('.pdf') ? "_blank" : "_self"}
                ref={(el) => (itemRefs.current[fileIndex] = el)}
                className={cn(
                  "flex items-center gap-3 px-3 py-2.5 rounded-md",
                  "hover:bg-accent transition-all duration-150",
                  "text-sm group",
                  isSelected && "bg-accent ring-1 ring-primary/20",
                  isOwnPage && "text-muted-foreground"
                )}
              >
                <FileText className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                <span className={cn(
                  "flex-1 truncate",
                  // isSelected && "text-foreground"
                )}>
                  {file.name}
                </span>
                <span className="text-xs text-muted-foreground flex-shrink-0">
                  {formatFileSize(file.size)}
                </span>
              </Link>
            );
          })}
        </div>
      )}

    </div>
  );
}