
import { Link, useNavigate, useParams } from "react-router-dom";
import { Folder, FileText } from "lucide-react";

import { cn } from "@/lib/utils";
import { isFullscreenActive } from "@/lib/constants";
import { DirectoryEntry, FileEntry } from "../../plugins/cathedral-plugin/src/lib";
import { cathedralPluginConfig } from "../../cathedral-plugin.config";
import { Breadcrumbs } from "./Breadcrumbs";
import { useState, useEffect, useRef, useCallback } from "react";
import { useKeyBindings } from "@/hooks/useKeyBindings";
import { findReadme } from "../../plugins/cathedral-plugin/src/client";
import { formatDate } from "@/lib/format-date";
import { formatFileSize } from "@/lib/format-file-size";

export default function Browser({ directory }: { directory: DirectoryEntry }) {
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
    if (selectedIndex !== null && itemRefs.current[selectedIndex]) {
      itemRefs.current[selectedIndex]?.scrollIntoView({
        block: "nearest",
        behavior: "smooth",
      });
    }
  }, [selectedIndex]);

  // Navigation actions
  const goHome = useCallback(() => navigate("/"), [navigate]);
  const selectNext = useCallback(() => {
    setSelectedIndex((prev) => prev === null ? 0 : (prev + 1) % allItems.length);
  }, [allItems.length]);
  const selectPrevious = useCallback(() => {
    setSelectedIndex((prev) => prev === null ? 0 : (prev - 1 + allItems.length) % allItems.length);
  }, [allItems.length]);
  const openSelected = useCallback(() => {
    if (selectedIndex !== null && allItems[selectedIndex]) {
      navigate(`/${allItems[selectedIndex].path}`);
    }
  }, [selectedIndex, allItems, navigate]);
  const goToParent = useCallback(() => {
    if (currentPath) {
      const pathParts = currentPath.split("/");
      pathParts.pop();
      const parentPath = pathParts.join("/");
      navigate(parentPath ? `/${parentPath}` : "/");
    }
  }, [currentPath, navigate]);
  const clearSelection = useCallback(() => setSelectedIndex(null), []);

  // Global keybindings (always active unless fullscreen)
  useKeyBindings(
    [
      { key: ",", action: goHome },
    ],
    { enabled: () => !isFullscreenActive() }
  );

  // Browser navigation keybindings
  useKeyBindings(
    [
      { key: "ArrowDown", action: selectNext },
      { key: "ArrowUp", action: selectPrevious },
      { key: ["Enter", "ArrowRight"], action: openSelected },
      { key: "ArrowLeft", action: goToParent },
      { key: "Escape", action: clearSelection },
    ],
    { enabled: () => !isFullscreenActive() && allItems.length > 0 }
  );

  return (
    <div className="border rounded-lg overflow-hidden shadow-sm print:hidden">
      <div className="px-4 py-3 flex bg-muted/50 justify-between items-center gap-4 border-b">
        <Breadcrumbs />
      </div>

      {directory.children.length === 0 && (
        <p className="text-muted-foreground text-center py-8 text-sm">No content found.</p>
      )}

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
              <span className="flex-1 truncate">
                {file.name}
              </span>
              <span className="text-xs text-muted-foreground flex-shrink-0">
                {formatFileSize(file.size)}
              </span>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
