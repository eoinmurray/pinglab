import { Link, useNavigate, useParams } from "react-router-dom";
import { Folder, FileText, ChevronRight, Home } from "lucide-react";
import { useEffect, useState } from "react";
import * as Collapsible from "@radix-ui/react-collapsible";

import { DirectoryEntry, FileEntry } from "../../plugins/cathedral-plugin/src/lib";
import { cathedralPluginConfig } from "../../cathedral-plugin.config";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarRail,
  useSidebar,
} from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";

function formatFileSize(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

function useRootDirectory() {
  const [rootDirectory, setRootDirectory] = useState<DirectoryEntry | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${cathedralPluginConfig.contentPrefix}/.cathedral.json`);
        const json = await res.json();
        setRootDirectory(json);
      } catch (err) {
        console.error("Failed to fetch root directory:", err);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  return { rootDirectory, loading };
}

function TreeItem({
  item,
  currentPath,
  expandedPaths,
  toggleExpanded,
}: {
  item: DirectoryEntry | FileEntry;
  currentPath: string;
  expandedPaths: Set<string>;
  toggleExpanded: (path: string) => void;
}) {
  const isActive = currentPath === item.path;

  if (item.type === "file") {
    return (
      <SidebarMenuItem>
        <SidebarMenuButton
          asChild
          isActive={isActive}
          className="pl-2"
        >
          <Link
            to={item.path.endsWith('.pdf') ? `${cathedralPluginConfig.contentPrefix}/${item.path}` : `/${item.path}`}
            target={item.path.endsWith('.pdf') ? "_blank" : "_self"}
            className="flex items-center justify-between"
          >
            <span className="flex items-center gap-2">
              <FileText className="h-4 w-4" />
              <span className="truncate">{item.name}</span>
            </span>
            <span className="text-xs text-muted-foreground flex-shrink-0">
              {formatFileSize(item.size)}
            </span>
          </Link>
        </SidebarMenuButton>
      </SidebarMenuItem>
    );
  }

  // It's a directory
  const isExpanded = expandedPaths.has(item.path);
  const folders = item.children.filter((c): c is DirectoryEntry => c.type === "directory");
  const files = item.children.filter((c): c is FileEntry => c.type === "file");
  const sortedChildren = [...folders, ...files];

  return (
    <SidebarMenuItem>
      <Collapsible.Root open={isExpanded} onOpenChange={() => toggleExpanded(item.path)}>
        <div className="flex items-center">
          <Collapsible.Trigger asChild>
            <button
              disabled
              className="p-1 rounded-sm"
              onClick={(e) => {
                e.stopPropagation();
                toggleExpanded(item.path);
              }}
            >
              <ChevronRight
                className={cn(
                  "h-4 w-4 transition-transform",
                  isExpanded && "rotate-90"
                )}
              />
            </button>
          </Collapsible.Trigger>
          <SidebarMenuButton
            asChild
            isActive={isActive}
            className="flex-1"
          >
            <Link
              to={`/${item.path}`}
              className="flex items-center justify-between"
            >
              <span className="flex items-center gap-2">
                <Folder className="h-4 w-4" />
                <span className="truncate font-medium">{item.name}</span>
              </span>
              <span className="text-xs text-muted-foreground flex-shrink-0">
                {item.children.length}
              </span>
            </Link>
          </SidebarMenuButton>
        </div>
        <Collapsible.Content>
          <SidebarMenuSub>
            {sortedChildren.map((child) => (
              <TreeItem
                key={child.path}
                item={child}
                currentPath={currentPath}
                expandedPaths={expandedPaths}
                toggleExpanded={toggleExpanded}
              />
            ))}
          </SidebarMenuSub>
        </Collapsible.Content>
      </Collapsible.Root>
    </SidebarMenuItem>
  );
}

export function AppSidebar() {
  const navigate = useNavigate();
  const { "*": currentPath = "" } = useParams();
  const { open, setOpen, isMobile, setOpenMobile } = useSidebar();
  const { rootDirectory, loading } = useRootDirectory();

  // Track which folders are expanded
  const [expandedPaths, setExpandedPaths] = useState<Set<string>>(new Set());

  // Auto-expand folders in the current path
  useEffect(() => {
    if (currentPath) {
      const pathParts = currentPath.split("/");
      const pathsToExpand = new Set<string>();
      let accPath = "";
      for (const part of pathParts) {
        accPath = accPath ? `${accPath}/${part}` : part;
        pathsToExpand.add(accPath);
      }
      setExpandedPaths(prev => {
        const next = new Set(prev);
        pathsToExpand.forEach(p => next.add(p));
        return next;
      });
    }
  }, [currentPath]);

  const toggleExpanded = (path: string) => {
    setExpandedPaths(prev => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  };

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const isFullscreenActive = document.querySelector('.fixed.inset-0.bg-black\\/90') ||
                                 document.querySelector('.slides-container.fixed.inset-0');

      // Toggle sidebar with .
      if (e.key === "." && !e.metaKey && !e.ctrlKey && !e.altKey && !e.shiftKey) {
        if (isFullscreenActive) return;
        e.preventDefault();
        if (isMobile) {
          setOpenMobile(!open);
        } else {
          setOpen(!open);
        }
        return;
      }

      // Navigate to homepage with ,
      if (e.key === "," && !e.metaKey && !e.ctrlKey && !e.altKey && !e.shiftKey) {
        if (isFullscreenActive) return;
        e.preventDefault();
        navigate("/");
        return;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [open, navigate, setOpen, isMobile, setOpenMobile]);

  const folders = rootDirectory?.children.filter((c): c is DirectoryEntry => c.type === "directory") ?? [];
  const files = rootDirectory?.children.filter((c): c is FileEntry => c.type === "file") ?? [];
  const sortedChildren = [...folders, ...files];

  return (
    <Sidebar>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupContent>
            {loading && (
              <p className="text-muted-foreground text-center py-8 text-sm">Loading...</p>
            )}
            {!loading && sortedChildren.length === 0 && (
              <p className="text-muted-foreground text-center py-8 text-sm">No content found.</p>
            )}
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton
                  asChild
                  isActive={currentPath === ""}
                >
                  <Link to="/" className="flex items-center gap-2">
                    <Folder className="h-4 w-4" />
                    <span className="font-medium">root</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
              {sortedChildren.map((item) => (
                <TreeItem
                  key={item.path}
                  item={item}
                  currentPath={currentPath}
                  expandedPaths={expandedPaths}
                  toggleExpanded={toggleExpanded}
                />
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarRail />
    </Sidebar>
  );
}
