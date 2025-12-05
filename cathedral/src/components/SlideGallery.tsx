import { useMemo } from "react";
import { useTheme } from "next-themes";
import { useDirectory } from "../../plugins/cathedral-plugin/src/client";
import { FileEntry } from "../../plugins/cathedral-plugin/src/lib";
import { cathedralPluginConfig } from "../../cathedral-plugin.config";
import { minimatch } from "minimatch";
import { useParams } from "react-router-dom";
import { cn } from "@/lib/utils";

function filterPathsByTheme(paths: string[], theme: string | undefined): string[] {
  const pathGroups = new Map<string, { light?: string; dark?: string; original?: string }>();

  paths.forEach(path => {
    if (path.endsWith('_light.png')) {
      const baseName = path.replace('_light.png', '');
      const group = pathGroups.get(baseName) || {};
      group.light = path;
      pathGroups.set(baseName, group);
    } else if (path.endsWith('_dark.png')) {
      const baseName = path.replace('_dark.png', '');
      const group = pathGroups.get(baseName) || {};
      group.dark = path;
      pathGroups.set(baseName, group);
    } else {
      pathGroups.set(path, { original: path });
    }
  });

  const filtered: string[] = [];
  pathGroups.forEach((group, baseName) => {
    if (group.original) {
      filtered.push(group.original);
    } else {
      const isDark = theme === 'dark';
      const preferredPath = isDark ? group.dark : group.light;
      const fallbackPath = isDark ? group.light : group.dark;
      filtered.push(preferredPath || fallbackPath || baseName);
    }
  });

  return filtered;
}

export default function SlideGallery({
  path,
  relativePath,
  globs = null,
  single = false,
  limit,
}: {
  path?: string,
  relativePath?: string,
  globs?: string[] | null,
  single?: boolean,
  limit?: number,
}) {
  const { "*": paramPath = "." } = useParams();

  if (relativePath) {
    const basePath = paramPath === "." ? "" : paramPath;
    path = basePath + (basePath.endsWith("/") ? "" : "/") + relativePath;
  }

  const { directory } = useDirectory(path);

  const imageChildren = directory?.children
    .filter((child): child is FileEntry => {
      return !!child.name.match(/\.(png|jpeg|gif|svg|webp)$/i) && child.type === "file";
    });

  const paths = imageChildren?.map(child => child.path) || [];

  if (globs && globs.length > 0) {
    const matchedPaths = paths.filter(p => {
      return globs.some(glob => minimatch(p.split('/').pop() || '', glob));
    });
    paths.splice(0, paths.length, ...matchedPaths);
  }

  paths.sort((a, b) => {
    const nums = (s: string) => (s.match(/\d+/g) || []).map(Number);
    const na = nums(a);
    const nb = nums(b);
    const len = Math.max(na.length, nb.length);
    for (let i = 0; i < len; i++) {
      const diff = (na[i] ?? 0) - (nb[i] ?? 0);
      if (diff !== 0) return diff;
    }
    return a.localeCompare(b);
  });

  const { resolvedTheme } = useTheme();
  let filteredPaths = useMemo(() => filterPathsByTheme(paths, resolvedTheme), [paths, resolvedTheme]);

  if (limit) {
    filteredPaths = filteredPaths.slice(0, limit);
  }

  const gridCols = useMemo(() => {
    const count = filteredPaths.length;
    if (count === 1) return "grid-cols-1";
    if (count === 2) return "md:grid-cols-2 sm:grid-cols-1";
    if (count <= 4) return "md:grid-cols-3 sm:grid-cols-1";
    return "grid-cols-3";
  }, [filteredPaths.length]);

  if (!directory) {
    return (
      <div className="not-prose h-[50vh] flex items-center justify-center">
        <div className="text-muted-foreground font-mono text-xs">loading...</div>
      </div>
    );
  }

  if (filteredPaths.length === 0) {
    return (
      <div className="not-prose h-[50vh] flex items-center justify-center">
        <div className="text-muted-foreground font-mono text-xs">no images</div>
      </div>
    );
  }

  // Single image - maximize to fill available space
  if (single) {
    return (
      <div className="not-prose w-full h-[90vh] flex items-center justify-center">
        <img
          src={`${cathedralPluginConfig.contentPrefix}/${filteredPaths[0]}`}
          alt=""
          className="max-h-full max-w-full w-auto h-auto object-contain"
        />
      </div>
    );
  }

  // Grid of images - maximize to fill screen
  return (
    <div className="not-prose w-full h-[100vh] lg:px-12">
      <div className={cn("grid h-full w-full", gridCols, "gap-2")}>
        {filteredPaths.map((imgPath) => (
          <div key={imgPath} className="flex items-center justify-center overflow-hidden min-h-0">
            <img
              src={`${cathedralPluginConfig.contentPrefix}/${imgPath}`}
              alt=""
              className="max-h-full max-w-full w-auto h-auto object-contain"
            />
          </div>
        ))}
      </div>
    </div>
  );
}
