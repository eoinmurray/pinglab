import { useMemo } from "react";
import { useTheme } from "next-themes";
import { useDirectory } from "../../../../plugins/cathedral-plugin/src/client";
import { FileEntry } from "../../../../plugins/cathedral-plugin/src/lib";
import { minimatch } from "minimatch";

function sortPathsNumerically(paths: string[]): void {
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
}

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


export function useGalleryImages({
  path,
  globs = null,
  limit,
}: {
  path?: string;
  globs?: string[] | null;
  limit?: number;
}) {
  const { resolvedTheme } = useTheme();

  let resolvedPath = path;

  const { directory } = useDirectory(resolvedPath);

  const paths = useMemo(() => {
    if (!directory) return [];

    const imageChildren = directory.children.filter((child): child is FileEntry => {
      return !!child.name.match(/\.(png|jpeg|gif|svg|webp)$/i) && child.type === "file";
    });

    let imagePaths = imageChildren.map(child => child.path);

    if (globs && globs.length > 0) {
      imagePaths = imagePaths.filter(p => {
        return globs.some(glob => minimatch(p.split('/').pop() || '', glob));
      });
    }

    sortPathsNumerically(imagePaths);
    let filtered = filterPathsByTheme(imagePaths, resolvedTheme);

    if (limit) {
      filtered = filtered.slice(0, limit);
    }

    return filtered;
  }, [directory, globs, resolvedTheme, limit]);

  return {
    paths,
    isLoading: !directory,
    isEmpty: directory !== undefined && paths.length === 0,
  };
}
