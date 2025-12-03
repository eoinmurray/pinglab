import { useState, useEffect } from "react";
import { DirectoryEntry, FileEntry } from "./lib";
import { cathedralPluginConfig } from "../../../cathedral-plugin.config";

async function parsePath(directory: DirectoryEntry, path: string): Promise<{ directory: DirectoryEntry; file: FileEntry | null }> {
  const parts = path === "." ? [] : path.split("/").filter(Boolean);

  let file = null;
  let currentDir = directory;

  for (let i = 0; i < parts.length; i++) {
    const part = parts[i];
    const isLastPart = i === parts.length - 1;

    // Check if this part matches a file (only on last part)
    if (isLastPart) {
      const matchedFile = currentDir.children.find(
        (child) => child.type === "file" && child.name === part
      ) as FileEntry | undefined;

      if (matchedFile) {
        file = matchedFile;
        break;
      }
    }

    // Otherwise, look for a directory
    const nextDir = currentDir.children.find(
      (child) => child.type === "directory" && child.name === part
    ) as DirectoryEntry | undefined;

    if (!nextDir) {
      throw new Error(`Path not found: ${path}`);
    }

    currentDir = nextDir;
  }

  return { directory: currentDir, file };
}

export function findReadme(directory: DirectoryEntry): FileEntry | null {
  const readme = directory.children.find((child) => 
    child.type === "file" && 
    [
      "README.md", "Readme.md", "readme.md",
      "README.mdx", "Readme.mdx", "readme.mdx"
    ].includes(child.name)
  ) as FileEntry | undefined;

  return readme || null;
}

export function findSlides(directory: DirectoryEntry): FileEntry | null {
  const readme = directory.children.find((child) =>
    child.type === "file" &&
    [
      "SLIDES.md", "Slides.md", "slides.md",
      "SLIDES.mdx", "Slides.mdx", "slides.mdx"
    ].includes(child.name)
  ) as FileEntry | undefined;

  return readme || null;
}


export type DirectoryError =
  | { type: 'config_not_found'; message: string }
  | { type: 'path_not_found'; message: string; status: 404 }
  | { type: 'fetch_error'; message: string }
  | { type: 'parse_error'; message: string };

export function useDirectory(path: string = ".") {
  const [directory, setDirectory] = useState<DirectoryEntry | null>(null);
  const [file, setFile] = useState<FileEntry | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<DirectoryError | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);

    (async () => {
      try {
        const response = await fetch(`${cathedralPluginConfig.contentPrefix}/.cathedral.json`);

        if (!response.ok) {
          if (response.status === 404) {
            setError({ type: 'config_not_found', message: '.cathedral.json not found' });
          } else {
            setError({ type: 'fetch_error', message: `Failed to fetch: ${response.status} ${response.statusText}` });
          }
          return;
        }

        let json;
        try {
          json = await response.json();
        } catch {
          setError({ type: 'parse_error', message: 'Failed to parse .cathedral.json' });
          return;
        }

        let parsed: { directory: DirectoryEntry; file: FileEntry | null };
        try {
          parsed = await parsePath(json, path);
        } catch {
          setError({ type: 'path_not_found', message: `Path not found: ${path}`, status: 404 });
          return;
        }

        parsed.directory.children.sort((a: any, b: any) => {
          let aDate, bDate;
          if (a.children) {
            const readme = findReadme(a);
            if (readme && readme.frontmatter && readme.frontmatter.date) {
              aDate = new Date(readme.frontmatter.date as string | number | Date)
            }
          }
          if (b.children) {
            const readme = findReadme(b);
            if (readme && readme.frontmatter && readme.frontmatter.date) {
              bDate = new Date(readme.frontmatter.date as string | number | Date)
            }
          }
          if (aDate && bDate) {
            return bDate.getTime() - aDate.getTime()
          }
          return 0;
        });

        setDirectory(parsed.directory);
        setFile(parsed.file);
      } catch (err: any) {
        setError({ type: 'fetch_error', message: err.message || 'Unknown error' });
      } finally {
        setLoading(false);
      }
    })();

    return () => {};
  }, [path]);

  return { directory, file, loading, error };
}

export function useFileContent(path: string) {
  const [blob, setBlob] = useState<Blob | null>(null);
  const [content, setContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);

    (async () => {
      try {
        const res = await fetch(`${cathedralPluginConfig.contentPrefix}/${path}`, {
          signal: controller.signal,
        });

        if (!res.ok) {
          throw new Error(`Failed to fetch: ${res.status} ${res.statusText}`);
        }

        const fetchedBlob = await res.blob();
        setBlob(fetchedBlob);

        // Try to read as text - some binary files may fail
        try {
          const text = await fetchedBlob.text();
          setContent(text);
        } catch {
          // Binary file - text content not available
          setContent(null);
        }
      } catch (err) {
        if (err instanceof Error && err.name === 'AbortError') {
          return; // Ignore abort errors
        }
        setError(err instanceof Error ? err : new Error('Unknown error'));
      } finally {
        setLoading(false);
      }
    })();

    return () => controller.abort();
  }, [path]);

  return { blob, content, loading, error };
}

export function isSimulationRunning() {
  const [running, setRunning] = useState<boolean>(false);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;

    const fetchStatus = async () => {
      const response = await fetch(`${cathedralPluginConfig.contentPrefix}/.running`);

      // this is an elaborate workaround to stop devtools logging errors on 404s
      const text = await response.text()
      if (text === "") {
        setRunning(true);
      } else {
        setRunning(false);
      }
    };

    // Initial fetch
    fetchStatus();

    // Poll every second
    interval = setInterval(fetchStatus, 1000);

    return () => {
      clearInterval(interval);
    };
  }, []);

  return running;
}