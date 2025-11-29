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

export function useDirectory(path: string = ".") {
  const [directory, setDirectory] = useState<DirectoryEntry | null>(null);
  const [file, setFile] = useState<FileEntry | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);

    (async () => {
      try {
        try {
          const res = await fetch(`${cathedralPluginConfig.contentPrefix}/.cathedral.json`);

          if (!res.ok) {
            throw new Error(`Failed to fetch directory structure: ${res.status} ${res.statusText}`);
          } 
          
          const json = await res.json();
          const parsed: {
            directory: DirectoryEntry;
            file: FileEntry | null;
          } = await parsePath(json, path)

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
          })

          setDirectory(parsed.directory);
          setFile(parsed.file);
        } catch (err: any) {
          setError('Failed to load .cathedral.json');
        }
        
      } finally {
        setLoading(false);
      }
    })();

    return () => {};
  }, [path]);

  return { directory, file, loading, error };
}

export function fetchFileContent(path: string) {
  const [blob, setBlob] = useState<unknown | null>(null);
  const [content, setContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    setLoading(true);

    (async () => {
      try {
        try {
          const res = await fetch(`${cathedralPluginConfig.contentPrefix}/${path}`);
          const blob = await res.blob()
          setBlob(blob);

          try {
            const text = await blob.text();
            setContent(text);
          } catch (err) {}

        } catch (err) {
          setError(err as Error);
        }
        
      } finally {
        setLoading(false);
      }
    })();

    return () => {};
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

    // Poll every 5 seconds
    interval = setInterval(fetchStatus, 1000);

    return () => {
      clearInterval(interval);
    };
  }, []);

  return running;
}