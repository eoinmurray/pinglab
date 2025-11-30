import { readdir, writeFile, stat, readFile } from 'fs/promises';
import { join, relative, basename, extname } from 'path';
import matter from 'gray-matter';

export const GITIGNORE_FILENAME = '.gitignore'

export type FileEntry = {
  type: 'file';
  name: string;
  path: string;
  size: number;
  frontmatter?: {
    title?: string;
    description?: string;
    date?: string;
  };
}

export type DirectoryEntry = {
  type: 'directory';
  name: string;
  path: string;
  children: (FileEntry | DirectoryEntry)[];
}

const MARKDOWN_EXTENSIONS = ['.md', '.mdx', '.markdown'];

/**
 * Check if a file is a markdown file
 */
function isMarkdownFile(filename: string): boolean {
  const ext = extname(filename).toLowerCase();
  return MARKDOWN_EXTENSIONS.includes(ext);
}

/**
 * Parse frontmatter from a markdown file
 * Returns undefined if no frontmatter exists or file is not markdown
 */
async function parseFrontmatter(filePath: string): Promise<Record<string, unknown> | undefined> {
  try {
    const content = await readFile(filePath, 'utf-8');
    const { data } = matter(content);
    // Only return frontmatter if it has content
    if (Object.keys(data).length > 0) {
      return data;
    }
    return undefined;
  } catch {
    return undefined;
  }
}

/**
 * Check if a path is a directory
 */
async function isDirectory(path: string): Promise<boolean> {
  try {
    const stats = await stat(path);
    return stats.isDirectory();
  } catch {
    return false;
  }
}

/**
 * Parse .gitignore file and return array of patterns
 */
async function parseGitignore(gitignorePath: string): Promise<string[]> {
  try {
    const content = await readFile(gitignorePath, 'utf-8');
    return content
      .split('\n')
      .map(line => line.trim())
      .filter(line => line && !line.startsWith('#')); // Skip empty lines and comments
  } catch {
    return []; // Return empty array if .gitignore doesn't exist
  }
}

/**
 * Convert gitignore pattern to regex-like matching function
 * Supports:
 * - *.ext (any file ending with .ext)
 * - dir/ (directory anywhere in tree)
 * - dir/** (directory and all contents)
 * - **\/pattern (pattern anywhere)
 */
function createGitignoreMatcher(patterns: string[]): (relativePath: string, isDir: boolean) => boolean {
  return (relativePath: string, isDir: boolean) => {
    const pathParts = relativePath.split('/');
    const filename = basename(relativePath);

    for (const pattern of patterns) {
      // Handle directory patterns (ending with /)
      if (pattern.endsWith('/')) {
        const dirPattern = pattern.slice(0, -1);
        if (isDir) {
          // Match if any part of the path matches the directory name
          if (pathParts.includes(dirPattern) || relativePath === dirPattern || relativePath.startsWith(dirPattern + '/')) {
            return true;
          }
        }
        continue;
      }

      // Handle ** patterns
      if (pattern.includes('**')) {
        const regexPattern = pattern
          .replace(/\./g, '\\.')
          .replace(/\*\*/g, '.*')
          .replace(/\*/g, '[^/]*');
        const regex = new RegExp(`^${regexPattern}$`);
        if (regex.test(relativePath)) {
          return true;
        }
        continue;
      }

      // Handle simple wildcard patterns (*.ext)
      if (pattern.includes('*')) {
        const regexPattern = pattern
          .replace(/\./g, '\\.')
          .replace(/\*/g, '[^/]*');
        const regex = new RegExp(`^${regexPattern}$`);

        // Check if the pattern matches the full path or just the filename
        if (regex.test(relativePath) || regex.test(filename)) {
          return true;
        }
        continue;
      }

      // Exact match - check both full path and filename
      if (relativePath === pattern || filename === pattern) {
        return true;
      }

      // Prefix match for directories
      if (relativePath.startsWith(pattern + '/')) {
        return true;
      }
    }

    return false;
  };
}

/**
 * Recursively scan directory and build flat index
 */
async function scanDirectory(
  dirPath: string,
  rootPath: string,
  shouldIgnore: (relativePath: string, isDir: boolean) => boolean,
  depth: number = 0
): Promise<DirectoryEntry> {
  const entries = await readdir(dirPath);
  const children: (FileEntry | DirectoryEntry)[] = [];

  for (const entry of entries) {
    // Skip hidden files and the index file itself
    if (entry.startsWith('.')) {
      continue;
    }

    const fullPath = join(dirPath, entry);
    const relativePath = relative(rootPath, fullPath);
    const isDir = await isDirectory(fullPath);

    // Check if this path should be ignored
    if (shouldIgnore(relativePath, isDir)) {
      continue;
    }

    if (isDir) {
      // Recursively scan subdirectories
      const subDir = await scanDirectory(fullPath, rootPath, shouldIgnore, depth + 1);
      children.push(subDir);
    } else {
      // Add file entry
      const stats = await stat(fullPath);
      const fileEntry: FileEntry = {
        type: 'file',
        name: entry,
        path: relativePath,
        size: stats.size,
      };

      // Parse frontmatter for markdown files
      if (isMarkdownFile(entry)) {
        const frontmatter = await parseFrontmatter(fullPath);
        if (frontmatter) {
          fileEntry.frontmatter = frontmatter;
        }
      }

      children.push(fileEntry);
    }
  }

  return {
    type: 'directory',
    name: basename(dirPath),
    path: relative(rootPath, dirPath) || '.',
    children,
  };
}

/**
 * Build content for a single target
 */
async function buildTarget(target: string): Promise<void> {
  console.log(`${'-'.repeat(80)}`);
  console.log(`Building: ${target}`);
  console.log(`${'-'.repeat(80)}`);

  const gitignorePath = join(target, GITIGNORE_FILENAME);
  const ignorePatterns = await parseGitignore(gitignorePath);
  const shouldIgnore = createGitignoreMatcher(ignorePatterns);

  if (ignorePatterns.length > 0) {
    console.log(`   Found .gitignore with ${ignorePatterns.length} pattern${ignorePatterns.length !== 1 ? 's' : ''}`);
  } else {
    console.log(`   No .gitignore found or empty\n`);
  }

  const index = await scanDirectory(target, target, shouldIgnore);

  // Count files and directories
  function countEntries(entry: DirectoryEntry): { files: number; dirs: number } {
    let files = 0;
    let dirs = 0;

    for (const child of entry.children) {
      if (child.type === 'file') {
        files++;
      } else {
        dirs++;
        const counts = countEntries(child);
        files += counts.files;
        dirs += counts.dirs;
      }
    }

    return { files, dirs };
  }

  const counts = countEntries(index);
  console.log(`   Found ${counts.dirs} directories and ${counts.files} files`);

  // Write output
  const outputFile = join(target, '.cathedral.json');
  await writeFile(outputFile, JSON.stringify(index, null, 2));

  console.log(`   Generated ${outputFile}`);
}

/**
 * Build content for all configured targets
 */
async function buildAll(targets: string[]): Promise<void> {
  console.log(`\n${'='.repeat(80)}`);
  console.log(` ▗▄▄▖ ▗▄▖▗▄▄▄▖▗▖ ▗▖▗▄▄▄▖▗▄▄▄ ▗▄▄▖  ▗▄▖ ▗▖   
▐▌   ▐▌ ▐▌ █  ▐▌ ▐▌▐▌   ▐▌  █▐▌ ▐▌▐▌ ▐▌▐▌   
▐▌   ▐▛▀▜▌ █  ▐▛▀▜▌▐▛▀▀▘▐▌  █▐▛▀▚▖▐▛▀▜▌▐▌   
▝▚▄▄▖▐▌ ▐▌ █  ▐▌ ▐▌▐▙▄▄▖▐▙▄▄▀▐▌ ▐▌▐▌ ▐▌▐▙▄▄▖                           
  `);

  console.log(`Targets: ${targets.join(', ')}`);

  for (const target of targets) {
    await buildTarget(target);
  }

  console.log(`${'='.repeat(80)}`);
}


export { buildAll };
