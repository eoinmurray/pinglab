import { ComponentType } from 'react'

// Glob import all MDX files from biblio at build time
// Uses the @biblio alias defined in vite.config.ts
const mdxModules = import.meta.glob<MDXModule>('@biblio/**/*.mdx', { eager: false })

export type MDXModule = {
  default: ComponentType
  frontmatter?: {
    title?: string
    description?: string
    date?: string
    [key: string]: unknown
  }
}

export async function loadMDXContent(path: string): Promise<MDXModule | null> {
  // Normalize path: "experiment-1/README.mdx" -> "../biblio/experiment-1/README.mdx"
  // The cathedral plugin provides paths without the "biblio/" prefix
  // Vite resolves the @biblio alias to ../biblio in glob keys
  const normalizedPath = `../biblio/${path}`

  const loader = mdxModules[normalizedPath]
  if (!loader) {
    console.log('Available MDX paths:', Object.keys(mdxModules))
    console.log('Requested path:', normalizedPath)
    return null
  }

  return await loader()
}

export function getMDXPaths(): string[] {
  return Object.keys(mdxModules).map(p => p.replace(/^\//, ''))
}
