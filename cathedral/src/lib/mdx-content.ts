import { ComponentType } from 'react'

// Glob import all MDX files from biblio at build time
// Path is relative to this file: src/lib/mdx-content.ts
// ../../../biblio resolves to pinglab/biblio from pinglab/cathedral/src/lib
const mdxModules = import.meta.glob<MDXModule>('../../../biblio/**/*.mdx', { eager: false })

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
  // Normalize path: "experiment-1/README.mdx" -> "../../../biblio/experiment-1/README.mdx"
  // The cathedral plugin provides paths without the "biblio/" prefix
  const normalizedPath = `../../../biblio/${path}`

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
