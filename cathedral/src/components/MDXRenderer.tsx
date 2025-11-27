
import { useState, useEffect } from 'react'
import { compile, run } from '@mdx-js/mdx'
import * as runtime from 'react/jsx-runtime'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import Gallery from './Gallery'


const MDX_COMPONENTS = {
  Gallery,
  // Style headers with IDs for TOC navigation
  h1: (props: any) => {
    const id = props.children
      ?.toString()
      .toLowerCase()
      .replace(/[^a-z0-9\s-]/g, '')
      .replace(/\s+/g, '-')
    return <h1 id={id} className="text-3xl font-serif font-bold mt-8 mb-4" {...props} />
  },
  h2: (props: any) => {
    const id = props.children
      ?.toString()
      .toLowerCase()
      .replace(/[^a-z0-9\s-]/g, '')
      .replace(/\s+/g, '-')
    return <h2 id={id} className="text-2xl font-serif font-bold mt-6 mb-3" {...props} />
  },
  h3: (props: any) => {
    const id = props.children
      ?.toString()
      .toLowerCase()
      .replace(/[^a-z0-9\s-]/g, '')
      .replace(/\s+/g, '-')
    return <h3 id={id} className="text-xl font-serif font-bold mt-4 mb-2" {...props} />
  },

  h4: (props: any) => {
    const id = props.children
      ?.toString()
      .toLowerCase()
      .replace(/[^a-z0-9\s-]/g, '')
      .replace(/\s+/g, '-')
    return <h4 id={id} className="text-lg font-serif font-bold mt-3 mb-1" {...props} />
  },

  h5: (props: any) => {
    const id = props.children
      ?.toString()
      .toLowerCase()
      .replace(/[^a-z0-9\s-]/g, '')
      .replace(/\s+/g, '-')
    return <h5 id={id} className="text-base font-serif font-bold mt-2 mb-1" {...props} />
  },

  pre: (props: any) => {
    return (
      <pre
        className={`not-prose w-full overflow-x-auto p-2 md:p-4 text-xs md:text-sm bg-sidebar`}
        {...props}
      />
    )
  },
}


export function MDXRenderer({ content }: { content: string }) {
  // Strip frontmatter if present
    if (content.trim().startsWith('---')) {
      const frontmatterEnd = content.indexOf('---', 3)
      if (frontmatterEnd !== -1) {
        content = content.slice(frontmatterEnd + 3).trim()
      }
    }

  const [MDXContent, setMDXContent] = useState<any>(null)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    let cancelled = false

    async function compileMDX() {
      try {
        // Compile MDX to JavaScript
        const compiled = await compile(content, {
          outputFormat: 'function-body',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex as any],
        })

        // Run the compiled code to get the component
        const mod = await run(compiled, {
          ...runtime,
          baseUrl: import.meta.url,
        })

        if (!cancelled) {
          setMDXContent(() => mod.default)
          setError(null)
        }
      } catch (err) {
        console.error('MDX compilation error:', err)
        if (!cancelled) {
          setError(err as Error)
        }
      }
    }

    compileMDX()

    return () => {
      cancelled = true
    }
  }, [content])

  if (error) {
    return (
      <div className="text-red-600 p-4 border border-red-300 rounded">
        <h3 className="font-bold">MDX Compilation Error</h3>
        <pre className="text-sm mt-2 overflow-x-auto">{error.message}</pre>
      </div>
    )
  }

  if (!MDXContent) {
    return <div className="text-muted-foreground">Loading content...</div>
  }

  return (
    <div 
      className="prose max-w-full prose-slate dark:prose-invert md:prose-base"
    >
      <MDXContent components={MDX_COMPONENTS} />
    </div>
  )
}
