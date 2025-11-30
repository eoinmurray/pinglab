import { useState, useEffect } from 'react'
import { compile, run } from '@mdx-js/mdx'
import * as runtime from 'react/jsx-runtime'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import Gallery from './Gallery'

function generateId(children: unknown): string {
  return children
    ?.toString()
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, '')
    .replace(/\s+/g, '-') ?? ''
}

const MDX_COMPONENTS = {
  Gallery,
  h1: (props: React.HTMLAttributes<HTMLHeadingElement>) => {
    const id = generateId(props.children)
    return <h1 id={id} className="text-3xl font-semibold tracking-tight mt-10 mb-4" {...props} />
  },
  h2: (props: React.HTMLAttributes<HTMLHeadingElement>) => {
    const id = generateId(props.children)
    return <h2 id={id} className="text-2xl font-semibold tracking-tight mt-8 mb-3 pb-2 border-b" {...props} />
  },
  h3: (props: React.HTMLAttributes<HTMLHeadingElement>) => {
    const id = generateId(props.children)
    return <h3 id={id} className="text-xl font-medium mt-6 mb-2" {...props} />
  },
  h4: (props: React.HTMLAttributes<HTMLHeadingElement>) => {
    const id = generateId(props.children)
    return <h4 id={id} className="text-lg font-medium mt-4 mb-2" {...props} />
  },
  h5: (props: React.HTMLAttributes<HTMLHeadingElement>) => {
    const id = generateId(props.children)
    return <h5 id={id} className="text-base font-medium mt-3 mb-1" {...props} />
  },
  pre: (props: React.HTMLAttributes<HTMLPreElement>) => (
    <pre className="not-prose w-full overflow-x-auto p-4 text-sm bg-muted/50 rounded-lg border font-mono" {...props} />
  ),
  code: (props: React.HTMLAttributes<HTMLElement> & { className?: string }) => {
    const isInline = !props.className?.includes('language-')
    if (isInline) {
      return <code className="font-mono text-[0.9em] bg-muted px-1.5 py-0.5 rounded" {...props} />
    }
    return <code {...props} />
  },
}

export function RuntimeMDX({ content }: { content: string }) {
  const [MDXContent, setMDXContent] = useState<React.ComponentType<{ components: typeof MDX_COMPONENTS }> | null>(null)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    let cancelled = false

    async function compileMDX() {
      try {
        const compiled = await compile(content, {
          outputFormat: 'function-body',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex as never],
        })

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
      <div className="text-destructive p-4 border border-destructive/30 rounded-lg bg-destructive/5">
        <h3 className="font-semibold">MDX Compilation Error</h3>
        <pre className="text-sm mt-2 overflow-x-auto font-mono">{error.message}</pre>
      </div>
    )
  }

  if (!MDXContent) {
    return <div className="text-muted-foreground">Loading content...</div>
  }

  return (
    <div className="prose max-w-full prose-slate dark:prose-invert md:prose-base">
      <MDXContent components={MDX_COMPONENTS} />
    </div>
  )
}
