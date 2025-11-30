import { MDXProvider } from '@mdx-js/react'
import { ReactNode } from 'react'
import Gallery from './Gallery'
import 'katex/dist/katex.min.css'

function generateId(children: unknown): string {
  return children
    ?.toString()
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, '')
    .replace(/\s+/g, '-') ?? ''
}

// Components available to all MDX files
const components = {
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

export function MDXProviderWrapper({ children }: { children: ReactNode }) {
  return (
    <MDXProvider components={components}>
      <div className="prose max-w-full prose-slate dark:prose-invert md:prose-base">
        {children}
      </div>
    </MDXProvider>
  )
}
