import Gallery from '@/components/files/Gallery'
import { ParameterTable } from '@/components/files/ParameterTable'

function generateId(children: unknown): string {
  return children
    ?.toString()
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, '')
    .replace(/\s+/g, '-') ?? ''
}

// Shared MDX components - lab notebook / coder aesthetic
export const mdxComponents = {
  ParameterTable,
  Gallery,
  SlideGallery: Gallery, // Alias for backwards compatibility

  // Headings - clean sans-serif
  h1: (props: React.HTMLAttributes<HTMLHeadingElement>) => {
    const id = generateId(props.children)
    return (
      <h1
        id={id}
        className="text-2xl font-semibold tracking-tight mt-12 mb-4 first:mt-0"
        {...props}
      />
    )
  },
  h2: (props: React.HTMLAttributes<HTMLHeadingElement>) => {
    const id = generateId(props.children)
    return (
      <h2
        id={id}
        className="text-xl font-semibold tracking-tight mt-10 mb-3 pb-2 border-b border-border"
        {...props}
      />
    )
  },
  h3: (props: React.HTMLAttributes<HTMLHeadingElement>) => {
    const id = generateId(props.children)
    return (
      <h3
        id={id}
        className="text-lg font-medium tracking-tight mt-8 mb-2"
        {...props}
      />
    )
  },
  h4: (props: React.HTMLAttributes<HTMLHeadingElement>) => {
    const id = generateId(props.children)
    return (
      <h4
        id={id}
        className="text-base font-medium mt-6 mb-2"
        {...props}
      />
    )
  },
  h5: (props: React.HTMLAttributes<HTMLHeadingElement>) => {
    const id = generateId(props.children)
    return (
      <h5
        id={id}
        className="text-sm font-medium mt-4 mb-1"
        {...props}
      />
    )
  },

  // Code blocks - IDE/terminal style
  pre: (props: React.HTMLAttributes<HTMLPreElement>) => (
    <pre
      className="not-prose w-full overflow-x-auto p-4 text-sm bg-muted/50 border border-border rounded-md font-mono my-6"
      {...props}
    />
  ),
  code: (props: React.HTMLAttributes<HTMLElement> & { className?: string }) => {
    const isInline = !props.className?.includes('language-')
    if (isInline) {
      return (
        <code
          className="font-mono text-[0.85em] bg-muted px-1.5 py-0.5 rounded text-primary"
          {...props}
        />
      )
    }
    return <code {...props} />
  },

  // Blockquote
  blockquote: (props: React.HTMLAttributes<HTMLQuoteElement>) => (
    <blockquote
      className="border-l-2 border-primary pl-4 my-6 text-muted-foreground"
      {...props}
    />
  ),

  // Lists
  ul: (props: React.HTMLAttributes<HTMLUListElement>) => (
    <ul className="my-4 ml-6 list-disc marker:text-muted-foreground" {...props} />
  ),
  ol: (props: React.HTMLAttributes<HTMLOListElement>) => (
    <ol className="my-4 ml-6 list-decimal marker:text-muted-foreground" {...props} />
  ),
  li: (props: React.HTMLAttributes<HTMLLIElement>) => (
    <li className="mt-1.5" {...props} />
  ),

  // Links
  a: (props: React.AnchorHTMLAttributes<HTMLAnchorElement>) => (
    <a
      className="text-primary hover:underline underline-offset-2"
      {...props}
    />
  ),

  // Tables
  table: (props: React.TableHTMLAttributes<HTMLTableElement>) => (
    <div className="not-prose my-6 overflow-x-auto border border-border rounded-md">
      <table className="w-full text-sm border-collapse" {...props} />
    </div>
  ),
  thead: (props: React.HTMLAttributes<HTMLTableSectionElement>) => (
    <thead className="bg-muted/50" {...props} />
  ),
  tbody: (props: React.HTMLAttributes<HTMLTableSectionElement>) => (
    <tbody {...props} />
  ),
  tr: (props: React.HTMLAttributes<HTMLTableRowElement>) => (
    <tr className="border-b border-border last:border-b-0" {...props} />
  ),
  th: (props: React.ThHTMLAttributes<HTMLTableCellElement>) => (
    <th
      className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider"
      {...props}
    />
  ),
  td: (props: React.TdHTMLAttributes<HTMLTableCellElement>) => (
    <td className="px-4 py-3 align-top" {...props} />
  ),

  // Horizontal rule
  hr: (props: React.HTMLAttributes<HTMLHRElement>) => (
    <hr className="my-8 border-t border-border" {...props} />
  ),

  // Paragraph
  p: (props: React.HTMLAttributes<HTMLParagraphElement>) => (
    <p className="leading-relaxed mb-4 last:mb-0" {...props} />
  ),

  // Strong/emphasis
  strong: (props: React.HTMLAttributes<HTMLElement>) => (
    <strong className="font-semibold" {...props} />
  ),
  em: (props: React.HTMLAttributes<HTMLElement>) => (
    <em className="italic" {...props} />
  ),
}
