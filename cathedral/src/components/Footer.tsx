export default function Footer() {
  return (
    <footer className="mx-auto w-full max-w-[var(--content-width)] px-[var(--page-padding)] py-12 print:hidden">
      <div className="flex items-center justify-center gap-6 text-muted-foreground/40">
        <div className="h-px flex-1 bg-border/30" />
        <span className="font-mono text-[10px] tracking-widest uppercase">pinglab</span>
        <div className="h-px flex-1 bg-border/30" />
      </div>
    </footer>
  )
}
