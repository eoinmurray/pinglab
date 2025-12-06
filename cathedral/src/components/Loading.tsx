export default function Loading() {
  return (
    <main className="min-h-screen flex items-center justify-center bg-background">
      <div className="flex flex-col items-center gap-6 animate-fade-in">
        <div className="flex items-center gap-3">
          <div className="w-1.5 h-1.5 bg-primary/60 rounded-full animate-pulse" />
          <div className="w-1.5 h-1.5 bg-primary/40 rounded-full animate-pulse" style={{ animationDelay: '150ms' }} />
          <div className="w-1.5 h-1.5 bg-primary/20 rounded-full animate-pulse" style={{ animationDelay: '300ms' }} />
        </div>
        <p className="font-mono text-xs text-muted-foreground/50 tracking-widest uppercase">
          loading
        </p>
      </div>
    </main>
  )
}
