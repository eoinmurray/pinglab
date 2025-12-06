
import { DirectoryError } from "../../plugins/cathedral-plugin/src/client";

export function ErrorDisplay({ error, path }: { error: DirectoryError; path: string }) {
  const containerClass = "min-h-screen bg-background container mx-auto max-w-[var(--content-width)] py-24 px-[var(--page-padding)]";

  switch (error.type) {
    case 'config_not_found':
      return (
        <main className={containerClass}>
          <div className="text-center space-y-4">
            <h1 className="text-2xl font-semibold tracking-tight">Setup Required</h1>
            <p className="text-muted-foreground">
              Could not find <code className="font-mono text-sm bg-muted px-2 py-1">.cathedral.json</code>
            </p>
            <p className="text-muted-foreground/70 text-sm">
              Run the cathedral build script to generate the directory index.
            </p>
          </div>
        </main>
      );

    case 'path_not_found':
      return (
        <main className={containerClass}>
          <div className="text-center space-y-4">
            <h1 className="font-mono text-6xl tracking-tighter text-muted-foreground/30">404</h1>
            <p className="text-lg text-foreground">Page not found</p>
            <p className="text-muted-foreground text-sm">
              <code className="font-mono bg-muted px-2 py-1">{path}</code>
            </p>
          </div>
        </main>
      );

    case 'parse_error':
      return (
        <main className={containerClass}>
          <div className="text-center space-y-4">
            <h1 className="text-2xl font-semibold text-destructive">Configuration Error</h1>
            <p className="text-muted-foreground">
              Failed to parse <code className="font-mono text-sm bg-muted px-2 py-1">.cathedral.json</code>
            </p>
          </div>
        </main>
      );

    case 'fetch_error':
    default:
      return (
        <main className={containerClass}>
          <div className="text-center space-y-4">
            <h1 className="text-2xl font-semibold text-destructive">Error</h1>
            <p className="text-muted-foreground">{error.message}</p>
          </div>
        </main>
      );
  }
}
