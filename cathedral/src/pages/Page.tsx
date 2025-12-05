import { useParams } from "react-router-dom"
import Browser from "@/components/Browser";
import Gallery from "@/components/Gallery";
import File from "@/components/File";
import { useDirectory, findReadme, DirectoryError } from "../../plugins/cathedral-plugin/src/client";
import { FileEntry } from "../../plugins/cathedral-plugin/src/lib";
import Loading from "@/components/Loading";
import PostList from "@/components/PostList";
import { Layout } from "@/components/Layout";
import PageHeader from "@/components/PageHeader";
import { cn } from "@/lib/utils";

function ErrorDisplay({ error, path }: { error: DirectoryError; path: string }) {
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

export function Page() {
  const { "*": path = "." } = useParams();

  const { directory, file, loading, error } = useDirectory(path)

  const isRoot = path === "." || path === "";

  if (error) {
    return <ErrorDisplay error={error} path={path} />;
  }

  if (loading) {
    return (
      <Loading />
    )
  }

  const hasImageChildren = directory?.children
    .some((child): child is FileEntry => {
      return !!child.name.match(/\.(png|jpeg|gif|svg|webp)$/i) && child.type === "file";
    })

  const indexFile = findReadme(directory!);

  let fileToRender = file;
  if (!fileToRender && indexFile && indexFile.type === "file") {
    fileToRender = indexFile;
  }

  return (
    <Layout>
      <title>{`Pinglab ${path}`}</title>
      <main className={cn(
        "flex flex-col gap-6 mb-32",
        isRoot && "pt-20",
      )}>
        {/* Root page - post list */}
        {isRoot && directory && (
          <div className="animate-fade-in">
            <PostList directory={directory}/>
          </div>
        )}

        {/* Non-root pages */}
        {!isRoot && directory && <PageHeader directory={directory} />}

        {!isRoot && fileToRender && (
          <File file={fileToRender} />
        )}

        {!file && hasImageChildren && (
          <Gallery path={path} />
        )}
      </main>
    </Layout>
  )
}
