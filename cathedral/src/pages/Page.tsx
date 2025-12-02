
import { useParams } from "react-router-dom"
import Browser from "@/components/Browser";
import Gallery from "@/components/Gallery"; 
import File from "@/components/File";
import { useDirectory, findReadme, DirectoryError } from "../../plugins/cathedral-plugin/src/client";
import { FileEntry } from "../../plugins/cathedral-plugin/src/lib";
import { Welcome } from "@/components/Welcome";
import Loading from "@/components/Loading";
import PostList from "@/components/PostList";
import { Layout } from "@/components/Layout";

function ErrorDisplay({ error, path }: { error: DirectoryError; path: string }) {
  const containerClass = "min-h-screen bg-background container mx-auto max-w-4xl py-12 px-4";

  switch (error.type) {
    case 'config_not_found':
      return (
        <main className={containerClass}>
          <div className="text-center">
            <h1 className="text-2xl font-semibold mb-4">Setup Required</h1>
            <p className="text-muted-foreground">
              Could not find <code className="bg-muted px-1.5 py-0.5 rounded">.cathedral.json</code>
            </p>
            <p className="text-muted-foreground mt-2">
              Run the cathedral build script to generate the directory index.
            </p>
          </div>
        </main>
      );

    case 'path_not_found':
      return (
        <main className={containerClass}>
          <div className="text-center">
            <h1 className="text-4xl font-bold mb-4">404</h1>
            <p className="text-xl text-muted-foreground mb-2">Page not found</p>
            <p className="text-muted-foreground">
              The path <code className="bg-muted px-1.5 py-0.5 rounded">{path}</code> does not exist.
            </p>
          </div>
        </main>
      );

    case 'parse_error':
      return (
        <main className={containerClass}>
          <div className="text-center">
            <h1 className="text-2xl font-semibold text-red-600 mb-4">Configuration Error</h1>
            <p className="text-muted-foreground">
              Failed to parse <code className="bg-muted px-1.5 py-0.5 rounded">.cathedral.json</code>
            </p>
            <p className="text-muted-foreground mt-2">
              The file may be corrupted. Try rebuilding.
            </p>
          </div>
        </main>
      );

    case 'fetch_error':
    default:
      return (
        <main className={containerClass}>
          <div className="text-center">
            <h1 className="text-2xl font-semibold text-red-600 mb-4">Error</h1>
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
      <main className="flex flex-col gap-8 mx-auto max-w-4xl p-4 md:px-0 mb-24">
          {isRoot && <Welcome />}

          {!isRoot && directory && <Browser directory={directory} />}

          {isRoot && directory && <PostList directory={directory}/>}

          {!isRoot && fileToRender && ( <File file={fileToRender} /> )}

          {!file && hasImageChildren && (
            <Gallery
              path={path}
            />
          )}
      </main>
    </Layout>
  )
}
