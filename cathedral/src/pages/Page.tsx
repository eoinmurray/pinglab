import { useParams } from "react-router-dom"
import Gallery from "@/components/files/Gallery";
import File from "@/components/files/File";
import { useDirectory, findReadme } from "../../plugins/cathedral-plugin/src/client";
import { FileEntry } from "../../plugins/cathedral-plugin/src/lib";
import Loading from "@/components/Loading";
import PostList from "@/components/PostList";
import { Layout } from "@/components/Layout";
import { cn } from "@/lib/utils";
import { ErrorDisplay } from "@/components/PageError";


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

        {!isRoot && fileToRender && (
          <File file={fileToRender} directory={directory} />
        )}

        {!file && hasImageChildren && (
          <Gallery path={path} />
        )}
      </main>
    </Layout>
  )
}
