
import { useParams } from "react-router-dom"
import { Layout } from "@/components/Layout";
import Browser from "@/components/Browser";
import Gallery from "@/components/Gallery"; 
import File from "@/components/File";
import { useDirectory, findReadme } from "../../plugins/cathedral-plugin/src/client";
import { FileEntry } from "../../plugins/cathedral-plugin/src/lib";
import { Welcome } from "@/components/Welcome";
import Loading from "@/components/Loading";

export function Page() {
  const { "*": path = "." } = useParams();

  const { directory, file, loading, error } = useDirectory(path)

  if (error) {
    return (
      <main className="min-h-screen bg-background container mx-auto max-w-4xl py-12">
        <p className="text-center text-red-600">{error}</p>
      </main>
    )
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
          {path === '' && (
            <Welcome />
          )}

          {directory && <Browser directory={directory} alwaysOpen={path === ''} />}

          {fileToRender && ( <File file={fileToRender} /> )}

          {!file && hasImageChildren && (
            <Gallery
              path={path}
            />
          )}
      </main>
    </Layout>
  )
}
