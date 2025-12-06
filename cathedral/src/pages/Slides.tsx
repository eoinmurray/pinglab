
import { useParams } from "react-router-dom"
import { useFileContent } from "../../plugins/cathedral-plugin/src/client";
import { Slides as RenderSlides } from "@/components/files/Slides";
import Loading from "@/components/Loading";
import { Layout } from "@/components/Layout";


export function Slides() {
  const { "path": path = "." } = useParams();
  const filePath = `${path}/SLIDES.mdx`

  const { content, loading, error } = useFileContent(filePath);
  if (loading) {
    return (
      <Loading />
    )
  }

  if (error) {
    return (
      <main className="min-h-screen bg-background container mx-auto max-w-4xl py-12">
        <p className="text-center text-red-600">{error}</p>
      </main>
    )
  }
  
  return (
    <Layout>
      <title>{`Pinglab ${path}`}</title>
      {content && <RenderSlides content={content} />}
    </Layout>
  )
}
