import { useParams } from "react-router-dom"
import { useDirectory } from "../../plugins/cathedral-plugin/src/client";
import Loading from "@/components/Loading";
import PostList from "@/components/PostList";
import { ErrorDisplay } from "@/components/PageError";
import { RunningBar } from "@/components/RunningBar";
import { Header } from "@/components/Header";

export function Home() {
  const { "*": path = "." } = useParams();
  const { directory, loading, error } = useDirectory(path)

  if (error) {
    return <ErrorDisplay error={error} path={path} />;
  }

  if (loading) {
    return (
      <Loading />
    )
  }

  return (
    <div className="flex min-h-screen flex-col bg-background noise-overlay">
      <RunningBar />
      <Header />
      <main className="flex-1 mx-auto w-full max-w-[var(--content-width)] px-[var(--page-padding)]">
        <title>{`Pinglab ${path}`}</title>
        <main className="flex flex-col gap-6 mb-32 mt-32">
          {directory && (
            <div className="animate-fade-in">
              <PostList directory={directory}/>
            </div>
          )}
        </main>
      </main>
    </div>
  )
}
