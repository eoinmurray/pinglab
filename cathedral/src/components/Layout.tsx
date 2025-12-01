import { isSimulationRunning } from "../../plugins/cathedral-plugin/src/client";
import Footer from './Footer';
import { Header } from './Header';

type Props = {
  children: React.ReactNode;
};

export function Layout({ children }: Props) {
  const isRunning = isSimulationRunning();

  return (
    <div className="flex min-h-screen flex-col bg-background">
      {isRunning && (
        <div className="sticky top-0 z-50 px-4 py-2 bg-red-500 text-white font-mono text-xs text-center">
          <span className="inline-flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-current animate-pulse" />
            Simulation in progress, the page will auto-refresh when complete.
          </span>
        </div>
      )}

      <Header />

      <main className="flex-1">
        {children}
      </main>

      <Footer />
    </div>
  );
}
