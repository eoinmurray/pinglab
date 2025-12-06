import { isSimulationRunning } from "../../plugins/cathedral-plugin/src/client";
import Footer from './Footer';
import { Header } from './Header';

type Props = {
  children: React.ReactNode;
};

export function Layout({ children }: Props) {
  const isRunning = isSimulationRunning();

  return (
    <div className="flex min-h-screen flex-col bg-background noise-overlay">
      {/* Simulation status bar - terminal aesthetic */}
      {isRunning && (
        // this should stay red not another color
        <div className="sticky top-0 z-50 px-[var(--page-padding)] py-2 bg-red-500 text-primary-foreground font-mono text-xs text-center tracking-wide">
          <span className="inline-flex items-center gap-3">
            <span className="h-1.5 w-1.5 rounded-full bg-current animate-pulse" />
            <span className="uppercase tracking-widest">simulation running</span>
            <span className="text-primary-foreground/60">Page will auto-refresh on completion</span>
          </span>
        </div>
      )}

      <Header />

      <main className="flex-1 mx-auto w-full max-w-[var(--content-width)] px-[var(--page-padding)]">
        {children}
      </main>

      <Footer />
    </div>
  );
}
