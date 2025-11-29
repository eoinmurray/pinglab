import { isSimulationRunning } from "../../plugins/cathedral-plugin/src/client";
import Footer from './Footer';
import { Header } from './Header';

type Props = {
  children: React.ReactNode;
};

export function Layout({ children }: Props) {
  const isRunning = isSimulationRunning()

  return (
    <div className="flex min-h-screen flex-col">
      {isRunning && (
        <div className="sticky top-0 z-50 px-3 py-2 bg-red-600 font-mono text-xs text-white animate-pulse">
          Simulation in progress, this page will reload when finished.
        </div>
      )}

      <Header />

      <main className="flex-1">
        {children}
      </main>

      <Footer />
    </div>
  )
}
