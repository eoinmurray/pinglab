import { Link } from "react-router-dom";
import { ModeToggle } from "./ModeToggle";
import { useTheme } from "next-themes";
import { isSimulationRunning } from "../../plugins/cathedral-plugin/src/client";

export function Header() {
  const { resolvedTheme } = useTheme();
  const isRunning = isSimulationRunning()

  return (
    <header className="flex items-center gap-4 px-4 py-2 print:hidden">
      <h1>
        <Link to="/" className="uppercase text-2xl font-mono hover:underline">
        {">"}  
        </Link>
      </h1>

      {isRunning && (
      <div className="px-2 py-1 bg-red-600 font-mono text-xs rounded-md animate-pulse">
        Simulation in progress, this page will reload when finished.
      </div>
      )}

      <div className="flex-1" />
      <h2>
        <Link 
          to="https://github.com/eoinmurray/pinglab"
          target="_blank"
        >
          Github
        </Link>
      </h2>
      <ModeToggle />
    </header>
  )
}
