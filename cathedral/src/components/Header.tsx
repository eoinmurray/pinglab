import { Link } from "react-router-dom";
import { ModeToggle } from "./ModeToggle";
import { SiGithub } from "@icons-pack/react-simple-icons";

export function Header() {
  return (
    <header className="z-40 print:hidden">
      <div className="mx-auto w-full px-[var(--page-padding)] flex items-center gap-8 py-4">
        <nav className="flex items-center gap-1">
          <Link
            to="/"
            className="rounded-lg font-mono py-1.5 text-sm font-medium text-muted-foreground hover:underline"
          >
            pl
          </Link>
        </nav>

        <div className="flex-1" />

        {/* Navigation */}
        <nav className="flex items-center gap-2">
          <Link
            to="https://github.com/eoinmurray/pinglab"
            target="_blank"
            className="text-muted-foreground/70 hover:text-foreground transition-colors duration-300"
            aria-label="GitHub"
          >
            <SiGithub className="h-4 w-4" />
          </Link>
          <ModeToggle />
        </nav>
      </div>
    </header>
  );
}
