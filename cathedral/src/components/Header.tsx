import { Link } from "react-router-dom";
import { ModeToggle } from "./ModeToggle";
import { SiGithub } from "@icons-pack/react-simple-icons";

export function Header() {
  return (
    <header className="sticky top-0 z-40 print:hidden">
      <div className="mx-auto w-full px-[var(--page-padding)] flex items-center gap-8 py-4">
        {/* Logo - scholarly monospace */}
        <Link
          to="/"
          className="group flex items-center gap-2"
        >
          <span className="font-mono text-sm tracking-widest text-muted-foreground group-hover:text-foreground transition-colors duration-300">
            pl
          </span>
        </Link>

        <div className="flex-1" />

        {/* Navigation */}
        <nav className="flex items-center gap-6">
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
