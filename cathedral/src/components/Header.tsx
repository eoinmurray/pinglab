import { Link } from "react-router-dom";
import { ModeToggle } from "./ModeToggle";
import { SiGithub } from "@icons-pack/react-simple-icons";

export function Header() {
  return (
    <header className="z-40 print:hidden">
      <div className="mx-auto w-full px-[var(--page-padding)] flex items-center gap-8 py-4">
        {/* Logo - scholarly monospace */}
        <nav className="flex items-center gap-1">
          <span
            className="rounded-lg px-3 py-1.5 text-xs font-light transition-colors text-muted-foreground hover:bg-accent/50 hover:text-foreground"
          >
            Scroll or use keyboard to navigate.
          </span>
        </nav>
        <div className="flex-1" />

        {/* Navigation */}
        <nav className="flex items-center gap-1">
          <Link
            to="/"
            className="rounded-lg px-3 py-1.5 text-sm font-medium transition-colors text-muted-foreground hover:bg-accent/50 hover:text-foreground"
            aria-label="GitHub"
          >
            Posts
          </Link>
          <Link
            to="/docs"
            className="rounded-lg px-3 py-1.5 text-sm font-medium transition-colors text-muted-foreground hover:bg-accent/50 hover:text-foreground"
            aria-label="GitHub"
          >
            Docs
          </Link>

          <Link
            to="/biblio/docs/README.mdx"
            target="_blank"
            className="rounded-lg px-3 py-1.5 text-sm font-medium transition-colors text-muted-foreground hover:bg-accent/50 hover:text-foreground"
            aria-label="GitHub"
          >
            LLM
          </Link>

          

          <Link
            to="/logs"
            className="rounded-lg px-3 py-1.5 text-sm font-medium transition-colors text-muted-foreground hover:bg-accent/50 hover:text-foreground"
            aria-label="GitHub"
          >
            Logs
          </Link>
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
