import { Link } from "react-router-dom";
import { ModeToggle } from "./ModeToggle";
import { SiGithub } from "@icons-pack/react-simple-icons";

export function Header() {
  return (
    <header className="bg-background print:hidden">
      <div className="flex items-center gap-6 py-3 px-4 w-full">
        <Link
          to="/"
          className="font-mono text-lg font-medium tracking-tight hover:text-primary transition-colors"
        >
          pl
        </Link>

        <div className="flex-1" />

        <nav className="flex items-center gap-4">
          <Link
            to="https://github.com/eoinmurray/pinglab"
            target="_blank"
            className="text-muted-foreground hover:text-foreground transition-colors"
            aria-label="GitHub"
          >
            <SiGithub className="h-5 w-5" />
          </Link>
          <ModeToggle />
        </nav>
      </div>
    </header>
  );
}
