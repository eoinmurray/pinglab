import { Link } from "react-router-dom";
import { ModeToggle } from "./ModeToggle";

export function Header() {

  return (
    <header className="flex items-center gap-4 px-4 py-2 print:hidden">
      <h1>
        <Link to="/" className="uppercase text-2xl font-mono hover:underline">
        {">"}  
        </Link>
      </h1>

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
