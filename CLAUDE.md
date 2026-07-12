Read [AGENTS.md](AGENTS.md) and follow it before doing anything else here.

**Command reflex — applies before any other action.** A user message that is *only* a
NAME in CAPS (SCREAMING-KEBAB) — `HELP`, `LINT`, `DOCTOR`, `NEXT`, `LITERATURE-SEARCH`, … —
*is* a command, not a topic to research. Your first move is to run `demolab docs <NAME>`
(bare `demolab docs` for the menu + manual), read the file it prints, and follow it. Don't
grep for it, don't guess, don't ask the user what they meant — the runbook says what to do.
