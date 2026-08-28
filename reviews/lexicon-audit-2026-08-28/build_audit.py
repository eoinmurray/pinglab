#!/usr/bin/env python3
"""Rebuild a source-only lexicon audit; never import/run experiment code.

Run from any directory. Only writes generated files next to this script.
Candidate curation lives in candidates.txt; findings.md is authored separately.
"""
from __future__ import annotations

import bisect
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).splitlines()


def write_json(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def link(path, line, label=None):
    return f"[{label or Path(path).stem + ':' + str(line)}]({ROOT / path}:{line})"


def normalize(text):
    # One output character per source character until whitespace is collapsed.
    # Retain exact source offsets, even for terms spanning authored line breaks.
    result, offsets = [], []
    for i, c in enumerate(text):
        c = {"–": "-", "—": "-", "−": "-", "‑": "-", "’": "'", "µ": "μ"}.get(c, c)
        if c.isspace():
            if result and result[-1] == " ":
                continue
            c = " "
        result.append(c)
        offsets.append(i)
    return "".join(result), offsets


def contexts(lines):
    section, fence, bibliography, metadata = "Preamble", False, False, False
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#let meta = ("):
            metadata = True
        if stripped.startswith("#reference-list("):
            bibliography = True
        if stripped.startswith("#let report-body"):
            bibliography = False
        if re.match(r"={2,}\s", stripped):
            section = re.sub(r"\s*<[^>]+>\s*$", "", stripped.lstrip("= "))
        if stripped.startswith("```"):
            fence = not fence
        kind = "text"
        if bibliography:
            kind = "bibliography"
        elif stripped.startswith("//"):
            kind = "comment"
        elif metadata:
            kind = "metadata"
        elif fence or stripped.startswith(("#import", "#let", "let ", "#set", "#show")):
            kind = "code"
        result.append((section, kind))
        if metadata and stripped == ")":
            metadata = False
    return result


def main():
    tracked = command("git", "ls-files", "*.typ")
    visible = command("rg", "--files", "--hidden", "-g", "*.typ", "-g", "!.git")
    paths = sorted(set(tracked + visible))
    corpus = []
    for path in paths:
        p = ROOT / path
        if not p.is_file() or p.is_symlink():
            raise ValueError(f"Non-regular corpus member: {path}")
        source = p.read_text()
        lines = source.splitlines()
        normalized, offsets = normalize(source)
        title = re.search(r'^\s*title:\s*"([^"]+)"', source, re.M)
        corpus.append(dict(path=path, source=source, lines=lines, normalized=normalized,
                           offsets=offsets, starts=[0] + [m.end() for m in re.finditer("\n", source)],
                           contexts=contexts(lines), title=title.group(1) if title else "Shared Typst helper",
                           sha256=hashlib.sha256(p.read_bytes()).hexdigest()))

    candidates = []
    for line in (HERE / "candidates.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        group, term, pattern, recommendation = line.split(" || ")
        candidates.append(dict(id=f"L{len(candidates)+1:03}", group=group, term=term,
                               pattern=pattern, recommendation=recommendation))
    terms = []
    for candidate in candidates:
        regex = re.compile(r"(?<![A-Za-z0-9])(?:" + candidate["pattern"] + r")(?![A-Za-z0-9])", re.I)
        hits = []
        for file in corpus:
            for match in regex.finditer(file["normalized"]):
                start = file["offsets"][match.start()]
                end = file["offsets"][match.end()-1] + 1
                lineno = bisect.bisect_right(file["starts"], start)
                end_line = bisect.bisect_right(file["starts"], end-1)
                section, kind = file["contexts"][lineno-1]
                hits.append(dict(path=file["path"], line=lineno, end_line=end_line,
                                 start=start, end=end, match=file["source"][start:end],
                                 section=section, context=kind,
                                 excerpt=" ".join(file["source"][max(0, start-100):min(len(file["source"]), end+100)].split())))
        files = sorted({h["path"] for h in hits})
        substantive = sorted({h["path"] for h in hits if h["context"] not in {"comment", "metadata", "bibliography"}})
        terms.append(candidate | dict(file_count=len(files), substantive_file_count=len(substantive),
                                      occurrence_count=len(hits), files=files, occurrences=hits))

    # Complete mechanical indexes complement (but do not prove) semantic coverage.
    equations, inline, declarations = [], collections.defaultdict(list), collections.defaultdict(list)
    words = collections.defaultdict(list)
    for file in corpus:
        text = file["source"]
        # Mask raw code/comments for dollar-span extraction while preserving offsets.
        masked = re.sub(r"```[\s\S]*?```|`[^`\n]*`|(?m:^\s*//[^\n]*)",
                        lambda m: "".join("\n" if c == "\n" else " " for c in m.group()), text)
        for match in re.finditer(r"(?<!\\)\$((?:\\.|[^$])*?)(?<!\\)\$", masked, re.S):
            lineno = bisect.bisect_right(file["starts"], match.start())
            eq = text[match.start()+1:match.end()-1]
            equations.append(dict(path=file["path"], line=lineno, expression=eq))
        for lineno, line in enumerate(file["lines"], 1):
            for match in re.finditer(r"(?<!`)`([^`\n]+)`(?!`)", line):
                inline[match.group(1)].append(dict(path=file["path"], line=lineno))
            for match in re.finditer(r"(?:#?let)\s+([A-Za-z][\w-]*)", line):
                declarations[match.group(1)].append(dict(path=file["path"], line=lineno))
            if file["contexts"][lineno-1][1] in {"comment", "bibliography", "metadata", "code"}:
                continue
            cleaned = re.sub(r'"(?:https?://|/)[^"]*"|\$[^$]*\$|#[\w.-]+|<[^>]+>', " ", line)
            for token in re.findall(r"[A-Za-z][A-Za-z-]{2,}", cleaned.lower()):
                words[token].append(dict(path=file["path"], line=lineno))

    def repeated(mapping):
        return [dict(text=k, file_count=len({x['path'] for x in v}), occurrences=v)
                for k, v in sorted(mapping.items()) if len({x['path'] for x in v}) >= 2]

    write_json("corpus.json", dict(scope="Current tracked and non-ignored Typst sources; ignored snapshots, dependencies and retained run payloads excluded.",
                                  files=[{k: f[k] for k in ('path', 'title', 'sha256')} | {'lines':len(f['lines'])} for f in corpus]))
    write_json("occurrences.json", terms)
    math_tokens = collections.defaultdict(list)
    for eq in equations:
        # This is intentionally a lexical index, not a Typst parser or binding resolver.
        expression = re.sub(r"#[A-Za-z][\w.-]*", " ", eq['expression'])
        for token in set(re.findall(r'[A-Za-z]+(?:_[A-Za-z]+)*', expression)):
            math_tokens[token].append(dict(path=eq['path'], line=eq['line']))
    write_json("lexical-index.json", dict(equations=equations, repeated_math_tokens=repeated(math_tokens), repeated_inline_code=repeated(inline),
                                        repeated_declarations=repeated(declarations), repeated_words=repeated(words)))

    identifiers = ["# Repeated symbols and source identifiers", "",
                   "Mechanical index of names occurring in at least two files. Repetition alone is not a defect. "
                   "Math tokens include operators and textual subscripts; they are not resolved semantic variables. "
                   "The [notation audit](notation.md) interprets the important collisions. Locations for a mathematical "
                   "token point to the beginning of its enclosing dollar-delimited expression.", ""]
    for title, mapping in [("Mathematical tokens", math_tokens), ("Inline-code strings", inline),
                           ("Local declarations", declarations)]:
        identifiers += [f"## {title}", "", "| Token or name | Files | Every location |", "| --- | ---: | --- |"]
        for entry in repeated(mapping):
            locations = collections.defaultdict(set)
            for hit in entry['occurrences']:
                locations[hit['path']].add(hit['line'])
            refs = []
            for path, numbers in sorted(locations.items()):
                refs.append(Path(path).stem + ": " + ", ".join(link(path, n, str(n)) for n in sorted(numbers)))
            name = entry['text'].replace('|', '&#124;')
            identifiers.append(f"| `{name}` | {entry['file_count']} | " + "; ".join(refs) + " |")
        identifiers.append("")
    (HERE / "identifiers.md").write_text("\n".join(identifiers) + "\n")

    cross = [t for t in terms if t["substantive_file_count"] >= 2]
    local = [t for t in terms if t["substantive_file_count"] < 2]
    summary = dict(files=len(corpus), lines=sum(len(f['lines']) for f in corpus),
                   candidates=len(terms), cross_file_candidates=len(cross), supplementary_candidates=len(local),
                   matched_occurrences=sum(t['occurrence_count'] for t in terms),
                   math_spans=len(equations), repeated_inline_code=len(repeated(inline)),
                   repeated_declarations=len(repeated(declarations)), repeated_math_tokens=len(repeated(math_tokens)))
    write_json("summary.json", summary)
    table = ["# Candidate lexicon: complete cross-file inventory", "",
             "Generated from `candidates.txt`. Proposed names are recommendations, not adopted policy. "
             "Occurrences include prose, equations, code, metadata and comments; context labels in the evidence are heuristic. "
             "A cross-file candidate must occur outside comments, metadata and bibliography in at least two files. "
             "Counts measure textual reuse, not independent experiments or verified claims.", "",
             "Every row links to all matching locations in the evidence index. Matched spelling variants and exact excerpts are in `occurrences.json`. "
             "Nested categories may overlap; counts must not be summed as independent concepts.", ""]
    groups = list(dict.fromkeys(t['group'] for t in cross))
    for group in groups:
        table += [f"## {group}", "", "| Entry | Term or quantity | Files / matches | Proposed standard and boundary |", "| --- | --- | ---: | --- |"]
        for t in cross:
            if t['group'] != group:
                continue
            table.append(f"| [{t['id']}](evidence.md#{t['id'].lower()}) | {t['term']} | {t['file_count']} / {t['occurrence_count']} | {t['recommendation']} |")
        table.append("")
    table += ["## Supplementary local or reference-only candidates", "",
              "These were considered during the audit but did not meet the cross-file substantive-use rule. "
              "Keep them scoped locally unless future writing reuses them; zero-hit proposals are explicitly visible here.", "",
              "| Entry | Term | Files / matches | Recommendation |", "| --- | --- | ---: | --- |"]
    for t in local:
        table.append(f"| [{t['id']}](evidence.md#{t['id'].lower()}) | {t['term']} | {t['file_count']} / {t['occurrence_count']} | {t['recommendation']} |")
    (HERE / "catalogue.md").write_text("\n".join(table) + "\n")

    evidence = ["# All matched source locations", "", "Generated evidence. Links point to the audited working checkout. "
                "Line numbers are snapshot-specific; `corpus.json` records source hashes. Each listed number is a match-start line; "
                "multiline end locations and every exact excerpt are retained in `occurrences.json`.", ""]
    for t in terms:
        evidence += [f"## {t['id']}", "", f"**{t['term']}** — {t['file_count']} files; {t['occurrence_count']} matches.", "",
                     f"Search expression: `{t['pattern']}`", ""]
        for path in t['files']:
            hits = [h for h in t['occurrences'] if h['path'] == path]
            line_numbers = sorted({h['line'] for h in hits})
            evidence.append(f"- **{path}**: " + ", ".join(link(path, n, str(n)) for n in line_numbers))
        evidence.append("")
    (HERE / "evidence.md").write_text("\n".join(evidence) + "\n")
    inventory = ["# Corpus and reverse index", "", "All current repository Typst sources, including four shared helpers. "
                 "The deprecated exp048 article remains included. No retained run, remote asset or ignored snapshot was read as evidence.", "",
                 "| Source | Title | Lines | Candidate entries |", "| --- | --- | ---: | --- |"]
    for f in corpus:
        ids = [f"[{t['id']}](evidence.md#{t['id'].lower()})" for t in terms if f['path'] in t['files']]
        inventory.append(f"| {link(f['path'], 1, f['path'])} | {f['title']} | {len(f['lines'])} | {', '.join(ids) or 'None'} |")
    (HERE / "inventory.md").write_text("\n".join(inventory) + "\n")
    # Verify all offsets, counts, coverage and the source snapshot before finishing.
    for f in corpus:
        assert hashlib.sha256((ROOT / f['path']).read_bytes()).hexdigest() == f['sha256'], f"Source changed during audit: {f['path']}"
    sources = {f['path']: f['source'] for f in corpus}
    for t in terms:
        assert t['occurrence_count'] == len(t['occurrences'])
        for h in t['occurrences']:
            assert sources[h['path']][h['start']:h['end']] == h['match']
    assert set(tracked).issubset(sources)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
