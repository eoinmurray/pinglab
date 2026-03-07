#!/usr/bin/env python3
"""Resolve gallery comments in .typ files to actual #figure() calls with real image paths."""

from __future__ import annotations

import re
import glob as globmod
from pathlib import Path


POSTS_DIR = Path("src/posts")


def resolve_globs(base_path: str, globs: list[str]) -> list[str]:
    """Resolve glob patterns against actual files on disk. Returns dark variants only."""
    matches = []
    for g in globs:
        pattern = str(POSTS_DIR / base_path / g)
        found = sorted(globmod.glob(pattern))
        # Filter to dark variants only (skip light)
        dark = [f for f in found if '_light' not in f]
        if dark:
            matches.extend(dark)
        elif found:
            # No dark variant, use whatever we found
            matches.extend(found)
    # Make paths relative to POSTS_DIR
    return [str(Path(m).relative_to(POSTS_DIR)) for m in matches]


def process_file(typ_path: Path):
    """Process a single .typ file, resolving gallery comments to figures."""
    content = typ_path.read_text()
    lines = content.split('\n')
    out = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Detect gallery comment block:
        # // Gallery: glob1, glob2, ...
        # // Path: optional/path  (optional)
        # // #figure(image("..."), caption: [...])
        if stripped.startswith('// Gallery:'):
            glob_line = stripped[len('// Gallery:'):].strip()
            globs = [g.strip().rstrip(',') for g in glob_line.split(',') if g.strip()]

            # Check for optional Path line
            base_path = None
            caption = ''
            comment_lines = [i]
            j = i + 1

            while j < len(lines):
                next_stripped = lines[j].strip()
                if next_stripped.startswith('// Path:'):
                    base_path = next_stripped[len('// Path:'):].strip()
                    comment_lines.append(j)
                    j += 1
                elif next_stripped.startswith('// #figure('):
                    # Extract caption from the commented figure
                    cap_match = re.search(r'caption:\s*\[([^\]]*)\]', next_stripped)
                    if cap_match:
                        caption = cap_match.group(1)
                    comment_lines.append(j)
                    j += 1
                    break
                elif next_stripped == '':
                    # Blank line might follow
                    break
                else:
                    break

            # If no explicit path, try to infer from the file's #let json() calls
            if base_path is None:
                # Look back for #let config = json("_artifacts/xxx/config.json")
                for prev_line in lines[:i]:
                    m = re.search(r'json\("([^"]+)/(?:config|results)\.json"\)', prev_line)
                    if m:
                        base_path = m.group(1)

            if base_path is None:
                # Fallback: use study name from filename
                study = typ_path.stem
                base_path = f"_artifacts/{study}"

            # Resolve the globs
            image_paths = resolve_globs(base_path, globs)

            if image_paths:
                if len(image_paths) == 1:
                    # Single image figure
                    out.append(f'#figure(')
                    out.append(f'  image("{image_paths[0]}"),')
                    if caption:
                        out.append(f'  caption: [{caption}],')
                    out.append(f')')
                else:
                    # Multiple images: use a grid
                    # Determine reasonable column count
                    n = len(image_paths)
                    if n <= 3:
                        ncols = n
                    elif n <= 6:
                        ncols = 3
                    elif n <= 12:
                        ncols = 4
                    else:
                        ncols = 5

                    out.append(f'#figure(')
                    out.append(f'  grid(')
                    out.append(f'    columns: {ncols},')
                    out.append(f'    gutter: 4pt,')
                    for img in image_paths:
                        out.append(f'    image("{img}"),')
                    out.append(f'  ),')
                    if caption:
                        out.append(f'  caption: [{caption}],')
                    out.append(f')')
                out.append('')
            else:
                # No images found - keep as comment but note it
                out.append(f'// Gallery (no images found): {glob_line}')
                if caption:
                    out.append(f'// caption: {caption}')
                out.append('')

            # Skip all consumed comment lines
            i = max(comment_lines) + 1
            continue

        out.append(line)
        i += 1

    typ_path.write_text('\n'.join(out))


def main():
    typ_files = sorted(POSTS_DIR.glob('*.typ'))

    for f in typ_files:
        content = f.read_text()
        if '// Gallery:' in content:
            count = content.count('// Gallery:')
            print(f'Processing {f.name} ({count} galleries)')
            process_file(f)
        else:
            print(f'Skipping {f.name} (no galleries)')


if __name__ == '__main__':
    main()
