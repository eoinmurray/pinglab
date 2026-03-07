#!/usr/bin/env python3
"""Convert MDX/MD posts to Typst (.typ) format."""

from __future__ import annotations

import re
import sys
import json
from pathlib import Path


def convert_frontmatter(lines: list[str]) -> tuple[list[str], int]:
    """Extract YAML frontmatter and return Typst metadata lines + end index."""
    if not lines or lines[0].strip() != "---":
        return [], 0

    end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end == -1:
        return [], 0

    title = ""
    date = ""
    description = ""
    for line in lines[1:end]:
        if line.startswith("title:"):
            title = line.split(":", 1)[1].strip().strip('"')
        elif line.startswith("date:"):
            date = line.split(":", 1)[1].strip().strip('"')
        elif line.startswith("description:"):
            description = line.split(":", 1)[1].strip().strip('"')

    meta = []
    meta.append(f'// title: {title}')
    meta.append(f'// date: {date}')
    if description:
        meta.append(f'// description: {description}')
    meta.append("")
    return meta, end + 1


def convert_latex_to_typst(latex: str) -> str:
    """Convert LaTeX math to Typst math notation."""
    s = latex.strip()

    # Alignment environments
    s = re.sub(r'\\begin\{align\*?\}', '', s)
    s = re.sub(r'\\end\{align\*?\}', '', s)
    s = re.sub(r'\\begin\{aligned\}', '', s)
    s = re.sub(r'\\end\{aligned\}', '', s)

    # Cases environment
    s = re.sub(r'\\begin\{cases\}', 'cases(', s)
    s = re.sub(r'\\end\{cases\}', ')', s)

    # Subscripts/superscripts FIRST (so nested {} inside \frac are resolved)
    for _ in range(3):
        s = re.sub(r'_\{([^{}]*)\}', r'_(\1)', s)
        s = re.sub(r'\^\{([^{}]*)\}', r'^(\1)', s)

    # Fractions: \frac{a}{b} -> frac(a, b)
    # Handle nested fracs by doing multiple passes
    for _ in range(5):
        s = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'frac(\1, \2)', s)

    # \text{...} -> "..."
    s = re.sub(r'\\text\{([^}]*)\}', r'"\1"', s)
    s = re.sub(r'\\mathrm\{([^}]*)\}', r'"\1"', s)
    s = re.sub(r'\\mathbf\{([^}]*)\}', r'bold(\1)', s)

    # Bold: \mathbf{x} -> bold(x), \boldsymbol -> bold
    s = re.sub(r'\\boldsymbol\{([^}]*)\}', r'bold(\1)', s)
    s = re.sub(r'\\textbf\{([^}]*)\}', r'bold(\1)', s)

    # Tilde: \tilde{x} -> tilde(x)
    s = re.sub(r'\\tilde\{([^}]*)\}', r'tilde(\1)', s)

    # Bar: \bar{x} -> overline(x)
    s = re.sub(r'\\bar\s*(\w)', r'overline(\1)', s)
    s = re.sub(r'\\bar\{([^}]*)\}', r'overline(\1)', s)

    # Hat: \hat{x} -> hat(x)
    s = re.sub(r'\\hat\{([^}]*)\}', r'hat(\1)', s)

    # Vectors/bold
    s = re.sub(r'\\mathbf\{([^}]*)\}', r'bold(\1)', s)
    s = re.sub(r'\\bm\{([^}]*)\}', r'bold(\1)', s)

    # \sqrt{x} -> sqrt(x)
    s = re.sub(r'\\sqrt\{([^}]*)\}', r'sqrt(\1)', s)

    # Operators
    s = re.sub(r'\\sum_\{([^}]*)\}\^\{([^}]*)\}', r'sum_(\1)^(\2)', s)
    s = re.sub(r'\\sum_\{([^}]*)\}', r'sum_(\1)', s)
    s = re.sub(r'\\sum_(\w)', r'sum_\1', s)
    s = re.sub(r'\\prod_\{([^}]*)\}\^\{([^}]*)\}', r'product_(\1)^(\2)', s)
    s = re.sub(r'\\exp', 'exp', s)
    s = re.sub(r'\\log', 'log', s)
    s = re.sub(r'\\ln', 'ln', s)
    s = re.sub(r'\\max', 'max', s)
    s = re.sub(r'\\min', 'min', s)

    # Greek letters (common ones)
    greek = [
        'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
        'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'pi', 'rho', 'sigma',
        'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega',
        'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Phi', 'Psi', 'Omega',
    ]
    for g in greek:
        s = re.sub(rf'\\{g}(?![a-zA-Z])', g, s)

    # Partial derivative
    s = re.sub(r'\\partial', 'partial', s)

    # Arrows
    s = s.replace('\\rightarrow', 'arrow.r')
    s = s.replace('\\leftarrow', 'arrow.l')
    s = s.replace('\\Rightarrow', 'arrow.r.double')
    s = s.replace('\\Leftarrow', 'arrow.l.double')
    s = s.replace('\\leftrightarrow', 'arrow.l.r')

    # Relations
    s = s.replace('\\approx', 'approx')
    s = s.replace('\\sim', 'tilde')
    s = s.replace('\\neq', 'eq.not')
    s = s.replace('\\leq', 'lt.eq')
    s = s.replace('\\geq', 'gt.eq')
    s = s.replace('\\gg', 'gt.double')
    s = s.replace('\\ll', 'lt.double')
    s = s.replace('\\propto', 'prop')
    s = s.replace('\\in', 'in')
    s = s.replace('\\infty', 'infinity')
    s = s.replace('\\nabla', 'nabla')

    # Binary ops
    s = s.replace('\\cdot', 'dot')
    s = s.replace('\\times', 'times')
    s = s.replace('\\pm', 'plus.minus')

    # Dots
    s = s.replace('\\ldots', 'dots')
    s = s.replace('\\cdots', 'dots.c')

    # Spacing commands -> just space
    s = re.sub(r'\\[,;!]\s*', ' ', s)
    s = re.sub(r'\\quad', ' ', s)
    s = re.sub(r'\\qquad', '  ', s)

    # Final pass for any remaining subscript/superscript braces
    s = re.sub(r'_\{([^}]*)\}', r'_(\1)', s)
    s = re.sub(r'\^\{([^}]*)\}', r'^(\1)', s)

    # \left( \right) -> just ( )
    s = re.sub(r'\\left\s*([(\[|])', r'\1', s)
    s = re.sub(r'\\right\s*([)\]|])', r'\1', s)
    s = re.sub(r'\\left\s*\\\\', '\\\\', s)
    s = re.sub(r'\\right\s*\\\\', '\\\\', s)

    # \mathbf{1} -> bold(1)
    s = re.sub(r'\\mathbf\{([^}]*)\}', r'bold(\1)', s)
    # \mathbb{R} etc
    s = re.sub(r'\\mathbb\{([^}]*)\}', r'bb(\1)', s)

    # \operatorname{...} -> op("...")
    s = re.sub(r'\\operatorname\{([^}]*)\}', r'op("\1")', s)

    # Double backslash (line break in align) -> \
    s = s.replace('\\\\', '\\\n')

    # &= alignment -> &=
    # (Typst uses & for alignment too, so this is mostly fine)

    # Remove remaining single backslashes before unknown commands
    # (careful not to remove valid typst escapes)

    return s


def convert_js_expression(expr: str, config_var: str, results_var: str) -> str:
    """Convert JS template expression like {String(results?.xxx)} to Typst."""
    expr = expr.strip()

    # Remove outer String() wrapper
    expr = re.sub(r'^String\((.*)\)$', r'\1', expr)

    # Handle ternary: a ? b : "c" - use last occurrence of ? to handle complex cond
    # Find the ternary by looking for the pattern with : "..."
    ternary = re.search(r'\?\s*(.+?)\s*:\s*"([^"]*)"$', expr)
    if ternary:
        true_expr = ternary.group(1).strip()
        fallback = ternary.group(2)
        # Try to convert the true expression
        converted = convert_js_access(true_expr, config_var, results_var)
        if converted:
            return converted
        return fallback

    # Handle ?? fallback: a ?? "b" - use last ??
    nullish = re.search(r'\?\?\s*"([^"]*)"$', expr)
    if nullish:
        access = expr[:nullish.start()].strip()
        fallback = nullish.group(1)
        converted = convert_js_access(access, config_var, results_var)
        if converted:
            return converted
        return fallback

    # Direct access
    converted = convert_js_access(expr, config_var, results_var)
    if converted:
        return converted

    return expr


def convert_js_access(access: str, config_var: str, results_var: str) -> str | None:
    """Convert JS property access to Typst JSON access. Returns None if can't convert."""
    # Remove optional chaining
    clean = access.replace('?.[', '[').replace('?.', '.')

    # Handle method calls - strip them for static output
    clean = re.sub(r'\.toLocaleString\(\)', '', clean)
    clean = re.sub(r'\.toFixed\(\d+\)', '', clean)

    # Handle Math.round(...)
    clean = re.sub(r'Math\.round\((.+?)\)', r'\1', clean)

    # Handle array access like nodes?.find(...)
    if '.find(' in clean:
        # Try to extract: nodes.find((n) => n.id === "E").size
        m = re.search(r'(config|results)\.(\w+)\.find\(\((\w+)\)\s*=>\s*\3\.(\w+)\s*===\s*"([^"]+)"\)\.(\w+)', clean)
        if m:
            root = m.group(1)
            var = results_var if root == 'results' else config_var
            # Can't do find in Typst easily, return as comment
            return None
        return None

    # Handle arithmetic: (65 / results.cm_backward_scale)
    arith = re.match(r'\(?\s*(\d+(?:\.\d+)?)\s*/\s*(config|results)\.(\w+)\s*\)?', clean)
    if arith:
        return None  # Leave as placeholder

    # Handle arithmetic with parenthesized expressions
    if '/' in clean and ('results.' in clean or 'config.' in clean):
        return None

    # Simple property access: results.xxx or config.xxx.yyy
    if clean.startswith('results.') or clean.startswith('config.'):
        parts = clean.split('.')
        root = parts[0]
        var = results_var if root == 'results' else config_var
        rest_parts = parts[1:]
        # Handle array index like nodes[2] or bare [2]
        converted_parts = []
        for p in rest_parts:
            if not p:  # skip empty parts
                continue
            m = re.match(r'(\w+)\[(\d+)\]', p)
            if m:
                converted_parts.append(f'{m.group(1)}.at({m.group(2)})')
            elif re.match(r'^\[(\d+)\]$', p):
                idx = re.match(r'^\[(\d+)\]$', p).group(1)
                if converted_parts:
                    converted_parts[-1] = f'{converted_parts[-1]}.at({idx})'
                else:
                    converted_parts.append(f'at({idx})')
            else:
                converted_parts.append(p)
        path = '.'.join(converted_parts)
        return f'#{var}.{path}'

    return None


def extract_gallery_props(text: str) -> dict:
    """Extract props from a VeslxGallery JSX component."""
    props = {}

    # path
    m = re.search(r'path=\{([^}]+)\}', text)
    if m:
        props['path'] = m.group(1).strip().strip('"')

    # globs
    m = re.search(r'globs=\{\[([^\]]+)\]', text)
    if m:
        raw = m.group(1)
        props['globs'] = [g.strip().strip('"').strip("'") for g in raw.split(',')]

    # caption
    m = re.search(r'caption="([^"]*)"', text)
    if m:
        props['caption'] = m.group(1)

    # captionLabel
    m = re.search(r'captionLabel="([^"]*)"', text)
    if m:
        props['caption_label'] = m.group(1)

    # size
    m = re.search(r'size="([^"]*)"', text)
    if m:
        props['size'] = m.group(1)

    return props


def gallery_to_typst(props: dict, artifact_path: str) -> list[str]:
    """Convert VeslxGallery props to Typst figure comment block."""
    lines = []
    path = props.get('path', '')
    globs = props.get('globs', [])
    caption = props.get('caption', '')
    label = props.get('caption_label', '')
    size = props.get('size', '')

    # Resolve the path
    if not path or path == 'path':
        resolved_path = artifact_path
    else:
        resolved_path = path.strip('"').strip("'")

    # Generate a comment showing what images this references
    glob_str = ', '.join(globs)
    lines.append(f'// Gallery: {glob_str}')
    if resolved_path != artifact_path:
        lines.append(f'// Path: {resolved_path}')

    # Generate actual image references for each glob
    if globs:
        first_glob = globs[0]
        # Replace glob wildcards with a concrete guess
        img_name = first_glob.replace('*', 'dark')
        img_path = f'{resolved_path}/{img_name}'
        lines.append(f'// #figure(image("{img_path}"), caption: [{caption}])')

    lines.append('')
    return lines


def convert_md_table(table_lines: list[str], config_var: str, results_var: str) -> list[str]:
    """Convert a markdown table to a Typst table."""
    if len(table_lines) < 2:
        return table_lines

    # Parse header
    header = [c.strip() for c in table_lines[0].strip('|').split('|')]
    cols = len(header)

    # Skip separator line (line 1)
    rows = []
    for line in table_lines[2:]:
        cells = [c.strip() for c in line.strip('|').split('|')]
        rows.append(cells)

    def process_cell(cell: str) -> str:
        """Process a table cell: convert JS expressions, inline latex, bold."""
        c = process_inline_js(cell, config_var, results_var)
        c = convert_inline_latex(c)
        c = re.sub(r'\*\*([^*]+)\*\*', r'*\1*', c)
        c = c.replace('\\*', '*')
        return c

    out = []
    out.append(f'#table(')
    out.append(f'  columns: {cols},')
    # Header cells
    header_cells = ', '.join(f'[{process_cell(h)}]' for h in header)
    out.append(f'  {header_cells},')
    # Data rows
    for row in rows:
        # Pad row to correct number of columns
        while len(row) < cols:
            row.append('')
        cells = ', '.join(f'[{process_cell(c)}]' for c in row)
        out.append(f'  {cells},')
    out.append(')')
    out.append('')
    return out


def process_inline_js(line: str, config_var: str, results_var: str) -> str:
    """Replace {String(...)} and {expr} JS expressions in a line."""
    def replacer(m):
        expr = m.group(1)
        converted = convert_js_expression(expr, config_var, results_var)
        if converted and converted.startswith('#'):
            return converted
        return str(converted)

    # Match {String(...)} with balanced parens - handle nested parens
    def find_string_exprs(s):
        result = []
        i = 0
        while i < len(s):
            # Look for {String(
            if s[i:].startswith('{String('):
                start = i
                depth = 0
                j = i
                while j < len(s):
                    if s[j] == '{':
                        depth += 1
                    elif s[j] == '}':
                        depth -= 1
                        if depth == 0:
                            result.append((start, j + 1, s[start + 1:j]))
                            break
                    j += 1
                i = j + 1
            else:
                i += 1
        return result

    # Process {String(...)} expressions
    exprs = find_string_exprs(line)
    for start, end, expr in reversed(exprs):
        converted = convert_js_expression(expr, config_var, results_var)
        if converted and converted.startswith('#'):
            line = line[:start] + converted + line[end:]
        else:
            line = line[:start] + str(converted) + line[end:]

    # Match simple {results.xxx} / {config.xxx}
    line = re.sub(r'\{(results\.[^}]+)\}', replacer, line)
    line = re.sub(r'\{(config\.[^}]+)\}', replacer, line)

    # Handle (1000 / results.E.xxx).toFixed(1) type expressions
    line = re.sub(r'\{\(1000\s*/\s*results\.([^)]+)\)\.toFixed\(\d+\)\}', r'#results.\1', line)

    return line


def convert_inline_latex(line: str) -> str:
    """Convert inline $...$ LaTeX to Typst math."""
    # Find all $...$ segments (not $$)
    parts = []
    i = 0
    while i < len(line):
        if line[i] == '$' and (i + 1 >= len(line) or line[i + 1] != '$'):
            # Find closing $
            end = line.find('$', i + 1)
            if end != -1:
                latex = line[i + 1:end]
                typst_math = convert_latex_to_typst(latex)
                parts.append(f'${typst_math}$')
                i = end + 1
                continue
        parts.append(line[i])
        i += 1
    return ''.join(parts)


def convert_file(input_path: Path, output_path: Path):
    """Convert a single MDX/MD file to Typst."""
    content = input_path.read_text()
    lines = content.split('\n')

    out = []
    i = 0

    # Determine study name from filename
    study_name = input_path.stem

    # Process frontmatter
    meta_lines, skip_to = convert_frontmatter(lines)
    out.extend(meta_lines)
    i = skip_to

    # Track what JSON files are imported
    config_var = None
    results_var = None
    has_path = False
    artifact_path = f"_artifacts/{study_name}"

    # First pass: scan ALL lines for imports/exports to collect metadata
    skip_lines = set()
    for j in range(i, len(lines)):
        stripped = lines[j].strip()

        if stripped.startswith('export const path'):
            m = re.search(r'"([^"]*)"', stripped)
            if m:
                artifact_path = m.group(1)
            has_path = True
            skip_lines.add(j)
        elif stripped.startswith('import config'):
            config_var = 'config'
            skip_lines.add(j)
        elif stripped.startswith('import results'):
            results_var = 'results'
            skip_lines.add(j)
        elif '<VeslxFrontMatter' in stripped:
            skip_lines.add(j)

    # Add JSON imports at top if needed
    json_imports = []
    if config_var:
        json_imports.append(f'#let config = json("{artifact_path}/config.json")')
    if results_var:
        json_imports.append(f'#let results = json("{artifact_path}/results.json")')
    if json_imports:
        out.extend(json_imports)
        out.append('')

    # Process remaining lines
    in_display_math = False
    math_buffer = []
    in_gallery = False
    gallery_buffer = []
    in_table = False
    table_buffer = []
    in_html_block = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip lines already consumed by first pass
        if i in skip_lines:
            i += 1
            continue

        # Skip remaining export/import lines
        if stripped.startswith('export const') or stripped.startswith('import '):
            i += 1
            continue

        # Handle VeslxGallery (multi-line JSX)
        if '<VeslxGallery' in stripped:
            in_gallery = True
            gallery_buffer = [stripped]
            if '/>' in stripped:
                # Single line gallery
                in_gallery = False
                props = extract_gallery_props('\n'.join(gallery_buffer))
                out.extend(gallery_to_typst(props, artifact_path))
            i += 1
            continue

        if in_gallery:
            gallery_buffer.append(stripped)
            if '/>' in stripped:
                in_gallery = False
                props = extract_gallery_props('\n'.join(gallery_buffer))
                out.extend(gallery_to_typst(props, artifact_path))
            i += 1
            continue

        # Handle VeslxPostList
        if '<VeslxPostList' in stripped:
            out.append(f'// VeslxPostList: {stripped}')
            if '/>' not in stripped:
                # multi-line
                while i + 1 < len(lines) and '/>' not in lines[i]:
                    i += 1
            i += 1
            continue

        # Handle VeslxPostListItem
        if '<VeslxPostListItem' in stripped:
            # Extract props
            buf = stripped
            while '/>' not in buf and i + 1 < len(lines):
                i += 1
                buf += ' ' + lines[i].strip()
            title_m = re.search(r'title="([^"]*)"', buf)
            desc_m = re.search(r'description="([^"]*)"', buf)
            href_m = re.search(r'href="([^"]*)"', buf)
            title = title_m.group(1) if title_m else ''
            desc = desc_m.group(1) if desc_m else ''
            href = href_m.group(1) if href_m else ''
            out.append(f'- *{title}* — {desc}')
            if href:
                out.append(f'  // href: {href}')
            i += 1
            continue

        # Handle HTML blocks
        if stripped.startswith('<div') or stripped.startswith('<pre') or stripped.startswith('<a '):
            out.append(f'// {stripped}')
            i += 1
            continue
        if stripped.startswith('</div') or stripped.startswith('</pre') or stripped.startswith('</a'):
            out.append(f'// {stripped}')
            i += 1
            continue

        # Handle display math $$...$$
        if stripped.startswith('$$') and not in_display_math:
            if stripped.endswith('$$') and len(stripped) > 2:
                # Single-line display math
                latex = stripped[2:-2]
                typst_math = convert_latex_to_typst(latex)
                out.append('$')
                out.append(f'  {typst_math}')
                out.append('$')
                out.append('')
                i += 1
                continue
            else:
                in_display_math = True
                math_buffer = []
                # Check if there's content after $$
                rest = stripped[2:].strip()
                if rest:
                    math_buffer.append(rest)
                i += 1
                continue

        if in_display_math:
            if stripped.endswith('$$'):
                rest = stripped[:-2].strip()
                if rest:
                    math_buffer.append(rest)
                full_latex = '\n'.join(math_buffer)
                typst_math = convert_latex_to_typst(full_latex)
                out.append('$')
                for mline in typst_math.split('\n'):
                    out.append(f'  {mline.strip()}')
                out.append('$')
                out.append('')
                in_display_math = False
                i += 1
                continue
            else:
                math_buffer.append(line)
                i += 1
                continue

        # Handle markdown tables
        if '|' in stripped and stripped.startswith('|') and stripped.endswith('|'):
            if not in_table:
                in_table = True
                table_buffer = []
            table_buffer.append(stripped)
            # Check if next line is also a table row
            if i + 1 < len(lines):
                next_stripped = lines[i + 1].strip()
                if not (next_stripped.startswith('|') and next_stripped.endswith('|')):
                    # End of table
                    in_table = False
                    out.extend(convert_md_table(table_buffer, config_var or 'config', results_var or 'results'))
            else:
                in_table = False
                out.extend(convert_md_table(table_buffer, config_var or 'config', results_var or 'results'))
            i += 1
            continue

        # Headings: # -> =
        heading_match = re.match(r'^(#{1,6})\s+(.*)', line)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2)
            title = process_inline_js(title, config_var or 'config', results_var or 'results')
            title = convert_inline_latex(title)
            # Convert markdown bold in heading
            title = re.sub(r'\*\*([^*]+)\*\*', r'*\1*', title)
            out.append('')
            out.append(f'{"=" * level} {title}')
            out.append('')
            i += 1
            continue

        # Regular text line - process inline elements
        processed = line

        # Process JS expressions
        processed = process_inline_js(processed, config_var or 'config', results_var or 'results')

        # Convert inline LaTeX
        processed = convert_inline_latex(processed)

        # Convert bold: **text** -> *text*
        processed = re.sub(r'\*\*([^*]+)\*\*', r'*\1*', processed)

        # Convert __ bold __: __text__ -> *text*
        processed = re.sub(r'__([^_]+)__', r'*\1*', processed)

        # Convert inline code: `code` stays as `code` in typst (same syntax)

        # Convert markdown links: [text](url) -> #link("url")[text]
        processed = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'#link("\2")[\1]', processed)

        # Convert \* escaped asterisks
        processed = processed.replace('\\*', '*')

        # Convert --- to em dash
        if processed.strip() == '---':
            processed = '---'

        out.append(processed)
        i += 1

    return '\n'.join(out)


def main():
    posts_dir = Path('src/posts')

    # Find all .md and .mdx files
    files = sorted(list(posts_dir.glob('*.md')) + list(posts_dir.glob('*.mdx')))

    for f in files:
        out_path = f.with_suffix('.typ')
        print(f'Converting {f.name} -> {out_path.name}')
        result = convert_file(f, out_path)
        out_path.write_text(result)
        print(f'  Written {out_path}')

    print(f'\nConverted {len(files)} files.')


if __name__ == '__main__':
    main()
