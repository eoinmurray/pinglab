"""Typeset diagram labels as self-contained SVG paths using LaTeX and dvisvgm.

Run from any directory with Python 3, latex and dvisvgm on PATH. Edit LABELS
below to change notation; rerunning replaces existing generated label groups.
The SVG keeps its house-style cards and wires, with no runtime font dependency
for mathematical labels. Temporary TeX products are not retained.
"""
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

SVG = 'http://www.w3.org/2000/svg'
XLINK = 'http://www.w3.org/1999/xlink'
ET.register_namespace('', SVG)
ET.register_namespace('xlink', XLINK)
PATH = Path(__file__).with_name('mean-field-network.svg')

# (x, baseline-ish lower edge): (LaTeX, size in diagram units, alignment, colour)
LABELS = {}
for x, y, formula, anchor, colour in [
    (535, 166, r'\mathbf W^{X_E\to E}', 'middle', '#24292F'),
    (535, 598, r'\mathbf W^{X_I\to I}', 'middle', '#24292F'),
    (580, 250, r'\mathbf W^{X\to E}', 'middle', '#24292F'),
    (580, 513, r'\mathbf W^{X\to I}', 'middle', '#24292F'),
    (760, 401, r'\mathbf W^{E\to I}', 'end', '#24292F'),
    (940, 401, r'\mathbf W^{I\to E}', 'start', '#AA5B63'),
    (1055, 94, r'\mathbf W^{E\to E}', 'middle', '#24292F'),
    (1055, 715, r'\mathbf W^{I\to I}', 'middle', '#AA5B63'),
]:
    LABELS[x, y] = (f'${formula}$', 23, anchor, colour)
for pop, title_y, detail_y, foot_y in [('E', 178, 207, 232), ('I', 613, 642, 667)]:
    LABELS[80, title_y] = (rf'\textbf{{Private source population }}$X_{pop}$', 18, 'start', '#24292F')
    LABELS[80, detail_y] = (rf'$N_{pop}$ independent sources $s^{{X_{pop}}}_k$', 17, 'start', '#59636E')
    LABELS[80, foot_y] = (r'Rate $r_{\mathrm{priv}}$ $\cdot$ diagonal $\mathbf W$', 15, 'start', '#59636E')
for pop, y in [('E', 246), ('I', 586)]:
    LABELS[732, y] = (rf'$N_{pop}$ neurons $\cdot$ rate $r_{pop}$', 17, 'start', '#59636E')
for pop, y in [('E', 292), ('I', 557)]:
    LABELS[580, y] = (rf'$M\times N_{pop}$', 16, 'middle', '#59636E')
LABELS[80, 393] = (r'\textbf{External source pool }$X$', 21, 'start', '#24292F')
LABELS[80, 422] = (r'$M$ sources $\cdot$ rate $r_X$', 17, 'start', '#59636E')


def render_label(folder, index, tex, size, x, y, anchor, colour):
    stem = f'label-{index}'
    (folder / f'{stem}.tex').write_text(
        r'\documentclass[10pt,border=0pt]{standalone}' '\n'
        r'\usepackage{amsmath}\usepackage[scaled]{helvet}' '\n'
        r'\renewcommand{\familydefault}{\sfdefault}' '\n'
        r'\begin{document}' + tex + r'\end{document}' '\n'
    )
    subprocess.run(['latex', '-interaction=nonstopmode', '-halt-on-error', f'{stem}.tex'],
                   cwd=folder, check=True, capture_output=True)
    subprocess.run(['dvisvgm', '--no-fonts', '--exact-bbox', f'{stem}.dvi', '-o', f'{stem}.svg'],
                   cwd=folder, check=True, capture_output=True)
    fragment = ET.parse(folder / f'{stem}.svg').getroot()
    left, top, width, height = map(float, fragment.attrib['viewBox'].split())
    scale = size / 10
    offset = {'start': 0, 'middle': width / 2, 'end': width}[anchor]
    group = ET.Element(f'{{{SVG}}}g', {
        'data-math-label': 'true', 'data-x': str(x), 'data-y': str(y),
        'data-tex': tex, 'role': 'img', 'aria-label': tex,
        'fill': colour,
        'transform': f'translate({x-offset*scale:.4f} {y-height*scale:.4f}) scale({scale:g}) translate({-left:g} {-top:g})',
    })
    # Prefix every glyph ID so several independent LaTeX fragments can coexist.
    raw = ''.join(ET.tostring(child, encoding='unicode') for child in fragment)
    ids = re.findall(r'\bid="([^"]+)"', raw)
    for old in sorted(ids, key=len, reverse=True):
        new = f'math{index}-{old}'
        raw = raw.replace(f'id="{old}"', f'id="{new}"').replace(f'#{old}"', f'#{new}"').replace(f'#{old})', f'#{new})')
    wrapper = ET.fromstring(f'<g xmlns="{SVG}" xmlns:xlink="{XLINK}">{raw}</g>')
    group.extend(deepcopy(child) for child in wrapper)
    return group


def main():
    tree = ET.parse(PATH)
    root = tree.getroot()
    found = set()
    with tempfile.TemporaryDirectory(prefix='mean-field-math-') as tmp:
        for index, child in enumerate(list(root)):
            if child.tag == f'{{{SVG}}}text':
                key = int(child.get('x', '-1')), int(child.get('y', '-1'))
            elif child.get('data-math-label'):
                key = int(child.get('data-x')), int(child.get('data-y'))
            else:
                continue
            if key not in LABELS:
                continue
            tex, size, anchor, colour = LABELS[key]
            group = render_label(Path(tmp), index, tex, size, *key, anchor, colour)
            position = list(root).index(child)
            root.remove(child)
            root.insert(position, group)
            found.add(key)
    assert found == set(LABELS), f'Missing label positions: {set(LABELS)-found}'
    ET.indent(tree)
    PATH.write_text(ET.tostring(root, encoding='unicode') + '\n')
    print(f'Typeset {len(found)} labels as embedded SVG paths: {PATH}')


if __name__ == '__main__':
    main()
