"""Presentation-only revisions of retained evidence, without changing samples."""

import copy
import xml.etree.ElementTree as ET

from pingstore.contracts import PingstoreError

SVG = "http://www.w3.org/2000/svg"


def historical_svg(source, destination, *, move_legend=False):
    """Remove the known producer stamp and optionally separate the legend.

    Paths, axes and glyph coordinates are preserved; only the legend receives
    a translation into an added margin. Callers retain both file hashes.
    """
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.fromstring(source.read_bytes(), parser=parser)
    if root.tag != f"{{{SVG}}}svg":
        raise PingstoreError("historical figure is not SVG")
    stamps = []
    for parent in root.iter():
        for child in list(parent):
            if child.tag == f"{{{SVG}}}g" and any(
                item.tag is ET.Comment
                and (item.text or "").strip() == "exp033-numerics"
                for item in child
            ):
                stamps.append((parent, child))
    if len(stamps) != 1:
        raise PingstoreError("expected exactly one historical exp033 stamp")
    stamps[0][0].remove(stamps[0][1])
    operations = ["remove producer stamp; preserve scientific paths"]
    if move_legend:
        legends = [el for el in root.iter() if el.get("id") == "legend_1"]
        if len(legends) != 1 or legends[0].get("transform"):
            raise PingstoreError("unexpected historical legend layout")
        x, y, width, height = map(float, root.attrib["viewBox"].split())
        if x != 0 or y != 0 or not root.attrib["height"].endswith("pt"):
            raise PingstoreError("unexpected historical figure bounds")
        width_text = root.attrib["viewBox"].split()[2]
        root.set("viewBox", f"0 -60 {width_text} {height + 60:g}")
        root.set("height", f"{height + 60:g}pt")
        legends[0].set("transform", "translate(0 -60)")
        operations.append("translate legend upward 60 SVG units into added margin")
    ET.register_namespace("", SVG)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    return operations


def article_numbers(numbers):
    """Qualify criterion labels, preserving every result and Boolean decision."""
    result = copy.deepcopy(numbers)
    labels = (
        "Reference 4D Hopf in the gamma band",
        "Sampled onset is consistent with supercriticality",
        "Tested two-rate QSS reduction rings down",
        "Three variables suffice in the tested QSS ring family",
    )
    if len(result["success_criteria"]) != len(labels):
        raise PingstoreError("unexpected exp033 scientific criteria")
    for criterion, label in zip(result["success_criteria"], labels):
        criterion["label"] = label
    return result
