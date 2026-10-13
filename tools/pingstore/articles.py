"""Read experiment metadata and explicit pins from authored Astro reports."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from .contracts import PingstoreError


class _ReportLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in keys:
                raise yaml.constructor.ConstructorError(
                    None, None, "report metadata keys must be unique strings", key_node.start_mark
                )
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


def articles(repo: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted((repo / "md" / "experiments").glob("exp[0-9][0-9][0-9]/report.mdx")):
        match = re.match(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|\Z)", path.read_text(), re.S)
        if match is None:
            raise PingstoreError(f"report requires YAML frontmatter: {path}")
        try:
            value = yaml.load(match[1], Loader=_ReportLoader)
        except yaml.YAMLError as exc:
            raise PingstoreError(f"invalid report frontmatter: {path}") from exc
        if not isinstance(value, dict) or value.get("id") != path.parent.name:
            raise PingstoreError(f"report ID must match its experiment: {path}")
        inputs = value.get("inputs", [])
        if not isinstance(inputs, list) or any(
            not isinstance(key, str) or not re.fullmatch(r"exp[0-9]{3}", key)
            for key in inputs
        ) or len(set(inputs)) != len(inputs):
            raise PingstoreError(f"invalid report inputs: {path}")
        pins = value.get("runs", {})
        if not isinstance(pins, dict) or any(
            key not in inputs or not isinstance(run_id, str)
            or not re.fullmatch(r"exp[0-9]{3}-r[0-9]+-(compute|analyse|present)", run_id)
            for key, run_id in pins.items()
        ):
            raise PingstoreError(f"invalid report presentation pins: {path}")
        value["inputs"] = inputs
        value["runs"] = pins
        result[path.parent.name] = value
    return result
