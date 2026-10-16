"""CPU dispatch must not inherit a GPU from the shared worker decorator."""

import ast
from pathlib import Path


def test_shared_worker_has_no_default_gpu():
    tree = ast.parse(Path(__file__).with_name("modal_compute.py").read_text())
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "compute"
    )
    decorator = next(
        node
        for node in worker.decorator_list
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "function"
    )
    gpu = next(keyword.value for keyword in decorator.keywords if keyword.arg == "gpu")
    assert ast.literal_eval(gpu) is None
