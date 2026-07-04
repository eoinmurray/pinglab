"""snn tool package — the pinglab PING engine; entry-point at tool.py.

The submodules import their siblings by bare name (`import models as M`,
`from config import ...`). That resolves automatically when a module is run
as a script (its own directory is sys.path[0]) and under pytest (configured
via `pythonpath` in pyproject.toml). When the package is imported instead
(e.g. `from tools.snn import infer`), this __init__ runs first and puts the
package directory on sys.path so those bare imports still resolve.

This module also re-exports the symbols imported by tests.
"""

import sys as _sys
from pathlib import Path as _Path

# When the engine is imported as the package `tools.snn` (rather than run as a
# script or via pythonpath), put the package dir on sys.path so the submodules'
# bare sibling imports (`import models`, `from config import ...`) still resolve.
_pkg_dir = str(_Path(__file__).resolve().parent)
if _pkg_dir in _sys.path:
    _sys.path.remove(_pkg_dir)
_sys.path.insert(0, _pkg_dir)
