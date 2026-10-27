"""Retain compiled circuit figures through explicit presentation ancestry.

Native simulation snapshots are captured directly by compute.py. This module
only copies existing presentation assets; it never reconstructs a network.
"""


def copy_circuits(run):
    """Copy only circuit assets from this presentation's explicit compute ancestry."""
    import shutil

    from pingstore.library import source_run

    seen = set()
    files = set()

    def visit(reference):
        identity = reference["run_id"]
        if identity in seen:
            return
        seen.add(identity)
        source = source_run(run.directory.parents[1], identity, reference=reference)
        if source.record["experiment"] != run.record["experiment"]:
            return
        if source.record["stage"] == "compute":
            for path in source.export.glob("circuit*.svg"):
                target = run.export / path.name
                if target.exists() and target.read_bytes() != path.read_bytes():
                    raise ValueError(f"conflicting pinned circuit: {path.name}")
                shutil.copyfile(path, target)
                files.add(path.name)
        else:
            for parent in source.record["inputs"].values():
                visit(parent)

    for reference in run.record["inputs"].values():
        visit(reference)
    return sorted(files)
