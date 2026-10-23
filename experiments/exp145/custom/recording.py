"""Load this experiment's dense recording with bounded temporary memory.

Native SNNLab readers verify each block and its checksum. Assemble the validated
blocks directly, storing binary spikes as booleans in memory; scientific exports
remain unchanged. This avoids several simultaneous full float spike copies.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from snnlab.sim import RecordingData


def read_recording(path, scratch):
    path = Path(path)
    scratch.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads(line) for line in (path / "manifest.jsonl").read_text().splitlines()
    ]
    grouped = {}
    for row in rows:
        grouped.setdefault(row["signal"], []).append(row)
    signals, metadata = {}, {}
    for name, blocks in grouped.items():
        blocks.sort(key=lambda row: row["start_step"])
        end = blocks[-1]["end_step"]
        position = 0
        reference = None
        for row in blocks:
            if row["start_step"] != position:
                raise ValueError("recording blocks must be contiguous from step zero")
            source = (path / row["file"]).resolve()
            if not source.is_relative_to(path.resolve()):
                raise ValueError("recording block escapes source directory")
            link = scratch / "block.npz"
            link.unlink(missing_ok=True)
            link.hardlink_to(source)
            (scratch / "manifest.jsonl").write_text(
                json.dumps(dict(row, file="block.npz")) + "\n"
            )
            block = RecordingData.read(scratch).select_batch(0)
            info = block.info[name]
            values = block.signals[name]
            if info.axes != ("time", "cell") or info.step_indices != tuple(
                range(position, row["end_step"])
            ):
                raise ValueError("unsupported recording axes or step coordinates")
            canonical = replace(info, start_step=None, end_step=None, step_indices=None)
            if reference is None:
                reference = canonical
                dtype = bool if name.endswith(".spikes") else values.dtype
                signals[name] = np.empty((end, values.shape[1]), dtype=dtype)
            elif canonical != reference:
                raise ValueError("inconsistent recording block metadata")
            if name.endswith(".spikes") and not np.isin(values, (0, 1)).all():
                raise ValueError("spikes must be binary")
            signals[name][position : row["end_step"]] = values
            position = row["end_step"]
        metadata[name] = replace(
            reference,
            start_step=0,
            end_step=end,
            step_indices=tuple(range(end)),
            array_dtype=str(signals[name].dtype),
            conversion="binary_to_bool"
            if name.endswith(".spikes")
            else reference.conversion,
        )
    return RecordingData(signals, metadata)
