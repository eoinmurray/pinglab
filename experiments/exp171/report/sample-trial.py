"""Read display statistics from retained NPZ bytes using the reporting runtime."""

import ast
import io
import json
import math
import struct
import sys
import zipfile
from array import array

archive = zipfile.ZipFile(io.BytesIO(sys.stdin.buffer.read()))
image = int(sys.argv[2])


def read(name):
    with archive.open(name + ".npy") as f:
        assert f.read(6) == b"\x93NUMPY"
        version = f.read(2)
        size = struct.unpack(
            "<H" if version[0] == 1 else "<I", f.read(2 if version[0] == 1 else 4)
        )[0]
        header = ast.literal_eval(f.read(size).decode("latin1"))
        assert not header["fortran_order"]
        dtype = header["descr"]
        assert dtype in ("<f4", "<f8", "<i8", "<i4", "<u4")
        values = array(
            {"<f4": "f", "<f8": "d", "<i8": "q", "<i4": "i", "<u4": "I"}[dtype]
        )
        values.frombytes(f.read())
        if sys.byteorder != "little":
            values.byteswap()
        return header["shape"], values


def mean(x):
    return math.fsum(x) / len(x)


def row(name):
    shape, values = read(name)
    return values[position * shape[1] : (position + 1) * shape[1]]


kind = sys.argv[1]
if kind in ("dataset", "calibration"):
    _, ids = read("ids")
    position = ids.index(image)
    if kind == "dataset":
        result = {
            "amplitude": mean(row("amplitude")),
            "rate": mean(row("poisson_rate_hz")),
        }
    else:
        sd = row("target_sd")
        expected_sd = row("expected_sd")
        result = {
            "teacherMean": mean(row("target_mean")),
            "teacherVariance": mean([x * x for x in sd]),
            "teacherSd": mean(sd),
            "teacherRate": mean(row("teacher_i_rate_hz")),
            "expectedMean": mean(row("expected_mean")),
            "expectedVariance": mean([x * x for x in expected_sd]),
            "expectedSd": mean(expected_sd),
        }
else:
    shape, values = read("g_artificial")
    steps, channels = shape
    sums = [0.0] * channels
    squares = [0.0] * channels
    for start in range(0, len(values), channels):
        for j, value in enumerate(values[start : start + channels]):
            sums[j] += value
            squares[j] += value * value
    variances = [
        max(0, squares[j] / steps - (sums[j] / steps) ** 2) for j in range(channels)
    ]
    result = {
        "actualMean": mean(sums) / steps,
        "actualVariance": mean(variances),
        "actualSd": mean([math.sqrt(v) for v in variances]),
    }
assert all(math.isfinite(x) for x in result.values())
print(json.dumps(result, allow_nan=False))
