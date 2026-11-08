#!/usr/bin/env python3
"""make_case.py <case dir> [--refused] [--set FIELD=VALUE ...]: for a case dir holding splats.bin (the
fixture, tools/tests/stream_fixture.py) and scene.stream (its `convert.py stream`), writes what
stream_test checks the decode against and prints stream_test's arguments for it (but --chunk and
--expect-code, which test.sh gives):
- reference.bin, the Splats the decode must give, and tolerances.bin, a Splat of tolerances for each:
  - f32 fields: the fixture's values, bit for bit (tolerance 0);
  - f16 fields: the fixture's values rounded by numpy's float16 (to nearest, ties to even), bit for
    bit;
  - quantized fields: low + stepSize q in float64 (q read from the stream's bytes, low and stepSize
    from its header), rounded to float32, within 4 float32 ulps of max(|low|, |high|), high =
    low + stepSize (2^bits - 1);
  - the SH above the stream's degree: 0, bit for bit (the working buffer starts zeroed);
  with --refused, every value 0, bit for bit: a refused header decodes nothing;
- with --set, the stream's header patched first: FIELD is a StreamHeader field (splatCount) or a
  field group's (fields.0.encoding), VALUE a uint.
This is the test's own reference, written from the format's definition (docs/data-format.md), not
the decoder's code. Needs numpy."""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from splatdata import splats, stream  # noqa: E402

ULPS = 4  # the quantized fields' tolerance, in float32 ulps of max(|low|, |high|)
SIZES = {"f32": 4, "f16": 2, "u16": 2, "u8": 1}


def slots() -> dict:
    """Each group's components' float indices in a Splat (std430, 60 floats), in component order."""
    def at(name, first, n):
        return list(range(splats.SPLAT_DTYPE.fields[name][1] // 4 + first,
                          splats.SPLAT_DTYPE.fields[name][1] // 4 + first + n))
    return {"position": at("position", 0, 3), "scale": at("logScale", 0, 3), "rotation": at("rotation", 0, 4),
            "opacity": at("opacity", 0, 1), "dc": at("sh", 0, 3), "sh": at("sh", 3, 45)}


def patch(data: bytearray, assignment: str) -> None:
    name, value = assignment.split("=")
    header = np.frombuffer(data, stream.HEADER_DTYPE, 1)  # a view: writes go to data
    parts = name.split(".")
    if parts[0] == "fields":
        header["fields"][0, int(parts[1])][parts[2]] = int(value)
    else:
        header[parts[0]] = int(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("--refused", action="store_true", help="the header is refused: expect nothing decoded")
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    args = parser.parse_args()
    stream_path = os.path.join(args.dir, "scene.stream")
    data = bytearray(open(stream_path, "rb").read())
    for assignment in args.set:
        patch(data, assignment)
    with open(stream_path, "wb") as f:
        f.write(data)

    original = np.frombuffer(open(os.path.join(args.dir, "splats.bin"), "rb").read(), "<f4")
    n = len(original) // (splats.SPLAT_DTYPE.itemsize // 4)
    original = original.reshape(n, -1)
    reference = np.zeros_like(original)
    tolerances = np.zeros_like(original)
    if not args.refused:
        h = stream.read_header(bytes(data))
        if int(h["splatCount"]) != n:
            sys.exit(f"make_case.py: the stream has {int(h['splatCount'])} splats, splats.bin {n}")
        group_slots = slots()
        for g, group in enumerate(stream.GROUPS):
            f = h["fields"][g]
            encoding = stream.ENCODING_NAMES[int(f["encoding"])]
            columns = group_slots[group][:int(f["components"])]
            if encoding == "f32":
                reference[:, columns] = original[:, columns]
            elif encoding == "f16":
                reference[:, columns] = original[:, columns].astype(np.float16).astype(np.float32)
            else:
                start, stride = ((int(h["shOffset"]), int(h["shStride"])) if group == "sh"
                                 else (int(h["baseOffset"]), int(h["baseStride"])))
                size = SIZES[encoding]
                records = np.frombuffer(bytes(data), np.uint8, n * stride, start).reshape(n, stride)
                levels = (1 << (8 * size)) - 1
                for c, column in enumerate(columns):
                    at = int(f["offset"]) + c * size
                    q = records[:, at:at + size].copy().view("<u2" if size == 2 else "u1")[:, 0].astype(np.float64)
                    r = 0 if int(f["sharedRange"]) else c
                    low, step = np.float64(f["low"][r]), np.float64(f["stepSize"][r])
                    reference[:, column] = (low + step * q).astype(np.float32)
                    tolerances[:, column] = ULPS * np.spacing(np.float32(max(abs(low), abs(low + step * levels))))
    reference.tofile(os.path.join(args.dir, "reference.bin"))
    tolerances.tofile(os.path.join(args.dir, "tolerances.bin"))
    print(f"--stream {stream_path} --reference {os.path.join(args.dir, 'reference.bin')} "
          f"--tolerances {os.path.join(args.dir, 'tolerances.bin')} --splats {n} --stream-bytes {len(data)}")


if __name__ == "__main__":
    main()
