#!/usr/bin/env python3
"""tile_test's scenes (test.sh), from tools/make_synthetic.py's.

  scenes.py args <scene dir> <camera>
      Prints tile_test's arguments for a camera of a scene, from its manifest: the files, the splat
      count, the SH degree, the camera's size and how many splats at the end are degenerate.
  scenes.py degenerate <scene dir> <out dir>
      Copies a scene and appends splats the preprocess must cull, in front of camera 0: a zero
      quaternion (normalizing it is 0/0: NaN), logScale 100 (exp overflows to inf: inf * 0 is NaN)
      and logScale 30 (finite, but the 2D covariance's determinant overflows to inf).

Only the standard library, so it runs wherever test.sh does.
"""

import json
import os
import shutil
import struct
import sys

# docs/data-format.md's Splat: position, opacity, logScale, 4 bytes of padding, rotation, sh[48]
SPLAT = struct.Struct("<3ff3f4x4f48f")
assert SPLAT.size == 240

# (position, logScale, rotation (w, x, y, z)): in front of camera 0 (at the origin, looking down +z)
DEGENERATE = [
    ((0.05, -0.05, 3.0), (-3.0, -3.0, -3.0), (0.0, 0.0, 0.0, 0.0)),
    ((-0.05, 0.05, 3.5), (100.0, 100.0, 100.0), (1.0, 0.0, 0.0, 0.0)),
    ((0.0, 0.1, 5.0), (30.0, 30.0, 30.0), (1.0, 0.0, 0.0, 0.0)),
]


def manifest(scene):
    with open(os.path.join(scene, "manifest.json")) as f:
        return json.load(f)


def args(scene, camera):
    m = manifest(scene)
    cam = m["cameras"][camera]
    return [
        "--splats", os.path.join(scene, m["files"]["splats"]),
        "--cameras", os.path.join(scene, m["files"]["cameras"]),
        "--splat-count", str(m["splatCount"]),
        "--sh-degree", str(m["shDegree"]),
        "--camera", str(camera),
        "--width", str(cam["width"]),
        "--height", str(cam["height"]),
        "--culled-tail", str((m.get("source") or {}).get("degenerate", 0)),
    ]


def degenerate(scene, out):
    shutil.copytree(scene, out)
    m = manifest(out)
    with open(os.path.join(out, m["files"]["splats"]), "ab") as f:
        for position, log_scale, rotation in DEGENERATE:
            f.write(SPLAT.pack(*position, 2.0, *log_scale, *rotation, *([0.5] * 48)))
    m["splatCount"] += len(DEGENERATE)
    m["source"] = {"tile_test": "degenerate", "degenerate": len(DEGENERATE), "from": m.get("source")}
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(m, f, indent=1)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "args":
        print(" ".join(args(sys.argv[2], int(sys.argv[3]))))
    elif len(sys.argv) == 4 and sys.argv[1] == "degenerate":
        degenerate(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
