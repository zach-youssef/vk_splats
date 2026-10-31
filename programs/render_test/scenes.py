#!/usr/bin/env python3
"""render_test's arguments for a camera of a scene (tools/make_synthetic.py's), from its manifest.

  scenes.py <scene dir> <camera>
      The files, the splat count, the SH degree and the camera's size; for an S1 scene also the
      oracle's background and --oracle.

Only the standard library, so it runs wherever test.sh does.
"""

import json
import os
import sys


def args(scene, camera):
    with open(os.path.join(scene, "manifest.json")) as f:
        m = json.load(f)
    cam = m["cameras"][camera]
    out = [
        "--splats", os.path.join(scene, m["files"]["splats"]),
        "--cameras", os.path.join(scene, m["files"]["cameras"]),
        "--splat-count", str(m["splatCount"]),
        "--sh-degree", str(m["shDegree"]),
        "--camera", str(camera),
        "--width", str(cam["width"]),
        "--height", str(cam["height"]),
    ]
    source = m.get("source") or {}
    if source.get("scene") == "s1":
        r, g, b = source["oracle"]["background"]
        out += ["--oracle", "--background-r", repr(r), "--background-g", repr(g), "--background-b", repr(b)]
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    print(" ".join(args(sys.argv[1], int(sys.argv[2]))))
