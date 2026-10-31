#!/usr/bin/env python3
"""render_test's arguments for a camera of a scene (tools/make_synthetic.py's), from its manifest.

  scenes.py <scene dir> <camera>
      tools/scene_args.py's (the files, the splat count, the SH degree and the camera's index and
      size); for an S1 scene also the oracle's background and --oracle.

Only the standard library, so it runs wherever test.sh does.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from scene_args import scene_args  # noqa: E402


def args(scene, camera):
    out = scene_args(scene, str(camera))
    with open(os.path.join(scene, "manifest.json")) as f:
        source = json.load(f).get("source") or {}
    if source.get("scene") == "s1":
        r, g, b = source["oracle"]["background"]
        out += ["--oracle", "--background-r", repr(r), "--background-g", repr(g), "--background-b", repr(b)]
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    print(" ".join(args(sys.argv[1], int(sys.argv[2]))))
