#!/usr/bin/env python3
"""Prints splat_render's arguments for a camera of a scene directory (docs/data-format.md):

    build/splat_render/splat_render $(tools/scene_args.py <scene dir> <camera>) [more arguments]

ZGAL programs can't read the manifest or join paths, so this does: the splat and camera files,
the splat count, the SH degree and the camera's index and size. The camera is an index or a name
from the manifest's cameras. Paths are printed as given (relative to the current directory when
the scene's is); a path with whitespace can't go through $(...) unquoted, so it is refused.

Only the standard library.
"""

import json
import os
import sys
from typing import List


def scene_args(scene: str, camera: str) -> List[str]:
    with open(os.path.join(scene, "manifest.json")) as f:
        manifest = json.load(f)
    cameras = manifest["cameras"]
    names = [c.get("name") for c in cameras]
    if camera in names:
        index = names.index(camera)
    else:
        try:
            index = int(camera)
        except ValueError:
            raise SystemExit(f"scene_args: {scene} has no camera {camera!r}")
        if not 0 <= index < len(cameras):
            raise SystemExit(f"scene_args: camera {index} is past {scene}'s {len(cameras)} cameras")
    files = [os.path.join(scene, manifest["files"][k]) for k in ("splats", "cameras")]
    for path in files:
        if any(c.isspace() for c in path):
            raise SystemExit(f"scene_args: {path!r} has whitespace, which $(...) would split")
    cam = cameras[index]
    return ["--splats", files[0], "--cameras", files[1], "--splat-count", str(manifest["splatCount"]),
            "--sh-degree", str(manifest["shDegree"]), "--camera", str(index), "--width", str(cam["width"]),
            "--height", str(cam["height"])]


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit(__doc__)
    print(" ".join(scene_args(argv[0], argv[1])))


if __name__ == "__main__":
    main()
