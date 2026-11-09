#!/usr/bin/env python3
"""Prints a program's arguments for a scene directory (docs/data-format.md):

    build/splat_render/splat_render $(tools/scene_args.py <scene dir> <camera>) [more arguments]
    build/splat_train/splat_train $(tools/scene_args.py --train <scene dir>) [more arguments]
    build/splat_view/splat_view $(tools/scene_args.py --view <scene.stream>) [more arguments]
    tools/scene_args.py --query <url dir> <any of the above's arguments>

ZGAL programs can't read the manifest or join paths, so this does. For splat_render: the splat
and camera files, the splat count, the SH degree and the camera's index and size; the camera is
an index or a name from the manifest's cameras. For splat_train (--train): the splat, camera and
image files, the splat and camera counts, the SH degree, the largest camera size among those with
an image, the pixels the images hold (the file's size / 4) and the extent; the scene must have
images, in one batch (images_0.bin), as splat_train reads one. For splat_view (--view): the stream
file, its splat count (the header's) and its size in bytes, which splat_view needs before it reads
the stream (they size its buffers); splat_view checks the header itself. Paths are printed as given
(relative to the current directory when the scene's is); a path with whitespace can't go through
$(...) unquoted, so it is refused.

--query prints the same arguments as a program page's query string (its options as name=value,
positional arguments as arg=value, in order), with each path made <url dir>/<its file name>: a page
test serves the scene's files there (programs/<name>/test-page.sh).

--train also refuses a scene older than format version 2 (whose cameras don't mark the test
split, so every camera would be trained on) and a camera buffer whose heldOut flags disagree with
the manifest's test cameras: splat_train reads the split from the buffer.

Only the standard library.
"""

import json
import os
import struct
import sys
import urllib.parse
from typing import List

# The scene format version whose Camera buffer carries the split (heldOut, docs/data-format.md)
TRAIN_FORMAT_VERSION = 2
CAMERA_BYTES = 112
HELD_OUT_OFFSET = 108


class PathArg(str):
    """An argument that is a path (so --query can make it a URL)."""


def read_manifest(scene: str) -> dict:
    with open(os.path.join(scene, "manifest.json")) as f:
        return json.load(f)


def checked_path(path: str) -> PathArg:
    if any(c.isspace() for c in path):
        raise SystemExit(f"scene_args: {path!r} has whitespace, which $(...) would split")
    return PathArg(path)


def scene_files(scene: str, manifest: dict) -> List[str]:
    """The splat and camera files' paths."""
    return [checked_path(os.path.join(scene, manifest["files"][k])) for k in ("splats", "cameras")]


def scene_args(scene: str, camera: str) -> List[str]:
    manifest = read_manifest(scene)
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
    files = scene_files(scene, manifest)
    cam = cameras[index]
    return ["--splats", files[0], "--cameras", files[1], "--splat-count", str(manifest["splatCount"]),
            "--sh-degree", str(manifest["shDegree"]), "--camera", str(index), "--width", str(cam["width"]),
            "--height", str(cam["height"])]


def check_split(scene: str, manifest: dict) -> None:
    """Refuses a scene whose camera buffer doesn't hold the manifest's train/test split."""
    version = manifest.get("version")
    if version != TRAIN_FORMAT_VERSION:
        raise SystemExit(f"scene_args: {scene} has format version {version}; splat_train reads version "
                         f"{TRAIN_FORMAT_VERSION} (the split in the camera buffer): re-convert it")
    with open(os.path.join(scene, manifest["files"]["cameras"]), "rb") as f:
        data = f.read()
    if len(data) != CAMERA_BYTES * manifest["cameraCount"]:
        raise SystemExit(f"scene_args: {scene}'s camera buffer doesn't hold its {manifest['cameraCount']} cameras")
    held = [i for i in range(manifest["cameraCount"])
            if struct.unpack_from("<I", data, i * CAMERA_BYTES + HELD_OUT_OFFSET)[0]]
    if held != sorted(manifest["test"]):
        raise SystemExit(f"scene_args: {scene}'s camera buffer holds out cameras {held}, "
                         f"but its manifest's test cameras are {sorted(manifest['test'])}")


def train_args(scene: str) -> List[str]:
    manifest = read_manifest(scene)
    check_split(scene, manifest)
    images = manifest["files"].get("images", [])
    if len(images) != 1:
        raise SystemExit(f"scene_args: {scene} has {len(images)} image batches; splat_train reads exactly one")
    targets = checked_path(os.path.join(scene, images[0]))
    pixels, rest = divmod(os.path.getsize(targets), 4)
    if rest:
        raise SystemExit(f"scene_args: {targets} isn't a whole number of RGBA8 pixels")
    imaged = [c for c in manifest["cameras"] if "imageBatch" in c]
    if not imaged:
        raise SystemExit(f"scene_args: {scene}'s cameras have no images")
    files = scene_files(scene, manifest)
    return ["--splats", files[0], "--cameras", files[1], "--targets", targets,
            "--splat-count", str(manifest["splatCount"]), "--camera-count", str(manifest["cameraCount"]),
            "--sh-degree", str(manifest["shDegree"]),
            "--max-width", str(max(c["width"] for c in imaged)), "--max-height", str(max(c["height"] for c in imaged)),
            "--target-pixels", str(pixels), "--extent", repr(float(manifest["extent"]))]


# The stream header's splatCount: its third uint (lib/splat_stream.zgal's StreamHeader)
STREAM_SPLAT_COUNT_OFFSET = 8


def view_args(stream: str) -> List[str]:
    with open(checked_path(stream), "rb") as f:
        head = f.read(STREAM_SPLAT_COUNT_OFFSET + 4)
    if len(head) < STREAM_SPLAT_COUNT_OFFSET + 4:
        raise SystemExit(f"scene_args: {stream} is too short for a stream header")
    count = struct.unpack_from("<I", head, STREAM_SPLAT_COUNT_OFFSET)[0]
    return [checked_path(stream), "--splat-count", str(count), "--stream-bytes", str(os.path.getsize(stream))]


def query(url_dir: str, args: List[str]) -> str:
    """A page's query string for a program's arguments (every option takes a value): options as
    name=value, positionals as arg=value, paths as url_dir/<file name>."""
    def value(word: str) -> str:
        if isinstance(word, PathArg):
            word = url_dir.rstrip("/") + "/" + os.path.basename(word)
        return urllib.parse.quote(word, safe="/")
    pairs = []
    i = 0
    while i < len(args):
        if args[i].startswith("--"):
            pairs.append(f"{args[i][2:]}={value(args[i + 1])}")
            i += 2
        else:
            pairs.append(f"arg={value(args[i])}")
            i += 1
    return "&".join(pairs)


def program_args(argv: List[str]) -> List[str]:
    if len(argv) == 2 and argv[0] == "--train":
        return train_args(argv[1])
    if len(argv) == 2 and argv[0] == "--view":
        return view_args(argv[1])
    if len(argv) == 2:
        return scene_args(argv[0], argv[1])
    raise SystemExit(__doc__)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) >= 2 and argv[0] == "--query":
        print(query(argv[1], program_args(argv[2:])))
    else:
        print(" ".join(program_args(argv)))


if __name__ == "__main__":
    main()
