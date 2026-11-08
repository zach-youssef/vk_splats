#!/usr/bin/env python3
"""Converts COLMAP text exports and INRIA PLY files to the programs' inputs (docs/data-format.md).

  convert.py colmap <model dir> <out dir> [--images <dir>] [--max-size <px>] [--splats <ply>]
      A scene: the model's images as undistorted pinhole cameras (every 8th in name order held
      out), their images (undistorted, resized, packed), and splats initialized from points3D, or
      read from a PLY.
  convert.py ply-to-bin <in.ply> <out splats.bin>
  convert.py bin-to-ply <in splats.bin> <out.ply>
  convert.py stream <in.ply|splats.bin> <out dir> --config <file> [--cameras <cameras.bin>]
      A stream (<out dir>/scene.stream) of the splats, encoded as the JSON config says (the
      presets are tools/stream-configs/), and a copy of cameras.bin with --cameras; prints each
      group's largest error.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from splatdata import colmap, native, scene, splats, stream  # noqa: E402


def convert_colmap(args) -> None:
    model = colmap.read_model(args.model)
    images = sorted(model.images.values(), key=lambda im: im.name)
    if not images:
        sys.exit(f"{args.model}: no images")
    cameras = []
    for image in images:
        pixels = None
        if args.images:
            path = os.path.join(args.images, image.name)
            if not os.path.exists(path):
                sys.exit(f"{path}: missing (named in {args.model}/images.txt)")
            pixels = scene.load_rgb(path)
        cameras.append(scene.pinhole_from_colmap(model, image, pixels, args.max_size))

    if args.splats:
        initial = splats.read_ply(args.splats)
        origin = {"splats": os.path.basename(args.splats)}
    else:
        initial = splats.from_points(model.points.xyz, model.points.rgb)
        origin = {"splats": "points3D"}
    manifest = scene.write_scene(
        args.out, initial, cameras, held_out=scene.test_split([c.name for c in cameras]),
        source={"colmap": os.path.basename(os.path.normpath(args.model)), **origin,
                "maxSize": args.max_size})
    print(f"{args.out}: {manifest['splatCount']} splats, {len(manifest['train'])} train and "
          f"{len(manifest['test'])} test cameras, {len(manifest['files']['images'])} image buffer(s)")


def ply_to_bin(args) -> None:
    with open(args.bin, "wb") as f:
        f.write(splats.to_std430(splats.read_ply(args.ply)))


def bin_to_ply(args) -> None:
    with open(args.bin, "rb") as f:
        splats.write_ply(args.ply, splats.from_std430(f.read()))


def convert_stream(args) -> None:
    try:
        config = stream.load_config(args.config)
        extension = os.path.splitext(args.input)[1].lower()
        if extension == ".ply":
            data, degree = native.read_ply(args.input)
        elif extension == ".bin":
            data, degree = native.read_splats_bin(args.input)
        else:
            sys.exit(f"{args.input}: the input must be an INRIA PLY (.ply) or a splats.bin (.bin)")
        if args.cameras:
            with open(args.cameras, "rb") as f:
                scene.cameras_from_std430(f.read())  # refuses a file that isn't Cameras
        out, report = native.encode_stream(data, stream.encode_config(config, degree, args.config))
    except native.EncodeError as e:
        # An explicit range too wide for a float step is the config's; anything else the input's
        explicit = (e.status == native.ENCODE_RANGE_OVERFLOW and e.group < native.GROUP_COUNT
                    and stream.range_modes(config)[e.group] == "explicit")
        sys.exit(f"{args.config if explicit else args.input}: {e}")
    except (OSError, ValueError) as e:
        sys.exit(str(e))
    path = os.path.join(args.out, "scene.stream")
    try:
        write_outputs(args.out, {"scene.stream": out}, {"cameras.bin": args.cameras} if args.cameras else {})
    except OSError as e:
        sys.exit(f"{args.out}: writing the outputs failed: {e}")
    print(f"{path}: {stream.format_report(out, report, stream.range_modes(config))}")


def write_outputs(out_dir: str, contents: dict, copies: dict) -> None:
    """Writes each file of `contents` (name: bytes) and copies each of `copies` (name: source)
    into out_dir, each to a temporary file first, renamed once all are written, so a failure
    leaves no partial file and no stream without its cameras. A copy onto itself is skipped."""
    os.makedirs(out_dir, exist_ok=True)
    temporary = {}
    try:
        for name, data in contents.items():
            fd, temporary[name] = tempfile.mkstemp(prefix=f".{name}.", dir=out_dir)
            with os.fdopen(fd, "wb") as f:
                f.write(data)
        for name, source in copies.items():
            target = os.path.join(out_dir, name)
            if os.path.exists(target) and os.path.samefile(source, target):
                continue
            fd, temporary[name] = tempfile.mkstemp(prefix=f".{name}.", dir=out_dir)
            os.close(fd)
            shutil.copyfile(source, temporary[name])
        for name, path in temporary.items():
            os.replace(path, os.path.join(out_dir, name))
            temporary[name] = None
    finally:
        for path in temporary.values():
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    c = commands.add_parser("colmap", help="a scene from a COLMAP text export")
    c.add_argument("model", help="directory with cameras.txt, images.txt and points3D.txt")
    c.add_argument("out", help="output directory")
    c.add_argument("--images", help="directory of the model's images (named as in images.txt); "
                                    "without it, only cameras and splats are written")
    c.add_argument("--max-size", type=int, help="scale images down so the longer side is at most this")
    c.add_argument("--splats", help="an INRIA PLY to use instead of initializing from points3D")
    c.set_defaults(run=convert_colmap)

    p = commands.add_parser("ply-to-bin", help="an INRIA PLY to a splats.bin")
    p.add_argument("ply")
    p.add_argument("bin")
    p.set_defaults(run=ply_to_bin)

    b = commands.add_parser("bin-to-ply", help="a splats.bin to an INRIA PLY")
    b.add_argument("bin")
    b.add_argument("ply")
    b.set_defaults(run=bin_to_ply)

    s = commands.add_parser("stream", help="a stream file of an INRIA PLY or a splats.bin")
    s.add_argument("input", help="an INRIA PLY (.ply) or a splats.bin (.bin)")
    s.add_argument("out", help="output directory: scene.stream, and cameras.bin with --cameras")
    s.add_argument("--config", required=True,
                   help="the JSON encoding config (docs/data-format.md; presets in tools/stream-configs/)")
    s.add_argument("--cameras", help="a cameras.bin to copy beside the stream")
    s.set_defaults(run=convert_stream)

    args = parser.parse_args(argv)
    if getattr(args, "max_size", None) is not None and args.max_size < 1:
        parser.error("--max-size must be positive")
    args.run(args)


if __name__ == "__main__":
    main()
