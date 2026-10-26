#!/usr/bin/env python3
"""Converts COLMAP text exports and INRIA PLY files to the programs' inputs (docs/data-format.md).

  convert.py colmap <model dir> <out dir> [--images <dir>] [--max-size <px>] [--splats <ply>]
      A scene: the model's images as undistorted pinhole cameras (every 8th in name order held
      out), their images (undistorted, resized, packed), and splats initialized from points3D, or
      read from a PLY.
  convert.py ply-to-bin <in.ply> <out splats.bin>
  convert.py bin-to-ply <in splats.bin> <out.ply>
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from splatdata import colmap, scene, splats  # noqa: E402


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

    args = parser.parse_args(argv)
    if getattr(args, "max_size", None) is not None and args.max_size < 1:
        parser.error("--max-size must be positive")
    args.run(args)


if __name__ == "__main__":
    main()
