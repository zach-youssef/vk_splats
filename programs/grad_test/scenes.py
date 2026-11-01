#!/usr/bin/env python3
"""grad_test's targets and arguments for a camera of a scene (tools/make_synthetic.py's).

  scenes.py <scene dir> <camera> <targets file>
      Writes the scene's targets to <targets file> (RGBA8, the red in the low byte, row by row,
      every camera's image in the manifest's order, as an images_<k>.bin holds them) and prints
      tools/scene_args.py's arguments, --targets, the camera's --target-offset, --target-pixels (all
      the cameras' pixels) and the background:
      - an S1 variant with an oracle target (the gradient variant): the constant target
        (round(255 t) in every pixel), the oracle's background, --oracle and the unrounded target
        (grad_check.h rounds it as here);
      - any other: a smooth pattern unlike the render (a smooth loss for the finite differences),
        on a background that isn't grey (0.3, 0.5, 0.7).

Needs numpy.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from scene_args import scene_args  # noqa: E402

PATTERN_BACKGROUND = (0.3, 0.5, 0.7)


def pattern(width, height, k):
    """A smooth RGB pattern in [0.15, 0.85], different for each camera k."""
    x, y = np.meshgrid((np.arange(width) + 0.5) / width, (np.arange(height) + 0.5) / height)
    return np.stack([0.5 + 0.35 * np.sin(2.0 * np.pi * (1.5 * x + 0.25 * k)),
                     0.5 + 0.35 * np.cos(2.0 * np.pi * y),
                     0.15 + 0.7 * (x + y) / 2.0], axis=-1)


def rgba8(rgb):
    q = np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint32)
    return (q[..., 0] | (q[..., 1] << 8) | (q[..., 2] << 16) | (255 << 24)).astype("<u4").reshape(-1)


def args(scene, camera, targets):
    with open(os.path.join(scene, "manifest.json")) as f:
        manifest = json.load(f)
    source = manifest.get("source") or {}
    oracle = source.get("oracle") if source.get("scene") == "s1" else None
    if oracle is not None and "target" not in oracle:
        oracle = None
    images, offsets, offset = [], [], 0
    for k, cam in enumerate(manifest["cameras"]):
        w, h = cam["width"], cam["height"]
        if oracle is not None:
            rgb = np.broadcast_to(np.asarray(oracle["target"], dtype=np.float64), (h, w, 3))
        else:
            rgb = pattern(w, h, k)
        images.append(rgba8(rgb))
        offsets.append(offset)
        offset += w * h
    np.concatenate(images).tofile(targets)
    out = scene_args(scene, str(camera)) + ["--targets", targets, "--target-offset", str(offsets[camera]),
                                         "--target-pixels", str(offset)]
    if oracle is not None:
        r, g, b = oracle["background"]
        tr, tg, tb = oracle["target"]
        out += ["--oracle", "--target-r", repr(tr), "--target-g", repr(tg), "--target-b", repr(tb)]
    else:
        r, g, b = PATTERN_BACKGROUND
    return out + ["--background-r", repr(r), "--background-g", repr(g), "--background-b", repr(b)]


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    print(" ".join(args(sys.argv[1], int(sys.argv[2]), sys.argv[3])))
