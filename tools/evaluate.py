#!/usr/bin/env python3
"""Scores a trained PLY on a scene's held-out views, as the reference's metrics.py does:

    tools/evaluate.py <scene dir> <trained.ply> <splat_render> <out dir> [--validation]
                      [--frames N] [--background G] [--side-by-side CAMERA ...] [--thumbnail PX]

It writes the PLY's splats into <out dir>/scene (the scene's manifest and cameras, its splats
replaced; no images copied), renders each held-out camera with splat_render (the binary, or the
web build's .js under $NODE) into <out dir>/test-<k>.png, and compares each capture with the
scene's undistorted image (its manifest's png): the PSNR and the SSIM (splatdata.metrics, the
reference's definitions), per view and their mean, printed and written to <out dir>/metrics.json.
The captures are 8-bit, where splat_train scores its float render, so the PSNRs differ a little
(within about 0.5 dB: programs/splat_train/check.py).

--background must be the training run's (0 by default). A render that overflows splat_render's tile
entries (it grows them after the frame) is rendered again from the grown capacity, and fails if
that overflows too.

--side-by-side writes <out dir>/side-by-side-<k>.png for those cameras (held out or not): the
image on the left, the render on the right, each scaled so its longer side is --thumbnail px.

Needs numpy and Pillow; run from anywhere."""

import argparse
import json
import os
import re
import subprocess
import sys
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_args  # noqa: E402
from make_synthetic import renderer_command  # noqa: E402
from splatdata import metrics, scene, splats  # noqa: E402


def trained_scene(scene_dir: str, ply: str, out: str) -> dict:
    """Writes <out> as the scene with the PLY's splats; returns its manifest."""
    with open(os.path.join(scene_dir, "manifest.json")) as f:
        manifest = json.load(f)
    trained = splats.read_ply(ply)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, manifest["files"]["splats"]), "wb") as f:
        f.write(splats.to_std430(trained))
    with open(os.path.join(scene_dir, manifest["files"]["cameras"]), "rb") as f:
        cameras = f.read()
    with open(os.path.join(out, manifest["files"]["cameras"]), "wb") as f:
        f.write(cameras)
    manifest = {**manifest, "splatCount": len(trained), "files": {**manifest["files"], "images": []}}
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    return manifest


MAX_ENTRIES = 1 << 25  # splat_render's entryCapacityMax; it stays silent once at this, overflowed or not
GROWTH = re.compile(r"a frame needed \d+ tile entries, more than its capacity of \d+; growing it to (\d+)")


def render(renderer: str, scene_dir: str, camera: int, png: str, frames: int, background: float,
           validation: bool) -> np.ndarray:
    """Captures the camera. splat_render's tile entries start small and grow after a frame that
    overflowed them, which misses splats; if any frame did, renders again from the grown
    capacity, and fails if that one overflows too or the growth reaches splat_render's maximum (where it
    can no longer say whether a frame overflowed), rather than score a capture with missing splats."""
    command = renderer_command(renderer) + ["--frames", str(frames), "--capture", png,
                                            "--background", repr(float(background))]
    command += (["--validation"] if validation else []) + scene_args.scene_args(scene_dir, str(camera))
    for attempt in range(2):
        run = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if run.returncode != 0:
            raise RuntimeError(f"{' '.join(command)} failed ({run.returncode}):\n{run.stdout}")
        grown = GROWTH.findall(run.stdout)
        if not grown:
            return scene.load_rgb(png)
        if int(grown[-1]) >= MAX_ENTRIES:
            raise RuntimeError(f"camera {camera}: splat_render grew its tile entries to {grown[-1]}, its maximum, "
                               f"so the capture may miss splats:\n{run.stdout}")
        if attempt == 0:
            command += ["--initial-entries", grown[-1]]
    raise RuntimeError(f"camera {camera}: splat_render overflowed its tile entries even starting at "
                       f"{grown[-1]}, so the capture misses splats:\n{run.stdout}")


def thumbnail(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = size / max(w, h)
    return scene.resize_image(image, max(1, round(w * scale)), max(1, round(h * scale)))


def main(argv: Optional[List[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("scene")
    parser.add_argument("ply")
    parser.add_argument("renderer")
    parser.add_argument("out")
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--frames", type=int, default=8,
                        help="frames per render (the tile entries' growth needs a few)")
    parser.add_argument("--background", type=float, default=0.0)
    parser.add_argument("--side-by-side", type=int, nargs="*", default=[])
    parser.add_argument("--thumbnail", type=int, default=256)
    args = parser.parse_args(argv)

    with open(os.path.join(args.scene, "manifest.json")) as f:
        source = json.load(f)
    rendered_scene = os.path.join(args.out, "scene")
    manifest = trained_scene(args.scene, args.ply, rendered_scene)
    cameras = source["cameras"]
    for k in args.side_by_side:
        if not 0 <= k < len(cameras) or "png" not in cameras[k]:
            raise SystemExit(f"evaluate: camera {k} isn't one of the scene's cameras with an image")

    def view(k: int, png: str):
        got = render(args.renderer, rendered_scene, k, png, args.frames, args.background, args.validation)
        gt = scene.load_rgb(os.path.join(args.scene, cameras[k]["png"]))
        if got.shape != gt.shape:
            raise RuntimeError(f"camera {k}: the render is {got.shape[1]}x{got.shape[0]}, the image "
                               f"{gt.shape[1]}x{gt.shape[0]}")
        return gt, got

    results = []
    views = {}  # camera -> (image, render), for --side-by-side
    for k in source["test"]:
        gt, got = views[k] = view(k, os.path.join(args.out, f"test-{k}.png"))
        x, y = got / 255.0, gt / 255.0
        results.append({"camera": k, "name": cameras[k]["name"], "psnr": metrics.psnr(x, y),
                        "ssim": metrics.ssim(x, y)})
        print(f"evaluate: test view {k} PSNR {results[-1]['psnr']:.3f} dB, SSIM {results[-1]['ssim']:.4f}")
    summary = {"splats": manifest["splatCount"], "views": results,
               "psnr": float(np.mean([r["psnr"] for r in results])) if results else None,
               "ssim": float(np.mean([r["ssim"] for r in results])) if results else None}
    if results:
        print(f"evaluate: {manifest['splatCount']} splats; held-out PSNR {summary['psnr']:.3f} dB, "
              f"SSIM {summary['ssim']:.4f} over {len(results)} views")
    for k in args.side_by_side:
        gt, got = views[k] if k in views else view(k, os.path.join(args.out, f"camera-{k}.png"))
        pair = np.concatenate([thumbnail(gt, args.thumbnail), thumbnail(got, args.thumbnail)], axis=1)
        scene.save_png(os.path.join(args.out, f"side-by-side-{k}.png"), pair)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=1)
        f.write("\n")
    return summary


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        sys.exit(f"evaluate: {e}")
