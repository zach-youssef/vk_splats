#!/usr/bin/env python3
"""check.py <out dir> <splat_render> --min-psnr <dB> --min-ratio <r> --min-skipped <n>: checks test.sh's
splat_train run, from its log (<out dir>/train.log) and its PLY (<out dir>/trained.ply).

- the held-out PSNR splat_train reports is at least --min-psnr, and the first epoch's mean L1
  is at least --min-ratio times the last's;
- at least --min-skipped iterations were skipped for overflowing (test.sh starts the capacity
  small to cover the skip);
- the PLY holds the scene's splat count of finite splats;
- the PLY, written into a copy of the start scene and rendered by splat_render from each
  held-out camera, scores within 0.5 dB of the PSNR splat_train reported for that view against
  the ground truth's PNG (the capture is 8-bit, splat_train's render float, so they differ a
  little). This checks the readback, the PSNR and the export independently of the program.

Needs numpy and Pillow; run from the repository's root."""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, "tools")
import scene_args  # noqa: E402
from make_synthetic import renderer_command  # noqa: E402
from splatdata import scene, splats  # noqa: E402

TOLERANCE_DB = 0.5


def number(pattern, log):
    match = re.search(pattern, log)
    if not match:
        sys.exit(f"check: train.log has no line matching {pattern!r}")
    return match.groups()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out")
    parser.add_argument("renderer")
    parser.add_argument("--min-psnr", type=float, required=True)
    parser.add_argument("--min-ratio", type=float, required=True)
    parser.add_argument("--min-skipped", type=int, required=True)
    args = parser.parse_args()
    with open(os.path.join(args.out, "train.log")) as f:
        log = f.read()
    errors = []

    psnr, views = number(r"held-out PSNR (\S+) dB over (\d+) views", log)
    first, last, ratio = number(r"first epoch (\S+), last epoch (\S+), ratio (\S+)", log)
    print(f"check: held-out PSNR {float(psnr):.2f} dB over {views} views; L1 {float(first):.5f} -> "
          f"{float(last):.5f} (ratio {float(ratio):.2f})")
    if not float(psnr) >= args.min_psnr:
        errors.append(f"the held-out PSNR {psnr} dB is below {args.min_psnr}")
    if not float(ratio) >= args.min_ratio:
        errors.append(f"the loss fell {ratio} times, less than {args.min_ratio}")
    skipped, = number(r"(\d+) iterations skipped", log)
    print(f"check: {skipped} iterations skipped")
    if int(skipped) < args.min_skipped:
        errors.append(f"{skipped} iterations were skipped, fewer than {args.min_skipped}")

    start = os.path.join(args.out, "start")
    with open(os.path.join(start, "manifest.json")) as f:
        manifest = json.load(f)
    trained = splats.read_ply(os.path.join(args.out, "trained.ply"))
    if len(trained) != manifest["splatCount"]:
        errors.append(f"the PLY holds {len(trained)} splats, not {manifest['splatCount']}")
    fields = [trained.position, trained.log_scale, trained.rotation, trained.opacity_logit, trained.sh]
    if not all(np.all(np.isfinite(f)) for f in fields):
        errors.append("the PLY has a splat that isn't finite")

    reported = {int(i): float(p) for i, p in re.findall(r"test view (\d+) PSNR (\S+) dB", log)}
    if sorted(reported) != sorted(manifest["test"]):
        errors.append(f"the views scored are {sorted(reported)}, not the held-out {manifest['test']}")
    rerender = os.path.join(args.out, "trained")
    shutil.copytree(start, rerender)
    with open(os.path.join(rerender, manifest["files"]["splats"]), "wb") as f:
        f.write(splats.to_std430(trained))
    for i in manifest["test"]:
        png = os.path.join(args.out, f"trained-camera-{i}.png")
        command = renderer_command(args.renderer) + ["--validation", "--frames", "8", "--capture", png]
        command += scene_args.scene_args(rerender, str(i))
        run = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if run.returncode != 0:
            errors.append(f"rendering camera {i} failed ({run.returncode}):\n{run.stdout}")
            continue
        gt = scene.load_rgb(os.path.join(args.out, "gt", manifest["cameras"][i]["png"])).astype(np.float64)
        got = scene.load_rgb(png).astype(np.float64)
        mse = float(np.mean(((got - gt) / 255) ** 2))
        independent = math.inf if mse == 0 else 10 * math.log10(1 / mse)
        print(f"check: camera {i}: splat_train {reported.get(i, math.nan):.2f} dB, re-rendered {independent:.2f} dB")
        if not abs(independent - reported.get(i, math.nan)) <= TOLERANCE_DB:
            errors.append(f"camera {i}: re-rendered at {independent:.2f} dB, splat_train reported "
                          f"{reported.get(i)}")
    for e in errors:
        print(f"check: {e}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
