#!/usr/bin/env python3
"""make_pair.py <case> <out dir> [--width W] [--height H] [--seed S]: writes an image pair for ssim_test (<out dir>/render.bin:
vec4 floats a pixel, the render; <out dir>/targets.bin: RGBA8 pixels, the target) and prints ssim_test's
arguments for it: the images' size and --expected-ssim, their SSIM in float64 (the reference's: per
channel, the 11x11 Gaussian window, sigma 1.5, zero-padded, C1 = 0.01^2, C2 = 0.03^2, the mean over the
pixels and channels: splatdata.metrics). The cases:
- random: a random render in [-0.05, 1.05] (the rasterizer's render is unclamped) and a target
  correlated with it (0.7 of it plus noise, quantized), so the SSIM is neither 0 nor 1;
- identity: a random target and the render equal to it (its bytes / 255 as floats): SSIM 1;
- constant: a constant render (0.25, 0.5, 0.75) and target (153, 26, 77) / 255, with --interior-ssim the
  closed-form luminance term's mean over the channels, (2 a b + C1) / (a^2 + b^2 + C1), which every pixel
  whose window lies inside the image has (its variances and covariance are 0).
numpy's default_rng, so the same on every platform. Needs numpy."""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from splatdata.metrics import C1, ssim  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=["random", "identity", "constant"])
    parser.add_argument("out")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    w, h = args.width, args.height
    rng = np.random.default_rng(args.seed)
    extra = []
    if args.case == "random":
        render = rng.uniform(-0.05, 1.05, (h, w, 3)).astype(np.float32)
        noisy = 0.7 * render + 0.15 + 0.15 * rng.standard_normal((h, w, 3))
        target = np.clip(np.rint(noisy * 255), 0, 255).astype(np.uint8)
    elif args.case == "identity":
        target = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        render = (target.astype(np.float32) / np.float32(255))
    else:
        render = np.broadcast_to(np.array([0.25, 0.5, 0.75], np.float32), (h, w, 3)).copy()
        target = np.broadcast_to(np.array([153, 26, 77], np.uint8), (h, w, 3)).copy()
        a = render[0, 0].astype(np.float64)
        b = target[0, 0] / 255.0
        interior = ((2 * a * b + C1) / (a * a + b * b + C1)).mean()
        extra = ["--interior-ssim", repr(float(interior))]
    os.makedirs(args.out, exist_ok=True)
    rgba = np.concatenate([render, np.zeros((h, w, 1), np.float32)], axis=2)
    rgba.astype("<f4").tofile(os.path.join(args.out, "render.bin"))
    packed = np.concatenate([target, np.full((h, w, 1), 255, np.uint8)], axis=2)
    packed.tofile(os.path.join(args.out, "targets.bin"))
    expected = ssim(render.astype(np.float64), target / 255.0)
    print(" ".join([f"--render {args.out}/render.bin", f"--targets {args.out}/targets.bin", f"--width {w}",
                    f"--height {h}", f"--expected-ssim {float(expected)!r}"] + extra))


if __name__ == "__main__":
    main()
