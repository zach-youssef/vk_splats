#!/usr/bin/env python3
"""psnr.py <capture.png> <golden.png> <minimum dB>: prints the PSNR of a capture's RGB against a
golden's (8-bit, peak 255) and exits 1 when they differ in size or the PSNR is below the minimum.
Identical images print inf. Needs numpy and Pillow (tools/requirements.txt)."""

import math
import sys

import numpy as np
from PIL import Image


def rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float64)


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    capture, golden, minimum = rgb(sys.argv[1]), rgb(sys.argv[2]), float(sys.argv[3])
    if capture.shape != golden.shape:
        print(f"psnr: {sys.argv[1]} is {capture.shape[1]}x{capture.shape[0]}, the golden "
              f"{golden.shape[1]}x{golden.shape[0]}")
        sys.exit(1)
    mse = float(np.mean((capture - golden) ** 2))
    psnr = math.inf if mse == 0.0 else 10.0 * math.log10(255.0 ** 2 / mse)
    worst = int(np.max(np.abs(capture - golden)))
    print(f"psnr: {sys.argv[1]}: {psnr:.2f} dB against {sys.argv[2]} (largest difference {worst}/255)")
    if psnr < minimum:
        print(f"psnr: FAILED, below {minimum} dB")
        sys.exit(1)


if __name__ == "__main__":
    main()
