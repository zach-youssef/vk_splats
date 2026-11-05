"""Image metrics, as the reference code evaluates a trained scene: the PSNR and the SSIM (Wang et al.
2004, as `utils/loss_utils.ssim`: per channel, an 11x11 Gaussian window of sigma 1.5, zero-padded,
C1 = 0.01^2, C2 = 0.03^2, the mean over the pixels and channels). Images are H x W x C floats in
[0, 1] (8-bit ones divided by 255), compared in float64. Only numpy."""

import math

import numpy as np

C1 = 0.01 ** 2
C2 = 0.03 ** 2
WINDOW = 11
SIGMA = 1.5


def window() -> np.ndarray:
    """The separable window's 1D weights, summing to 1"""
    d = np.arange(WINDOW, dtype=np.float64) - WINDOW // 2
    g = np.exp(-d * d / (2 * SIGMA ** 2))
    return g / g.sum()


def blur(image: np.ndarray) -> np.ndarray:
    """The window's blur of each channel of an H x W x C image, zero-padded"""
    g = window()
    h, w = image.shape[:2]
    r = WINDOW // 2
    padded = np.pad(image, ((r, r), (r, r), (0, 0)))
    rows = sum(g[k] * padded[:, k:k + w] for k in range(WINDOW))
    return sum(g[k] * rows[k:k + h] for k in range(WINDOW))


def ssim_map(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Each pixel's and channel's SSIM"""
    if x.shape != y.shape or x.ndim != 3:
        raise ValueError(f"the images' shapes {x.shape} and {y.shape} aren't the same H x W x C")
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    mx, my = blur(x), blur(y)
    sx = blur(x * x) - mx * mx
    sy = blur(y * y) - my * my
    sxy = blur(x * y) - mx * my
    return (2 * mx * my + C1) * (2 * sxy + C2) / ((mx * mx + my * my + C1) * (sx + sy + C2))


def ssim(x: np.ndarray, y: np.ndarray) -> float:
    return float(ssim_map(x, y).mean())


def psnr(x: np.ndarray, y: np.ndarray) -> float:
    """10 log10(1 / MSE); inf for equal images"""
    if x.shape != y.shape:
        raise ValueError(f"the images' shapes {x.shape} and {y.shape} differ")
    mse = float(np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2))
    return math.inf if mse == 0 else 10 * math.log10(1 / mse)
