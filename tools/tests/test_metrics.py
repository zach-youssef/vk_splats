"""splatdata.metrics: the PSNR and the reference's SSIM, on cases with known values."""

import math

import numpy as np
import pytest

from splatdata import metrics


def test_window_is_the_normalized_gaussian():
    g = metrics.window()
    assert g.shape == (11,) and g.sum() == pytest.approx(1)
    assert g[5] / g[6] == pytest.approx(math.exp(1 / (2 * 1.5 ** 2)))
    assert np.allclose(g, g[::-1])


def test_ssim_of_an_image_with_itself_is_1():
    x = np.random.default_rng(0).uniform(0, 1, (20, 17, 3))
    assert metrics.ssim(x, x) == pytest.approx(1, abs=1e-12)


def test_ssim_of_constants_is_the_luminance_term_inside():
    a, b = 0.25, 0.75
    x = np.full((30, 30, 1), a)
    y = np.full((30, 30, 1), b)
    s = metrics.ssim_map(x, y)
    # Inside, the variances and covariance are 0, so only the luminance term is left
    assert s[5:-5, 5:-5] == pytest.approx((2 * a * b + metrics.C1) / (a * a + b * b + metrics.C1))
    # The zero padding darkens the border's means, which moves its SSIM
    assert not np.allclose(s[0, 0], s[15, 15])


def test_ssim_is_symmetric_and_falls_with_noise():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, (24, 40, 3))
    slight = np.clip(x + 0.02 * rng.standard_normal(x.shape), 0, 1)
    heavy = np.clip(x + 0.3 * rng.standard_normal(x.shape), 0, 1)
    assert metrics.ssim(x, slight) == pytest.approx(metrics.ssim(slight, x))
    assert 1 > metrics.ssim(x, slight) > metrics.ssim(x, heavy) > 0


def test_ssim_matches_scipys_gaussian_filter():
    # An independent blur: scipy's separable Gaussian, sigma 1.5, cut off at radius 5 (11 taps),
    # zero-padded (mode="constant"); the SSIM formula written out from the paper
    ndimage = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, (23, 31, 3))
    y = np.clip(0.7 * x + 0.3 * rng.uniform(0, 1, x.shape), 0, 1)

    def blur(image):
        return np.stack([ndimage.gaussian_filter(image[..., c], 1.5, mode="constant", truncate=5 / 1.5)
                         for c in range(3)], axis=-1)

    mx, my = blur(x), blur(y)
    covariance = blur(x * y) - mx * my
    expected = ((2 * mx * my + 0.01 ** 2) * (2 * covariance + 0.03 ** 2)
                / ((mx ** 2 + my ** 2 + 0.01 ** 2) * (blur(x * x) - mx ** 2 + blur(y * y) - my ** 2 + 0.03 ** 2)))
    assert metrics.ssim(x, y) == pytest.approx(expected.mean(), abs=1e-12)
    assert metrics.ssim_map(x, y) == pytest.approx(expected, abs=1e-12)


def test_ssim_refuses_mismatched_shapes():
    with pytest.raises(ValueError):
        metrics.ssim(np.zeros((4, 4, 3)), np.zeros((4, 5, 3)))


def test_psnr():
    x = np.zeros((4, 4, 3))
    assert metrics.psnr(x, x) == math.inf
    assert metrics.psnr(x, x + 0.1) == pytest.approx(20)
    with pytest.raises(ValueError):
        metrics.psnr(x, np.zeros((4, 4, 1)))
