"""Undistortion and resizing: the pinhole cameras' geometry, and the resampled images."""

import os

import numpy as np
import pytest
from conftest import REPO

from splatdata import colmap, scene

MODEL = os.path.join(REPO, "colmap", "colmap_exports3")

CAMERAS = [
    colmap.Camera(1, "SIMPLE_RADIAL", 160, 120, np.array([150.0, 80, 60, 0.08])),   # pincushion
    colmap.Camera(1, "SIMPLE_RADIAL", 160, 120, np.array([150.0, 81, 59, -0.12])),  # barrel
    colmap.Camera(1, "OPENCV", 160, 120, np.array([150.0, 140, 78, 62, -0.1, 0.02, 0.002, -0.001])),
]


@pytest.mark.parametrize("camera", CAMERAS)
def test_undistorted_view_stays_inside_the_source(camera):
    s = scene.undistorted_focal_scale(camera)
    w, h = camera.width, camera.height
    ys, xs = np.mgrid[0:h, 0:w] + 0.5
    src = scene._source_pixels(camera, np.stack([xs, ys], -1), s)
    assert src[..., 0].min() >= 0.5 - 1e-6 and src[..., 0].max() <= w - 0.5 + 1e-6
    assert src[..., 1].min() >= 0.5 - 1e-6 and src[..., 1].max() <= h - 0.5 + 1e-6
    # ... and is the widest such view: 1% wider leaves the source
    wider = scene._source_pixels(camera, np.stack([xs, ys], -1), s * 0.99)
    assert wider.min() < 0.5 - 1e-6 or (wider - [w, h]).max() > -0.5 + 1e-6


@pytest.mark.parametrize("camera", CAMERAS)
@pytest.mark.parametrize("max_size", [None, 64])
def test_undistorted_image_matches_the_pinhole_camera(camera, max_size):
    # A smooth pattern over normalized (undistorted) coordinates, rendered through the distortion:
    # after undistortion each pixel must show the pattern at its pinhole ray
    pattern = lambda n: 127.5 + 100 * np.sin(6 * n[..., 0]) * np.cos(5 * n[..., 1])  # noqa: E731
    h, w = camera.height, camera.width
    ys, xs = np.mgrid[0:h, 0:w] + 0.5
    source = pattern(camera.to_normalized(np.stack([xs, ys], -1)))
    source = np.repeat(np.clip(np.rint(source), 0, 255).astype(np.uint8)[..., None], 3, axis=2)

    image = colmap.Image(1, np.array([1.0, 0, 0, 0]), np.zeros(3), 1, "x.png")
    model = colmap.Model({1: camera}, {1: image}, None)
    pinhole = scene.pinhole_from_colmap(model, image, source, max_size)
    assert pinhole.image.shape == (pinhole.height, pinhole.width, 3)
    ys, xs = np.mgrid[0:pinhole.height, 0:pinhole.width] + 0.5
    expected = pattern(np.stack([(xs - pinhole.cx) / pinhole.fx, (ys - pinhole.cy) / pinhole.fy], -1))
    error = np.abs(pinhole.image[..., 0].astype(float) - expected)
    # Bilinear resampling (and Lanczos resizing) of a smooth pattern, plus rounding to 8 bits
    assert np.mean(error) < 1.0
    assert np.max(error) < (4 if max_size is None else 10)


def test_resized_cs5330_cameras_project_like_colmap():
    model = colmap.read_model(MODEL)
    points = model.points
    for image in list(model.images.values())[:5]:
        camera = model.cameras[image.camera_id]
        for max_size in (None, 187):
            pinhole = scene.pinhole_from_colmap(model, image, max_size=max_size)
            assert (pinhole.width, pinhole.height) == ((748, 748) if max_size is None else (187, 187))
            xyz = points.xyz[points.track_point[points.track_image_id == image.id]]
            p = pinhole.project(xyz)
            # Back from the pinhole image to the distorted one: the same pixels as COLMAP's model
            scale = scene.undistorted_focal_scale(camera)
            sx = camera.width / pinhole.width
            original = scene._source_pixels(camera, p * [sx, camera.height / pinhole.height], scale)
            np.testing.assert_allclose(original, colmap.project(camera, image, xyz), atol=1e-8)


def test_strong_barrel_distortion_does_not_fold_the_corners_back():
    # A wide-angle lens: r(1 - 0.3 r²) peaks at r = 1.054, and the view's corners (r = 1.25 at
    # s = 1) lie past the fold. Checking only that the border lands inside the source would pick
    # s = 0.55, where the top-left pixel shows the bottom-right corner
    camera = colmap.Camera(1, "SIMPLE_RADIAL", 800, 600, np.array([400.0, 400, 300, -0.3]))
    s = scene.undistorted_focal_scale(camera)
    corner = np.hypot(399.5, 299.5) / 400  # The corner pixels' centres, normalized at s = 1
    assert s == pytest.approx(corner / np.sqrt(1 / 0.9), rel=1e-6)  # The corners just reach the fold
    w, h = camera.width, camera.height
    ys, xs = np.mgrid[0:h, 0:w] + 0.5
    pixels = np.stack([xs, ys], -1)
    assert not scene.folds(camera, pixels, s)
    assert scene.folds(camera, pixels, s * 0.99)
    src = scene._source_pixels(camera, pixels, s)
    assert src.min() >= 0.5 - 1e-6 and (src - [w, h]).max() <= -0.5 + 1e-6
    # Each corner samples its own quadrant of the source
    for x, y in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        assert (src[y, x, 0] < w / 2) == (x == 0) and (src[y, x, 1] < h / 2) == (y == 0)


def test_distortion_that_folds_at_every_scale_is_rejected():
    camera = colmap.Camera(1, "SIMPLE_RADIAL", 800, 600, np.array([400.0, 400, 300, -5.0]))
    with pytest.raises(ValueError, match="folds"):
        scene.undistorted_focal_scale(camera)


@pytest.mark.parametrize("camera", [colmap.Camera(1, "PINHOLE", 8, 6, np.array([10.0, 11, 4, 3])), CAMERAS[0]])
def test_an_image_of_the_wrong_size_is_rejected(camera):
    # Not stretched to the camera's size, with or without distortion
    image = colmap.Image(1, np.array([1.0, 0, 0, 0]), np.zeros(3), 1, "x.png")
    model = colmap.Model({1: camera}, {1: image}, None)
    pixels = np.zeros((camera.height // 2, camera.width // 2, 3), np.uint8)
    with pytest.raises(ValueError, match="image is"):
        scene.pinhole_from_colmap(model, image, pixels)


def test_pinhole_cameras_are_copied_unchanged():
    camera = colmap.Camera(1, "PINHOLE", 8, 6, np.array([10.0, 11, 4, 3]))
    pixels = np.random.default_rng(0).integers(0, 256, size=(6, 8, 3), dtype=np.uint8)
    assert scene.undistorted_focal_scale(camera) == 1
    np.testing.assert_array_equal(scene.undistort_image(camera, pixels, 1.0), pixels)


def test_resized_size():
    assert scene.resized_size(748, 748, None) == (748, 748)
    assert scene.resized_size(748, 748, 1000) == (748, 748)
    assert scene.resized_size(1600, 900, 800) == (800, 450)
