"""Parses colmap/colmap_exports3 and checks the camera convention by reprojection: every point,
projected with its pose and camera, lands where COLMAP recorded its observations."""

import os

import numpy as np
import pytest
from conftest import REPO

from splatdata import colmap

MODEL = os.path.join(REPO, "colmap", "colmap_exports3")


@pytest.fixture(scope="module")
def model():
    return colmap.read_model(MODEL)


def observation_errors(model, misread_quaternion=False):
    """Each track entry's reprojection error in pixels, in Points3D's track order."""
    points = model.points
    errors = np.full(len(points.track_point), np.nan)
    for image in model.images.values():
        if misread_quaternion:  # The old code's bug: (w, x, y, z) read as (x, y, z, w)
            image = colmap.Image(image.id, np.roll(image.qvec, -1), image.tvec, image.camera_id, image.name)
        sel = points.track_image_id == image.id
        projected = colmap.project(model.cameras[image.camera_id], image, points.xyz[points.track_point[sel]])
        recorded = model.images[image.id].xys[points.track_point2D_idx[sel]]
        errors[sel] = np.linalg.norm(projected - recorded, axis=1)
    return errors


def test_counts_match_the_headers(model):
    # The headers: 1 camera, 46 images, 12900 points with mean track length 6.2789922480620159
    assert len(model.cameras) == 1
    assert len(model.images) == 46
    assert len(model.points.ids) == 12900
    assert len(model.points.track_point) / len(model.points.ids) == pytest.approx(6.2789922480620159, rel=1e-12)
    # Points with long tracks (21 or more tokens, which the old parser dropped) are all there
    lengths = np.bincount(model.points.track_point)
    assert np.sum(lengths >= 7) > 0


def test_tracks_and_points2D_agree(model):
    points = model.points
    for image in model.images.values():
        sel = points.track_image_id == image.id
        assert np.all(points.track_point2D_idx[sel] < len(image.xys))
        np.testing.assert_array_equal(image.point3D_ids[points.track_point2D_idx[sel]],
                                      points.ids[points.track_point[sel]])


def test_points_are_in_front_of_their_cameras(model):
    points = model.points
    for image in model.images.values():
        sel = points.track_image_id == image.id
        assert np.all(image.world_to_camera(points.xyz[points.track_point[sel]])[:, 2] > 0)


def test_reprojection_matches_colmaps_recorded_errors(model):
    errors = observation_errors(model)
    assert not np.any(np.isnan(errors))
    # COLMAP records each point's mean reprojection error (ERROR, pixels). Recomputing it from the
    # parsed poses, intrinsics and SIMPLE_RADIAL distortion reproduces it to rounding, which a
    # wrong convention (quaternion order, R vs Rᵀ, pixel-centre offset, distortion) can't.
    mean = np.bincount(model.points.track_point, errors) / np.bincount(model.points.track_point)
    np.testing.assert_allclose(mean, model.points.error, rtol=0, atol=1e-6)
    # The 1.5 px bound holds for the typical observation, not every one: COLMAP kept points whose
    # mean error is up to 3.8 px (22% are over 1.5 px), so all observations are under 5 px
    assert np.median(errors) < 1.5
    assert np.max(errors) < 5


def test_misread_quaternion_fails(model):
    # The check has teeth: the old code's quaternion order puts observations far off
    assert np.median(observation_errors(model, misread_quaternion=True)) > 50


def test_two_line_records_with_empty_points(tmp_path):
    (tmp_path / "images.txt").write_bytes(
        b"# comment\r\n"
        b"1 1 0 0 0 0 0 1 1 a b.png\r\n"
        b"\r\n"
        b"2 0.5 0.5 0.5 0.5 1 2 3 1 c.png\r\n"
        b"10 20 7 30.5 40 -1\r\n")
    images = colmap.read_images(str(tmp_path / "images.txt"))
    assert images[1].name == "a b.png" and len(images[1].xys) == 0
    np.testing.assert_array_equal(images[2].qvec, [0.5, 0.5, 0.5, 0.5])
    np.testing.assert_array_equal(images[2].xys, [[10, 20], [30.5, 40]])
    np.testing.assert_array_equal(images[2].point3D_ids, [7, -1])


def test_camera_models():
    for model, params in [("SIMPLE_PINHOLE", [100, 50, 40]), ("PINHOLE", [100, 110, 50, 40]),
                          ("SIMPLE_RADIAL", [100, 50, 40, 0.1]), ("RADIAL", [100, 50, 40, 0.1, -0.02]),
                          ("OPENCV", [100, 110, 50, 40, 0.1, -0.02, 0.001, -0.002])]:
        camera = colmap.Camera(1, model, 100, 80, np.array(params, dtype=float))
        assert camera.focal == ((100, 110) if "fy" in colmap.CAMERA_MODELS[model] else (100, 100))
        assert camera.principal_point == (50, 40)
        normalized = np.random.default_rng(0).uniform(-0.4, 0.4, size=(200, 2))
        np.testing.assert_allclose(camera.to_normalized(camera.to_pixels(normalized)), normalized, atol=1e-10)
    with pytest.raises(ValueError):
        colmap.Camera(1, "FISHEYE", 10, 10, np.zeros(4))
    with pytest.raises(ValueError):
        colmap.Camera(1, "PINHOLE", 10, 10, np.zeros(3))


def test_opencv_distortion_matches_colmaps_formula():
    k1, k2, p1, p2 = 0.1, -0.02, 0.001, -0.002
    camera = colmap.Camera(1, "OPENCV", 100, 80, np.array([100, 110, 50, 40, k1, k2, p1, p2]))
    u, v = 0.3, -0.2
    r2 = u * u + v * v
    radial = k1 * r2 + k2 * r2 * r2
    expected = [u + u * radial + 2 * p1 * u * v + p2 * (r2 + 2 * u * u),
                v + v * radial + 2 * p2 * u * v + p1 * (r2 + 2 * v * v)]
    np.testing.assert_allclose(camera.distort(np.array([u, v])), expected, rtol=1e-15)
