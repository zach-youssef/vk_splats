"""The scene writer and convert.py, end to end on a slice of colmap_exports3."""

import json
import os
import shutil

import numpy as np
import pytest
from conftest import REPO

import convert
from splatdata import colmap, quaternion, scene, splats

MODEL = os.path.join(REPO, "colmap", "colmap_exports3")
IMAGES = os.path.join(REPO, "colmap", "cs5330_dense_resized")


def test_every_eighth_image_in_name_order_is_held_out():
    names = [f"img{i:02}.jpg" for i in range(20)][::-1]
    held_out = scene.test_split(names)
    assert sorted(n for n, h in zip(names, held_out) if h) == ["img00.jpg", "img08.jpg", "img16.jpg"]


def camera(name, w, h, image=True, seed=0):
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8) if image else None
    return scene.PinholeCamera(name, w, h, 50.0, 60.0, w / 2, h / 2,
                               quaternion.to_matrix(rng.normal(size=4)), rng.normal(size=3), pixels)


def test_images_are_packed_rgba8_in_batches():
    cameras = [camera("a", 4, 3, seed=1), camera("b", 5, 2, image=False), camera("c", 2, 2, seed=2),
               camera("d", 3, 3, seed=3)]
    # 48 bytes for a, 16 for c, 36 for d: at most 64 bytes per batch puts a and c together
    buffers, placement = scene.pack_images(cameras, max_bytes=64)
    assert placement == [(0, 0), (scene.NO_IMAGE, 0), (0, 12), (1, 0)]
    assert [len(b) for b in buffers] == [64, 36]
    pixels = np.frombuffer(buffers[0], "<u4")
    a = cameras[0].image.astype(np.uint32)
    np.testing.assert_array_equal(pixels[:12], (a[..., 0] | a[..., 1] << 8 | a[..., 2] << 16 | 255 << 24).reshape(-1))


def test_camera_buffer_layout():
    cameras = [camera("a", 4, 3), camera("b", 5, 2, image=False)]
    data = scene.cameras_to_std430(cameras, [(0, 7), (scene.NO_IMAGE, 0)])
    assert len(data) == 2 * 112
    c = scene.cameras_from_std430(data)
    m = c[0]["worldToCamera"].reshape(4, 4).T  # column-major
    np.testing.assert_allclose(m[:3, :3], cameras[0].rotation, atol=1e-7)
    np.testing.assert_allclose(m[:3, 3], cameras[0].translation, atol=1e-6)
    np.testing.assert_array_equal(m[3], [0, 0, 0, 1])
    np.testing.assert_allclose(c[0]["center"], cameras[0].center, atol=1e-6)
    np.testing.assert_allclose(c[0]["intrinsics"], [50, 60, 2, 1.5])
    assert c[0]["pixelOffset"] == 7 and list(c[0]["size"]) == [4, 3] and c[0]["imageBatch"] == 0
    assert c[1]["imageBatch"] == scene.NO_IMAGE
    words = np.frombuffer(data, "<u4").reshape(2, 28)
    assert np.all(words[:, 27] == 0)  # Trailing padding


def test_extent_is_computed_from_the_training_cameras(tmp_path):
    cameras = [camera(str(i), 4, 3, image=False, seed=i) for i in range(4)]
    for c, x in zip(cameras, (0.0, 1.0, 2.0, 100.0)):
        c.translation = -np.einsum("ij,j->i", c.rotation, [x, 0, 0])  # centre (x, 0, 0)
    s = splats.from_points(np.random.default_rng(0).normal(size=(5, 3)), np.zeros((5, 3), np.uint8))
    held_out = [False, False, False, True]  # The outlier at x = 100 is a test camera
    manifest = scene.write_scene(str(tmp_path / "a"), s, cameras, held_out)
    assert manifest["extent"] == pytest.approx(1.1)
    manifest = scene.write_scene(str(tmp_path / "b"), s, cameras)
    assert manifest["extent"] == pytest.approx(1.1 * 74.25)


def sliced_model(tmp_path, count):
    """colmap_exports3 with only its first count images (tracks still refer to the others)."""
    out = tmp_path / "model"
    out.mkdir()
    for name in ("cameras.txt", "points3D.txt"):
        shutil.copy(os.path.join(MODEL, name), out / name)
    lines = open(os.path.join(MODEL, "images.txt"), "rb").read().splitlines(keepends=True)
    header = [line for line in lines if line.startswith(b"#")]
    data = [line for line in lines if not line.startswith(b"#")]
    (out / "images.txt").write_bytes(b"".join(header + data[:2 * count]))
    return str(out)


def test_convert_colmap_with_images(tmp_path):
    model_dir = sliced_model(tmp_path, 2)
    out = tmp_path / "scene"
    convert.main(["colmap", model_dir, str(out), "--images", IMAGES, "--max-size", "96"])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["splatCount"] == 12900 and manifest["cameraCount"] == 2
    assert manifest["files"] == {"splats": "splats.bin", "cameras": "cameras.bin", "images": ["images_0.bin"]}
    assert manifest["test"] == [0] and manifest["train"] == [1]
    model = colmap.read_model(model_dir)
    names = sorted(im.name for im in model.images.values())
    assert [c["name"] for c in manifest["cameras"]] == names

    assert os.path.getsize(out / "splats.bin") == 12900 * 240
    assert os.path.getsize(out / "images_0.bin") == 2 * 96 * 96 * 4
    cams = scene.cameras_from_std430((out / "cameras.bin").read_bytes())
    for entry, packed in zip(manifest["cameras"], cams):
        assert (entry["width"], entry["height"]) == (96, 96)
        assert list(packed["size"]) == [96, 96]
        assert packed["pixelOffset"] == entry["pixelOffset"] == entry["index"] * 96 * 96
        np.testing.assert_allclose(packed["intrinsics"], [entry["fx"], entry["fy"], entry["cx"], entry["cy"]], rtol=1e-6)
        # The packed pixels are the PNG's
        png = scene.load_rgb(str(out / entry["png"]))
        words = np.frombuffer((out / "images_0.bin").read_bytes(), "<u4")[entry["pixelOffset"]:][:96 * 96]
        np.testing.assert_array_equal(words & 0xFF, png[..., 0].reshape(-1))
        np.testing.assert_array_equal(words >> 24, 255)
    # The packed float32 cameras project the points onto COLMAP's observations: through
    # worldToCamera and the intrinsics as a kernel would, then back through the distortion at
    # the source's size
    original = model.cameras[1]
    scale = scene.undistorted_focal_scale(original)
    by_name = {im.name: im for im in model.images.values()}
    index = {int(i): k for k, i in enumerate(model.points.ids)}
    for entry, packed in zip(manifest["cameras"], cams):
        image = by_name[entry["name"]]
        seen = image.point3D_ids >= 0
        xyz = model.points.xyz[[index[int(i)] for i in image.point3D_ids[seen]]].astype(np.float32)
        m = packed["worldToCamera"].reshape(4, 4).T.astype(np.float64)
        p = np.einsum("ij,nj->ni", m[:3, :3], xyz.astype(np.float64)) + m[:3, 3]
        fx, fy, cx, cy = packed["intrinsics"].astype(np.float64)
        pixel = np.stack([fx * p[:, 0] / p[:, 2] + cx, fy * p[:, 1] / p[:, 2] + cy], 1)
        width, height = packed["size"]
        back = scene._source_pixels(original, pixel * [original.width / width, original.height / height], scale)
        np.testing.assert_allclose(back, colmap.project(original, image, model.points.xyz[
            [index[int(i)] for i in image.point3D_ids[seen]]]), atol=2e-3)
        assert np.median(np.linalg.norm(back - image.xys[seen], axis=1)) < 1.5

    # 668·(1.02 for the undistorted view)·96/748
    assert manifest["cameras"][0]["fx"] == pytest.approx(667.888 * 1.0207 * 96 / 748, rel=1e-3)
    assert manifest["cameras"][0]["cx"] == pytest.approx(48)

    # Splats round-trip through a PLY
    convert.main(["bin-to-ply", str(out / "splats.bin"), str(tmp_path / "s.ply")])
    convert.main(["ply-to-bin", str(tmp_path / "s.ply"), str(tmp_path / "s.bin")])
    assert (tmp_path / "s.bin").read_bytes() == (out / "splats.bin").read_bytes()


def test_convert_colmap_without_images_takes_splats_from_a_ply(tmp_path):
    s = splats.from_points(np.random.default_rng(0).normal(size=(10, 3)), np.zeros((10, 3), np.uint8))
    splats.write_ply(str(tmp_path / "s.ply"), s)
    out = tmp_path / "scene"
    convert.main(["colmap", MODEL, str(out), "--splats", str(tmp_path / "s.ply")])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["splatCount"] == 10 and manifest["cameraCount"] == 46
    assert manifest["files"]["images"] == [] and not (out / "images").exists()
    assert len(manifest["test"]) == 6 and len(manifest["train"]) == 40
    assert all("png" not in c for c in manifest["cameras"])
    assert np.all(scene.cameras_from_std430((out / "cameras.bin").read_bytes())["imageBatch"] == scene.NO_IMAGE)
    assert manifest["extent"] > 0
