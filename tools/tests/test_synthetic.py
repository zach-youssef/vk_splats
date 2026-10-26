"""make_synthetic.py: the scenes hold what docs/building-and-testing.md says, in the converter's
format, and the same seed writes the same bytes."""

import filecmp
import json
import math
import os

import numpy as np
import pytest

import make_synthetic as ms
from splatdata import quaternion, scene, splats


def read_scene(path):
    with open(os.path.join(path, "manifest.json")) as f:
        manifest = json.load(f)
    with open(os.path.join(path, "splats.bin"), "rb") as f:
        s = splats.from_std430(f.read())
    with open(os.path.join(path, "cameras.bin"), "rb") as f:
        c = scene.cameras_from_std430(f.read())
    assert manifest["splatCount"] == len(s) and manifest["cameraCount"] == len(c)
    return manifest, s, c


def world_to_camera(camera):
    return camera["worldToCamera"].astype(np.float64).reshape(4, 4).T  # column-major


def project(camera, splat_set):
    """Camera-space means, 2D means and 3-sigma radii (px) of the splats, by the general EWA
    projection (Eqs. 5, 6; the 0.3 low-pass term and the radius's 0.1 floor from the reference
    code), evaluated here independently of the generator."""
    m = world_to_camera(camera)
    fx, fy, cx, cy = camera["intrinsics"].astype(np.float64)
    p = np.einsum("ij,nj->ni", m[:3, :3], splat_set.position.astype(np.float64)) + m[:3, 3]
    r = quaternion.to_matrix(splat_set.rotation)
    s = np.exp(splat_set.log_scale.astype(np.float64))
    sigma = np.einsum("nij,nj,nkj->nik", r, s * s, r)
    x, y, z = p.T
    jac = np.zeros((len(p), 2, 3))
    jac[:, 0, 0], jac[:, 0, 2] = fx / z, -fx * x / z**2
    jac[:, 1, 1], jac[:, 1, 2] = fy / z, -fy * y / z**2
    t = np.einsum("nij,jk->nik", jac, m[:3, :3])
    cov = np.einsum("nij,njk,nlk->nil", t, sigma, t) + 0.3 * np.eye(2)
    mid = 0.5 * (cov[:, 0, 0] + cov[:, 1, 1])
    det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] ** 2
    radius = np.ceil(3 * np.sqrt(mid + np.sqrt(np.maximum(mid * mid - det, 0.1))))
    return p, np.stack([fx * x / z + cx, fy * y / z + cy], 1), radius


def tree_bytes(path):
    out = {}
    for root, _, files in os.walk(path):
        for name in files:
            full = os.path.join(root, name)
            with open(full, "rb") as f:
                out[os.path.relpath(full, path)] = f.read()
    return out


# --- S1 ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def s1(tmp_path_factory):
    out = tmp_path_factory.mktemp("s1")
    ms.main(["s1", str(out)])
    return out


def test_s1_writes_every_variant(s1):
    assert sorted(os.listdir(s1)) == sorted(ms.S1_VARIANTS)
    for name in ms.S1_VARIANTS:
        manifest, _, cameras = read_scene(os.path.join(s1, name))
        assert manifest["source"]["generator"] == "make_synthetic.py"
        assert manifest["source"]["variant"] == name
        assert manifest["files"]["images"] == [] and np.all(cameras["imageBatch"] == scene.NO_IMAGE)
        # Not the camera-based 0 (all cameras at the origin): 1.1 times the farthest splat's distance
        _, got, _ = read_scene(os.path.join(s1, name))
        far = np.max(np.linalg.norm(got.position.astype(np.float64), axis=1))
        assert manifest["extent"] == pytest.approx(1.1 * far, rel=1e-6) and manifest["extent"] > 0


# The numbers of tests/cpp/closed_form_test.cpp's setups, written out again so a change on either
# side shows: (f, width, height, splats as (x, y, z, scales, theta, opacity logit, dc), background,
# target, camera rotations phi)
EXPECTED_S1 = {
    "isotropic": (50, 33, 33, [(0, 0, 4, (0.2, 0.2, 0.2), 0.3, 0, (1, 0, -1))], (0, 0, 0), None, [0]),
    "anisotropic": (50, 33, 33, [(0, 0, 4, (0.3, 0.1, 0.5), 0, 0, (1, 0, -1))], (0, 0, 0), None, [0]),
    "anisotropic-rot90": (50, 33, 33, [(0, 0, 4, (0.3, 0.1, 0.5), math.pi / 2, 0, (1, 0, -1))], (0, 0, 0), None,
                          [0]),
    "anisotropic-rot45": (50, 33, 33, [(0, 0, 4, (0.3, 0.1, 0.5), math.pi / 4, 0, (1, 0, -1))], (0, 0, 0), None,
                          [0]),
    "off-axis": (50, 33, 33, [(0.4, -0.5, 4, (0.25, 0.08, 0.15), 0.9, 0, (1, 0, -1))], (0, 0, 0), None, [0]),
    "one-splat": (50, 33, 33, [(0, 0, 4, (0.3, 0.1, 0.2), 0, 0.5, (1, 0, -1))], (0.1, 0.2, 0.3), None, [0]),
    "one-splat-clamped": (50, 33, 33, [(0, 0, 4, (0.3, 0.1, 0.2), 0, 12, (1, 0, -1))], (0.1, 0.2, 0.3), None,
                          [0]),
    "gradient": (20, 15, 15, [(0.15, -0.1, 2, (0.45, 0.35, 0.3), 0.4, 0.8, (0.8, -0.6, 0.2))], (0.1, 0.2, 0.3),
                 (0.7, 0.4, 0.05), [0]),
    "two-splat": (50, 33, 33, [(0, 0, 6, (0.5, 0.3, 0.2), 0, 1.2, (-1, 1.5, 0.5)),
                               (0, 0, 3, (0.2, 0.2, 0.2), 0, 0.4, (1.5, -1, 0))], (0, 0.1, 0.2), None, [0]),
    "rotated-camera": (50, 33, 33, [(0.3, -0.2, 4, (0.3, 0.08, 0.15), 0.7, 0, (0, 0, 0))], (0, 0, 0), None,
                       [0.4, -1.1, 2.5]),
    "rotated-camera-aligned": (50, 33, 33, [(0.3, -0.2, 4, (0.3, 0.08, 0.15), 0, 0, (0, 0, 0))], (0, 0, 0), None,
                               [0.4, -1.1, 2.5]),
}


@pytest.mark.parametrize("name", sorted(EXPECTED_S1))
def test_s1_variant_matches_the_oracle_test_setup(s1, name):
    f, width, height, expected, background, target, phis = EXPECTED_S1[name]
    manifest, got, cameras = read_scene(os.path.join(s1, name))
    oracle = manifest["source"]["oracle"]
    assert (oracle["f"], oracle["width"], oracle["height"]) == (f, width, height)
    assert oracle["background"] == pytest.approx(background)
    assert oracle.get("target") == (None if target is None else pytest.approx(target))
    assert oracle["cameraPhi"] == pytest.approx(phis)
    assert len(oracle["splats"]) == len(expected) == len(got)

    for k, (x, y, z, scale, theta, opacity, dc) in enumerate(expected):
        o = oracle["splats"][k]
        assert [o["x"], o["y"], o["z"]] == pytest.approx([x, y, z])
        assert o["logScale"] == pytest.approx(np.log(scale))
        assert (o["theta"], o["opacityLogit"]) == pytest.approx((theta, opacity))
        assert o["dc"] == pytest.approx(dc)
        # The buffer holds the same splat, as the kernels read it (float32)
        np.testing.assert_allclose(got.position[k], [x, y, z], atol=1e-7)
        np.testing.assert_allclose(got.log_scale[k], np.log(scale), rtol=1e-6)
        np.testing.assert_allclose(got.rotation[k], [math.cos(theta / 2), 0, 0, math.sin(theta / 2)], atol=1e-7)
        assert got.opacity_logit[k] == pytest.approx(opacity)
        np.testing.assert_allclose(got.sh[k, 0], dc, atol=1e-7)
        assert np.all(got.sh[k, 1:] == 0)

    assert len(cameras) == len(phis)
    for camera, phi in zip(cameras, phis):
        np.testing.assert_allclose(camera["intrinsics"], [f, f, width / 2, height / 2])
        assert list(camera["size"]) == [width, height]
        m = world_to_camera(camera)
        c, s = math.cos(phi), math.sin(phi)
        np.testing.assert_allclose(m[:3, :3], [[c, -s, 0], [s, c, 0], [0, 0, 1]], atol=1e-7)
        np.testing.assert_array_equal(m[:3, 3], 0)


def test_s1_rotated_camera_sees_world_to_camera_about_z(s1):
    # closed_form.h's world_to_camera_about_z: mean Rz(phi) p, rotation theta + phi
    manifest, got, cameras = read_scene(os.path.join(s1, "rotated-camera"))
    world = manifest["source"]["oracle"]["splats"][0]
    for camera, phi in zip(cameras, manifest["source"]["oracle"]["cameraPhi"]):
        p, _, _ = project(camera, got)
        c, s = math.cos(phi), math.sin(phi)
        np.testing.assert_allclose(p[0], [c * world["x"] - s * world["y"], s * world["x"] + c * world["y"], world["z"]],
                                   atol=1e-6)
        r = world_to_camera(camera)[:3, :3] @ quaternion.to_matrix(got.rotation[0])
        assert math.atan2(r[1, 0], r[0, 0]) == pytest.approx(math.remainder(world["theta"] + phi, 2 * math.pi),
                                                             abs=1e-6)


def test_s1_gradient_footprint_covers_the_image(s1):
    # The setup the gradient oracle needs: the 3-sigma square covers [0, 15]^2
    _, got, cameras = read_scene(os.path.join(s1, "gradient"))
    _, mean, radius = project(cameras[0], got)
    assert mean[0, 0] - radius[0] <= 0 and mean[0, 0] + radius[0] >= 15
    assert mean[0, 1] - radius[0] <= 0 and mean[0, 1] + radius[0] >= 15


def test_s1_variant_option_writes_one(tmp_path):
    ms.main(["s1", str(tmp_path), "--variant", "two-splat"])
    assert os.listdir(tmp_path) == ["two-splat"]


# --- S2 ---------------------------------------------------------------------------------------------

def test_s2_cases(tmp_path):
    ms.main(["s2", str(tmp_path), "--seed", "3"])
    manifest, got, cameras = read_scene(tmp_path)
    cases = manifest["source"]["cases"]
    assert len(got) == len(cases) == 8 and cases.count("overlapping") == 5
    assert [list(c["size"]) for c in cameras] == [[64, 64], [80, 48]]
    np.testing.assert_allclose(world_to_camera(cameras[0]), np.eye(4), atol=1e-7)
    assert not np.allclose(world_to_camera(cameras[1])[:3, :3], np.eye(3), atol=0.1)

    for k, camera in enumerate(cameras):
        p, mean, radius = project(camera, got)
        w, h = camera["size"]
        behind = cases.index("behind")
        assert p[behind, 2] < 0, f"camera {k}"
        for i in (i for i, c in enumerate(cases) if c == "overlapping"):
            assert 0 < mean[i, 0] < w and 0 < mean[i, 1] < h, f"camera {k} splat {i}"
    p, mean, radius = project(cameras[0], got)
    overlapping = [i for i, c in enumerate(cases) if c == "overlapping"]
    depths = p[overlapping, 2]
    assert depths.max() - depths.min() <= 0.1 and len(np.unique(depths)) == 5
    # Every pair of overlapping splats' 3-sigma discs intersect
    for i in overlapping:
        for j in overlapping:
            assert np.linalg.norm(mean[i] - mean[j]) < radius[i] + radius[j]
    partial, off = cases.index("off-screen-partial"), cases.index("off-screen")
    assert mean[partial, 0] > 64 and mean[partial, 0] - radius[partial] < 64
    assert mean[off, 0] - radius[off] > 64
    # Anisotropic, unnormalized rotations, every SH band in use
    s = np.exp(got.log_scale[overlapping])
    assert np.all(s.max(1) / s.min(1) > 1.05)
    assert not np.allclose(np.linalg.norm(got.rotation, axis=1), 1)
    assert np.all(np.abs(got.sh[:, 1:]).max(axis=(0, 2)) > 0)


# --- S3 ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def s3(tmp_path_factory):
    out = tmp_path_factory.mktemp("s3")
    ms.main(["s3", str(out)])
    return out


def test_s3_splats_and_cameras(s3):
    manifest, got, cameras = read_scene(s3)
    assert len(got) == 2000 and manifest["shDegree"] == 3
    assert np.all(np.linalg.norm(got.position, axis=1) <= 1 + 1e-6)
    assert np.all(np.abs(got.sh[:, 1:]).max(axis=(0, 2)) > 0)
    assert [list(c["size"]) for c in cameras] == [[128, 128]] * 24 + [[160, 96]]
    assert manifest["test"] == [0, 8, 16] and len(manifest["train"]) == 22
    wide = [c["index"] for c in manifest["cameras"] if c["width"] != c["height"]]
    assert wide == [24] and manifest["cameras"][24]["split"] == "train"
    for k, camera in enumerate(cameras):
        m = world_to_camera(camera)
        assert np.linalg.det(m[:3, :3]) == pytest.approx(1, abs=1e-6)
        assert np.linalg.norm(camera["center"]) == pytest.approx(4, abs=1e-5)
        # Looking at the origin, with the whole unit ball in view
        p, mean, _ = project(camera, got)
        origin = m[:3, 3]
        assert origin[:2] == pytest.approx([0, 0], abs=1e-5) and origin[2] == pytest.approx(4, abs=1e-5)
        w, h = camera["size"]
        assert np.all(p[:, 2] > 2.9)
        assert np.all((mean > 0) & (mean < [w, h])), f"camera {k}"
        # Up in the image is world +z (apart from the views near the poles)
        assert m[1, 2] <= 0
    # The cameras are spread over the whole sphere
    centers = np.array([c["center"] for c in cameras[:24]])
    assert np.linalg.norm(centers.mean(0)) < 0.5 and centers[:, 2].min() < -3.5 and centers[:, 2].max() > 3.5


def test_s3_count(tmp_path):
    ms.main(["s3", str(tmp_path), "--count", "50"])
    assert read_scene(tmp_path)[0]["splatCount"] == 50


# --- Determinism ------------------------------------------------------------------------------------

@pytest.mark.parametrize("command", [["s1"], ["s2", "--seed", "5"], ["s3", "--seed", "5", "--count", "300"]])
def test_same_seed_same_bytes(tmp_path, command):
    a, b = tmp_path / "a", tmp_path / "b"
    ms.main([command[0], str(a)] + command[1:])
    ms.main([command[0], str(b)] + command[1:])
    files = tree_bytes(a)
    assert files and files == tree_bytes(b)


@pytest.mark.parametrize("command", ["s2", "s3"])
def test_another_seed_other_splats(tmp_path, command):
    ms.main([command, str(tmp_path / "a"), "--seed", "1"] + (["--count", "100"] if command == "s3" else []))
    ms.main([command, str(tmp_path / "b"), "--seed", "2"] + (["--count", "100"] if command == "s3" else []))
    assert not filecmp.cmp(tmp_path / "a" / "splats.bin", tmp_path / "b" / "splats.bin", shallow=False)
    assert filecmp.cmp(tmp_path / "a" / "cameras.bin", tmp_path / "b" / "cameras.bin", shallow=False)
