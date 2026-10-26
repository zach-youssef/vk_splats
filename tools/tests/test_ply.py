"""INRIA PLY files and the std430 Splat buffer."""

import numpy as np
import pytest

from splatdata import splats


def random_splats(n, seed=0):
    rng = np.random.default_rng(seed)
    return splats.Splats(position=rng.normal(size=(n, 3)), log_scale=rng.normal(size=(n, 3)),
                         rotation=rng.normal(size=(n, 4)), opacity_logit=rng.normal(size=n),
                         sh=rng.normal(size=(n, 16, 3)))


def assert_same(a, b):
    for name in ("position", "log_scale", "rotation", "opacity_logit", "sh"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)


def test_ply_round_trip(tmp_path):
    original = random_splats(257)
    splats.write_ply(str(tmp_path / "a.ply"), original)
    assert_same(splats.read_ply(str(tmp_path / "a.ply")), original)


def write_raw_ply(path, names, rows, extra_header=()):
    header = ["ply", "format binary_little_endian 1.0", "comment test", f"element vertex {len(rows)}"]
    header += [f"property float {n}" for n in names] + list(extra_header) + ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode())
        f.write(np.asarray(rows, dtype="<f4").tobytes())


def test_ply_layout_is_inrias(tmp_path):
    # f_rest is channel-major (the reference code's features_rest.transpose(1, 2).flatten()):
    # f_rest_{c·15 + k − 1} is coefficient k of channel c
    names = splats.ply_property_names()
    row = np.arange(len(names), dtype=np.float32)
    write_raw_ply(str(tmp_path / "a.ply"), names, [row])
    s = splats.read_ply(str(tmp_path / "a.ply"))
    value = dict(zip(names, row))
    np.testing.assert_array_equal(s.position[0], [value["x"], value["y"], value["z"]])
    for c in range(3):
        assert s.sh[0, 0, c] == value[f"f_dc_{c}"]
        for k in range(1, 16):
            assert s.sh[0, k, c] == value[f"f_rest_{c * 15 + k - 1}"]
    assert s.opacity_logit[0] == value["opacity"]
    np.testing.assert_array_equal(s.log_scale[0], [value[f"scale_{i}"] for i in range(3)])
    np.testing.assert_array_equal(s.rotation[0], [value[f"rot_{i}"] for i in range(4)])
    # ... and writing it back gives the same bytes after the header (normals zeroed)
    splats.write_ply(str(tmp_path / "b.ply"), s)
    body = lambda p: open(p, "rb").read().split(b"end_header\n", 1)[1]  # noqa: E731
    expected = row.copy()
    expected[3:6] = 0
    assert body(str(tmp_path / "b.ply")) == expected.tobytes()


def test_ply_of_lower_degree_and_other_order(tmp_path):
    # SH degree 1 (9 f_rest), properties shuffled, a face element after the vertices
    names = splats.ply_property_names(rest_per_channel=3)
    order = np.random.default_rng(1).permutation(len(names))
    rows = np.random.default_rng(2).normal(size=(4, len(names)))
    write_raw_ply(str(tmp_path / "a.ply"), [names[i] for i in order], rows[:, order],
                  ["element face 0", "property list uchar int vertex_indices"])
    s = splats.read_ply(str(tmp_path / "a.ply"))
    value = {n: rows[:, i].astype(np.float32) for i, n in enumerate(names)}
    for c in range(3):
        for k in range(1, 4):
            np.testing.assert_array_equal(s.sh[:, k, c], value[f"f_rest_{c * 3 + k - 1}"])
    assert np.all(s.sh[:, 4:, :] == 0)


def test_ply_errors(tmp_path):
    names = [n for n in splats.ply_property_names() if n != "opacity"]
    write_raw_ply(str(tmp_path / "a.ply"), names, [np.zeros(len(names))])
    with pytest.raises(ValueError, match="opacity"):
        splats.read_ply(str(tmp_path / "a.ply"))
    (tmp_path / "b.ply").write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
    with pytest.raises(ValueError, match="binary_little_endian"):
        splats.read_ply(str(tmp_path / "b.ply"))


def test_std430_layout():
    s = random_splats(3)
    data = splats.to_std430(s)
    assert len(data) == 3 * 240
    floats = np.frombuffer(data, "<f4").reshape(3, 60)
    np.testing.assert_array_equal(floats[:, 0:3], s.position)
    np.testing.assert_array_equal(floats[:, 3], s.opacity_logit)
    np.testing.assert_array_equal(floats[:, 4:7], s.log_scale)
    assert np.all(floats[:, 7] == 0)  # std430 padding before the vec4
    np.testing.assert_array_equal(floats[:, 8:12], s.rotation)
    np.testing.assert_array_equal(floats[:, 12:60], s.sh.reshape(3, 48))  # sh[3k + c]
    assert_same(splats.from_std430(data), s)
    with pytest.raises(ValueError):
        splats.from_std430(data[:-4])


def test_initialization_from_points():
    # A unit lattice: every point's 3 nearest neighbours are at distance 1, so log-scale 0
    g = np.arange(5.0)
    points = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    rgb = np.random.default_rng(0).integers(0, 256, size=(len(points), 3), dtype=np.uint8)
    s = splats.from_points(points, rgb)
    np.testing.assert_allclose(s.log_scale, 0, atol=1e-6)
    np.testing.assert_array_equal(s.rotation, np.tile([1, 0, 0, 0], (len(points), 1)))
    np.testing.assert_allclose(1 / (1 + np.exp(-s.opacity_logit)), 0.1, rtol=1e-6)
    np.testing.assert_allclose(splats.SH_C0 * s.sh[:, 0, :] + 0.5, rgb / 255, atol=1e-6)
    assert np.all(s.sh[:, 1:, :] == 0)


def test_nearest_neighbour_scale_is_the_rms_of_three():
    # Distances 1, 2 and 3 to the neighbours of the origin: log(sqrt((1 + 4 + 9) / 3))
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3], [0, 0, 100.0]])
    assert splats.nearest_neighbour_log_scale(points)[0] == pytest.approx(0.5 * np.log(14 / 3))
