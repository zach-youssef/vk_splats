"""libsplat_native through its ctypes binding (splatdata.native): every C function, the PLY
reader's layout, SH degrees, property types and errors, and the splats.bin reader. The stream
encoder's functions are tested in test_stream.py."""

import errno

import numpy as np
import pytest

from splatdata import native, splats

# A PLY scalar type of each kind, as numpy writes it
TYPES = {"char": "<i1", "uchar": "<u1", "short": "<i2", "ushort": "<u2", "int": "<i4",
         "uint": "<u4", "float": "<f4", "double": "<f8", "int8": "<i1", "uint8": "<u1",
         "int16": "<i2", "uint16": "<u2", "int32": "<i4", "uint32": "<u4", "float32": "<f4",
         "float64": "<f8"}


def write_typed_ply(path, names, types, columns, extra_header=()):
    """A binary little-endian PLY of one vertex element: property names[i] of PLY type types[i],
    its values columns[i] (one per vertex)."""
    dtype = np.dtype([(n, TYPES[t]) for n, t in zip(names, types)])
    data = np.zeros(len(columns[0]) if columns else 0, dtype)
    for n, column in zip(names, columns):
        data[n] = column
    header = ["ply", "format binary_little_endian 1.0", "comment test", f"element vertex {len(data)}"]
    header += [f"property {t} {n}" for n, t in zip(names, types)] + list(extra_header) + ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode())
        f.write(data.tobytes())
    return {n: data[n].astype(np.float64).astype(np.float32) for n in names}


def random_fixture(path, degree, seed, mixed_types):
    """A PLY of SH degree `degree`: 5 vertices, the properties shuffled, of mixed types or all
    floats, with a face element after the vertices. Returns each property's values as float32."""
    rng = np.random.default_rng(seed)
    names = splats.ply_property_names(rest_per_channel=(degree + 1) ** 2 - 1)
    names = [names[i] for i in rng.permutation(len(names))]
    types = [rng.choice(list(TYPES)) if mixed_types else "float" for _ in names]
    columns = []
    for t in types:
        info = np.iinfo(TYPES[t]) if TYPES[t][1] in "iu" else None
        columns.append(rng.integers(info.min, info.max, size=5, endpoint=True) if info
                       else rng.normal(size=5) * 10)
    return write_typed_ply(path, names, types, columns,
                           ["element face 1", "property list uchar int vertex_indices"])


def expected_splats(values, degree):
    """The Splats a PLY of these property values holds (the reader's contract, independently)."""
    n = len(values["x"])
    per_channel = (degree + 1) ** 2 - 1
    sh = np.zeros((n, 16, 3), np.float32)
    for c in range(3):
        sh[:, 0, c] = values[f"f_dc_{c}"]
        for k in range(1, per_channel + 1):
            sh[:, k, c] = values[f"f_rest_{c * per_channel + k - 1}"]
    return splats.Splats(
        position=np.stack([values[a] for a in "xyz"], 1),
        log_scale=np.stack([values[f"scale_{i}"] for i in range(3)], 1),
        rotation=np.stack([values[f"rot_{i}"] for i in range(4)], 1),
        opacity_logit=values["opacity"], sh=sh)


def read_native(path):
    data, degree = native.read_ply(path)
    return splats.from_std430(data), degree


def assert_same(a, b):
    for name in ("position", "log_scale", "rotation", "opacity_logit", "sh"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)


def test_library_matches_the_binding():
    lib = native.library()
    assert lib.splat_native_splat_size() == native.SPLAT_SIZE == splats.SPLAT_DTYPE.itemsize
    assert lib.splat_native_splats_struct_size() == native.ctypes.sizeof(native._Splats)


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize("mixed_types", [False, True])
def test_every_degree_and_type(tmp_path, degree, mixed_types):
    path = str(tmp_path / "a.ply")
    values = random_fixture(path, degree, seed=degree, mixed_types=mixed_types)
    s, read_degree = read_native(path)
    assert read_degree == degree
    assert_same(s, expected_splats(values, degree))
    # The padding after logScale is zero
    assert np.all(np.frombuffer(native.read_ply(path)[0], "<f4").reshape(-1, 60)[:, 7] == 0)


def test_write_ply_round_trip_and_empty(tmp_path):
    rng = np.random.default_rng(3)
    n = 10000  # 248-byte rows: more than two of the reader's 1 MiB chunks
    original = splats.Splats(position=rng.normal(size=(n, 3)), log_scale=rng.normal(size=(n, 3)),
                             rotation=rng.normal(size=(n, 4)), opacity_logit=rng.normal(size=n),
                             sh=rng.normal(size=(n, 16, 3)))
    splats.write_ply(str(tmp_path / "a.ply"), original)
    s, degree = read_native(str(tmp_path / "a.ply"))
    assert degree == 3
    assert_same(s, original)
    empty = splats.Splats(position=np.zeros((0, 3)), log_scale=np.zeros((0, 3)),
                          rotation=np.zeros((0, 4)), opacity_logit=np.zeros(0), sh=np.zeros((0, 16, 3)))
    splats.write_ply(str(tmp_path / "b.ply"), empty)
    assert native.read_ply(str(tmp_path / "b.ply")) == (b"", 3)


def ply_bytes(header_lines, body=b""):
    return ("\n".join(header_lines) + "\n").encode() + body


def float_header(names, count=1, fmt="binary_little_endian 1.0", extra=()):
    return (["ply", f"format {fmt}", f"element vertex {count}"] + [f"property float {n}" for n in names]
            + list(extra) + ["end_header"])


ALL = splats.ply_property_names()
ROW = np.zeros(len(ALL), "<f4").tobytes()

# A malformed file, and what the error must say
INVALID = {
    "not-ply": (b"plx\n", "not a PLY file"),
    "empty": (b"", "not a PLY file"),
    "no-end-header": (b"ply\nformat binary_little_endian 1.0\n", "no end_header"),
    "ascii": (ply_bytes(["ply", "format ascii 1.0", "element vertex 0", "end_header"]), "binary_little_endian"),
    "big-endian": (ply_bytes(float_header(ALL, fmt="binary_big_endian 1.0")) + ROW, "binary_little_endian"),
    "no-vertex": (ply_bytes(["ply", "format binary_little_endian 1.0", "end_header"]), "no vertex element"),
    "face-first": (ply_bytes(["ply", "format binary_little_endian 1.0", "element face 0", "end_header"]),
                   "the first element is face"),
    "bad-count": (ply_bytes(["ply", "format binary_little_endian 1.0", "element vertex -1", "end_header"]),
                  "isn't a count"),
    "missing": (ply_bytes(float_header([n for n in ALL if n not in ("opacity", "rot_2")])),
                "missing properties opacity, rot_2"),
    "list": (ply_bytes(float_header(ALL, extra=["property list uchar int vertex_indices"])),
             "list property vertex_indices"),
    "unknown-type": (ply_bytes(["ply", "format binary_little_endian 1.0", "element vertex 0",
                                "property half x", "end_header"]), "unknown type half"),
    "twice": (ply_bytes(float_header(ALL + ["x"])), "x appears twice"),
    "rest-gap": (ply_bytes(float_header([n for n in ALL if n != "f_rest_7"])), "numbered"),
    "rest-degree": (ply_bytes(float_header(splats.ply_property_names(rest_per_channel=2))),
                    "6 f_rest properties"),
    "truncated": (ply_bytes(float_header(ALL, count=3)) + ROW * 2 + ROW[:10], "3 vertices declared, 2 present"),
    "not-ascii": (b"ply\ncomment \xff\n", "isn't ASCII"),
}


@pytest.mark.parametrize("name", sorted(INVALID))
def test_invalid_files(tmp_path, name):
    content, message = INVALID[name]
    path = tmp_path / f"{name}.ply"
    path.write_bytes(content)
    with pytest.raises(ValueError, match=message) as error:
        native.read_ply(str(path))
    assert str(path) in str(error.value)


def test_unreadable_file(tmp_path):
    with pytest.raises(FileNotFoundError) as error:
        native.read_ply(str(tmp_path / "missing.ply"))
    assert error.value.errno == errno.ENOENT
    with pytest.raises(OSError):
        native.read_ply(str(tmp_path))  # a directory: opening may succeed, reading fails


def test_path_with_nul_is_refused(tmp_path):
    with pytest.raises(ValueError, match="NUL"):
        native.read_ply(str(tmp_path / "a.ply") + "\0b")


def test_many_properties_allocate_what_the_file_holds(tmp_path):
    # 30000 extra double properties (a 0.7 MB header) and one vertex: the reader's buffers are
    # bounded by the file, so neither memory (it took 1 GB) nor time (a quadratic duplicate check)
    # grows with the property count squared or a fixed chunk of vertices
    import resource
    import sys
    import time
    names = ALL + [f"extra_{i}" for i in range(30000)]
    header = (["ply", "format binary_little_endian 1.0", "element vertex 1"]
              + [f"property float {n}" for n in ALL] + [f"property double extra_{i}" for i in range(30000)]
              + ["end_header"])
    path = tmp_path / "wide.ply"
    path.write_bytes(ply_bytes(header, ROW + bytes(8 * (len(names) - len(ALL)))))
    unit = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes on macOS, KiB on Linux
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit
    start = time.monotonic()
    data, degree = native.read_ply(str(path))
    assert time.monotonic() - start < 5
    assert len(data) == native.SPLAT_SIZE and degree == 3
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit - before < 200 << 20



def test_splats_bin_reader(tmp_path):
    rng = np.random.default_rng(5)
    s = splats.Splats(position=rng.normal(size=(7, 3)), log_scale=rng.normal(size=(7, 3)),
                      rotation=rng.normal(size=(7, 4)), opacity_logit=rng.normal(size=7),
                      sh=rng.normal(size=(7, 16, 3)))
    data = splats.to_std430(s)
    (tmp_path / "a.bin").write_bytes(data)
    assert native.read_splats_bin(str(tmp_path / "a.bin")) == (data, 3)
    (tmp_path / "empty.bin").write_bytes(b"")
    assert native.read_splats_bin(str(tmp_path / "empty.bin")) == (b"", 3)
    (tmp_path / "partial.bin").write_bytes(data[:-4])
    with pytest.raises(ValueError, match="1676 bytes isn't a whole number of 240-byte Splats"):
        native.read_splats_bin(str(tmp_path / "partial.bin"))
    with pytest.raises(FileNotFoundError):
        native.read_splats_bin(str(tmp_path / "missing.bin"))
    with pytest.raises(ValueError, match="NUL"):
        native.read_splats_bin(str(tmp_path / "a.bin") + "\0b")
    (tmp_path / "directory.bin").mkdir()
    with pytest.raises((OSError, ValueError), match="not a regular file|directory"):
        native.read_splats_bin(str(tmp_path / "directory.bin"))
