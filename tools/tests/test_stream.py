"""convert.py stream end to end on the fixture (stream_fixture.py): the presets' layouts, a PLY and
its splats.bin giving the same bytes, the printed report, the cameras copy, and the config errors.
The encoding itself is checked in tests/cpp/splat_stream_test.cpp; the binding in test_native.py."""

import json
import os

import numpy as np
import pytest
from conftest import TOOLS

import convert
from splatdata import native, scene, splats, stream
from stream_fixture import fixture_splats, write_fixture

CONFIGS = os.path.join(TOOLS, "stream-configs")
# Each preset's (base, SH) strides at degree 3 (docs/data-format.md)
PRESETS = {"full": (56, 180), "f16": (36, 92), "web": (32, 48)}


def preset(name):
    return os.path.join(CONFIGS, f"{name}.json")


def convert_stream(input_path, out_dir, config, *extra):
    convert.main(["stream", str(input_path), str(out_dir), "--config", str(config), *extra])
    return (out_dir / "scene.stream").read_bytes()


@pytest.fixture
def fixture(tmp_path):
    s = write_fixture(str(tmp_path / "in"), count=40, seed=1)
    return tmp_path / "in", s


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_presets_ply_and_bin_give_the_same_stream(tmp_path, fixture, capsys, name):
    directory, s = fixture
    from_ply = convert_stream(directory / "fixture.ply", tmp_path / "a", preset(name))
    report = capsys.readouterr().out
    from_bin = convert_stream(directory / "splats.bin", tmp_path / "b", preset(name))
    assert from_ply == from_bin
    base, sh = PRESETS[name]
    h = stream.read_header(from_ply)
    assert (int(h["magic"]), int(h["splatCount"]), int(h["shDegree"])) == (1397510227, 40, 3)
    assert (int(h["baseStride"]), int(h["shStride"])) == (base, sh)
    assert len(from_ply) == native.STREAM_HEADER_SIZE + 40 * (base + sh)
    # The header's camera is the default camera of the splats
    assert from_ply[32:144] == native.default_camera(splats.to_std430(s))
    # The report: the layout, then a line per group with its encoding and errors
    lines = report.splitlines()
    assert lines[0].startswith(f"{tmp_path / 'a' / 'scene.stream'}: 40 splats, SH degree 3, {base + sh} B")
    groups = {line.split()[0]: line.split()[1:4] for line in lines[2:]}
    assert list(groups) == list(native.GROUP_NAMES)
    config = json.load(open(preset(name)))
    for g in native.GROUP_NAMES:
        encoding, error, activated = groups[g]
        assert encoding == config[g]["encoding"]
        if encoding == "f32":
            assert float(error) == float(activated) == 0
        else:
            assert float(error) > 0
    assert not os.path.exists(tmp_path / "a" / "cameras.bin")


def test_full_preset_copies_every_value(tmp_path, fixture):
    directory, s = fixture
    out = convert_stream(directory / "splats.bin", tmp_path / "out", preset("full"))
    h = stream.read_header(out)
    f = np.frombuffer(splats.to_std430(s), "<f4").reshape(40, 60)
    base = np.frombuffer(out, "<f4", 40 * 14, int(h["baseOffset"])).reshape(40, 14)
    values = {"position": f[:, 0:3], "scale": f[:, 4:7], "rotation": f[:, 8:12], "opacity": f[:, 3:4],
              "dc": f[:, 12:15]}
    for g, name in enumerate(native.GROUP_NAMES[:5]):
        field = h["fields"][g]
        start = int(field["offset"]) // 4
        got = base[:, start:start + int(field["components"])]
        assert np.array_equal(got.view("<u4"), values[name].view("<u4")), name  # -0.0 too
    sh = np.frombuffer(out, "<f4", 40 * 45, int(h["shOffset"])).reshape(40, 45)
    assert np.array_equal(sh.view("<u4"), f[:, 15:60].view("<u4"))


def write_config(tmp_path, config, name="c.json"):
    path = tmp_path / name
    path.write_text(json.dumps(config) if not isinstance(config, str) else config)
    return path


def web_config():
    return json.load(open(preset("web")))


def test_sh_degree_and_explicit_ranges(tmp_path, fixture, capsys):
    directory, _ = fixture
    config = web_config()
    config["shDegree"] = 1
    config["position"] = {"encoding": "u16", "range": [[-8, 8], [-8, 8], [-1, 65001]]}
    config["sh"] = {"encoding": "u8", "range": [-1.5, 1.5]}
    out = convert_stream(directory / "fixture.ply", tmp_path / "out", write_config(tmp_path, config))
    h = stream.read_header(out)
    assert int(h["shDegree"]) == 1 and int(h["fields"][5]["components"]) == 9
    position = h["fields"][0]
    assert list(position["low"][:3]) == [-8, -8, -1]
    assert np.allclose(position["low"][:3] + position["stepSize"][:3] * 65535, [8, 8, 65001])
    assert float(h["fields"][5]["low"][0]) == -1.5
    report = capsys.readouterr().out
    assert "explicit [-8, 8], [-8, 8], [-1, 6.5e+04]" in report
    assert "explicit [-1.5, 1.5]" in report
    # Degree 0: no SH record
    config["shDegree"] = 0
    out = convert_stream(directory / "fixture.ply", tmp_path / "zero", write_config(tmp_path, config))
    assert int(stream.read_header(out)["shStride"]) == 0


def test_cameras_are_copied(tmp_path, fixture):
    directory, _ = fixture
    cameras = np.zeros(3, scene.CAMERA_DTYPE)
    cameras["size"] = [64, 48]
    (tmp_path / "cameras.bin").write_bytes(cameras.tobytes())
    convert_stream(directory / "fixture.ply", tmp_path / "out", preset("web"), "--cameras",
                   str(tmp_path / "cameras.bin"))
    assert (tmp_path / "out" / "cameras.bin").read_bytes() == cameras.tobytes()
    # A file that isn't Cameras is refused, and nothing is written
    (tmp_path / "bad.bin").write_bytes(b"\0" * 100)
    with pytest.raises(SystemExit, match="100 bytes isn't a whole number of 112-byte Cameras"):
        convert_stream(directory / "fixture.ply", tmp_path / "bad", preset("web"), "--cameras",
                       str(tmp_path / "bad.bin"))
    assert not os.path.exists(tmp_path / "bad")


def drop(config, key):
    del config[key]
    return config


def edit(key, value):
    def apply(config):
        config[key] = value
        return config
    return apply


# A bad config, and what the error must say
BAD_CONFIGS = {
    "not-json": ("{", "not JSON"),
    "not-object": ("[]", "a config must be a JSON object"),
    "missing": (lambda c: drop(c, "sh"), r"missing sh \(every key is required\)"),
    "unknown": (edit("normal", {"encoding": "f32"}), "unknown keys normal"),
    "degree": (edit("shDegree", 4), 'shDegree must be 0 to 3 or "data", not 4'),
    "degree-bool": (edit("shDegree", True), "shDegree must be 0 to 3"),
    "group-not-object": (edit("dc", "f16"), "dc must be an object"),
    "group-unknown": (edit("dc", {"encoding": "f16", "bits": 16}), "dc: unknown keys bits"),
    "encoding": (edit("dc", {"encoding": "f8"}), 'dc.encoding must be one of f32, f16, u16, u8, not "f8"'),
    "no-encoding": (edit("dc", {}), "dc.encoding must be one of"),
    "range-on-float": (edit("dc", {"encoding": "f16", "range": "data"}), "dc: f16 takes no range"),
    "no-range": (edit("opacity", {"encoding": "u8"}), 'opacity: u8 needs a range: "data" or \\[lo, hi\\]'),
    "range-word": (edit("opacity", {"encoding": "u8", "range": "auto"}), 'opacity.range must be "data" or'),
    "range-short": (edit("opacity", {"encoding": "u8", "range": [1]}), r"opacity.range must be \[lo, hi\]"),
    "range-string": (edit("opacity", {"encoding": "u8", "range": [0, "1"]}), r"opacity.range must be \[lo, hi\]"),
    "range-order": (edit("opacity", {"encoding": "u8", "range": [1, 0]}), "has lo above hi"),
    "range-f32": (edit("opacity", {"encoding": "u8", "range": [0, 1e39]}), "isn't finite in float32"),
    "range-huge-int": (edit("opacity", {"encoding": "u8", "range": [0, 10 ** 400]}), "isn't finite in float32"),
    "range-count": (edit("scale", {"encoding": "u8", "range": [[0, 1], [0, 1]]}), "2 \\[lo, hi\\] pairs for 3 components"),
    "range-pair": (edit("scale", {"encoding": "u8", "range": [[0, 1], [0, 1], [2, 1]]}), r"scale.range\[2\]"),
    "sh-per-component": (edit("sh", {"encoding": "u8", "range": [[0, 1]]}), "sh shares one range"),
}


@pytest.mark.parametrize("name", sorted(BAD_CONFIGS))
def test_bad_configs_are_refused(tmp_path, fixture, name):
    directory, _ = fixture
    change, message = BAD_CONFIGS[name]
    config = change if isinstance(change, str) else change(web_config())
    path = write_config(tmp_path, config)
    with pytest.raises(SystemExit, match=message) as error:
        convert_stream(directory / "fixture.ply", tmp_path / "out", path)
    assert str(error.value).startswith(str(path))
    assert not os.path.exists(tmp_path / "out")


def test_bad_inputs_are_refused(tmp_path, fixture):
    directory, _ = fixture
    # More SH bands than the input holds (a degree-1 PLY)
    names = splats.ply_property_names(rest_per_channel=3)
    header = ["ply", "format binary_little_endian 1.0", "element vertex 1"]
    header += [f"property float {n}" for n in names] + ["end_header"]
    row = np.zeros(len(names), "<f4")
    row[names.index("rot_0")] = 1
    (tmp_path / "d1.ply").write_bytes(("\n".join(header) + "\n").encode() + row.tobytes())
    config = web_config()
    config["shDegree"] = 2
    with pytest.raises(SystemExit, match="shDegree 2 is above the input's SH degree, 1"):
        convert_stream(tmp_path / "d1.ply", tmp_path / "out", write_config(tmp_path, config))
    out = convert_stream(tmp_path / "d1.ply", tmp_path / "out", preset("web"))  # "data": degree 1
    assert int(stream.read_header(out)["shDegree"]) == 1
    # The encoder's refusals name the splat
    s = fixture_splats(8, seed=2, zero_rotation=True)
    (tmp_path / "zero.bin").write_bytes(splats.to_std430(s))
    with pytest.raises(SystemExit, match="zero.bin: splat 1's rotation has length 0"):
        convert_stream(tmp_path / "zero.bin", tmp_path / "zero", preset("web"))
    convert_stream(tmp_path / "zero.bin", tmp_path / "zero", preset("full"))  # f32 copies it
    s = fixture_splats(8, seed=2)
    s.sh[5, 7, 1] = np.inf
    (tmp_path / "inf.bin").write_bytes(splats.to_std430(s))
    with pytest.raises(SystemExit, match="splat 5's sh isn't finite"):
        convert_stream(tmp_path / "inf.bin", tmp_path / "inf", preset("full"))
    # Unknown input types and unreadable files
    with pytest.raises(SystemExit, match="must be an INRIA PLY \\(.ply\\) or a splats.bin"):
        convert_stream(tmp_path / "a.txt", tmp_path / "x", preset("full"))
    with pytest.raises(SystemExit, match="missing.ply"):
        convert_stream(tmp_path / "missing.ply", tmp_path / "x", preset("full"))
    with pytest.raises(SystemExit, match="240-byte Splats"):
        (tmp_path / "short.bin").write_bytes(b"\0" * 10)
        convert_stream(tmp_path / "short.bin", tmp_path / "x", preset("full"))
    assert not os.path.exists(tmp_path / "x")


def test_fixture_cli(tmp_path):
    import stream_fixture
    stream_fixture.main([str(tmp_path / "f"), "--count", "5", "--seed", "3"])
    assert native.read_splats_bin(str(tmp_path / "f" / "splats.bin"))[0] == native.read_ply(str(tmp_path / "f" / "fixture.ply"))[0]
    with pytest.raises(ValueError, match="at least 4 splats"):
        fixture_splats(3, seed=0)


def test_explicit_range_overflow_names_the_config(tmp_path, fixture):
    directory, _ = fixture
    config = web_config()
    config["position"] = {"encoding": "u8", "range": [-3e38, 3e38]}
    path = write_config(tmp_path, config)
    with pytest.raises(SystemExit, match="the range of position is too wide") as error:
        convert_stream(directory / "fixture.ply", tmp_path / "out", path)
    assert str(error.value).startswith(str(path))


def test_failed_writes_leave_nothing(tmp_path, fixture, monkeypatch):
    directory, _ = fixture
    (tmp_path / "file").write_bytes(b"")
    with pytest.raises(SystemExit, match="writing the outputs failed"):
        convert_stream(directory / "fixture.ply", tmp_path / "file", preset("web"))
    # The cameras copy fails after the stream is written: neither is left, nor a temporary file
    cameras = np.zeros(1, scene.CAMERA_DTYPE).tobytes()
    (tmp_path / "cameras.bin").write_bytes(cameras)

    def fail(*_):
        raise OSError("disk full")
    monkeypatch.setattr(convert.shutil, "copyfile", fail)
    with pytest.raises(SystemExit, match="writing the outputs failed: disk full"):
        convert_stream(directory / "fixture.ply", tmp_path / "out", preset("web"), "--cameras",
                       str(tmp_path / "cameras.bin"))
    assert os.listdir(tmp_path / "out") == []
