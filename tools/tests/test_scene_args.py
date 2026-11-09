"""scene_args.py: splat_render's and splat_train's arguments from a scene's manifest, splat_view's
from a stream."""

import json
import os
import sys

import pytest

import convert
import make_synthetic as ms
import scene_args
from conftest import REPO
from stream_fixture import write_fixture


@pytest.fixture(scope="module")
def s2(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("s2") / "s2")
    ms.make_s2(out, 0)
    return out


def as_dict(args):
    assert len(args) % 2 == 0
    return dict(zip(args[::2], args[1::2]))


def test_by_index_and_by_name(s2):
    by_index = scene_args.scene_args(s2, "1")
    assert by_index == scene_args.scene_args(s2, "cam_1")
    a = as_dict(by_index)
    assert a == {"--splats": os.path.join(s2, "splats.bin"), "--cameras": os.path.join(s2, "cameras.bin"),
                 "--splat-count": "8", "--sh-degree": "3", "--camera": "1", "--width": "80", "--height": "48"}


@pytest.mark.parametrize("camera", ["2", "-1", "cam_9", "x"])
def test_unknown_camera(s2, camera):
    with pytest.raises(SystemExit):
        scene_args.scene_args(s2, camera)


def test_whitespace_refused(tmp_path):
    out = str(tmp_path / "a scene")
    ms.make_s2(out, 0)
    with pytest.raises(SystemExit, match="whitespace"):
        scene_args.scene_args(out, "0")


def test_golden_scene():
    """The committed S3 scene splat_render's test renders: its cameras 0 and 24 (non-square)."""
    golden = os.path.join(REPO, "programs", "splat_render", "golden", "s3")
    assert as_dict(scene_args.scene_args(golden, "0"))["--width"] == "128"
    a = as_dict(scene_args.scene_args(golden, "cam_24_wide"))
    assert (a["--camera"], a["--width"], a["--height"], a["--splat-count"]) == ("24", "160", "96", "2000")


def test_train_args(tmp_path):
    out = tmp_path / "gt"
    ms.main(["s3", str(out), "--count", "40"])
    with pytest.raises(SystemExit, match="0 image batches"):
        scene_args.train_args(str(out))
    # Give it images: two sizes, so the largest of each is taken
    fake = tmp_path / "fake"
    fake.write_text(f"#!{sys.executable}\nimport sys, numpy as np\nfrom PIL import Image\na = sys.argv[1:]\n"
                    "v = lambda n: a[a.index(n) + 1]\n"
                    "Image.fromarray(np.zeros((int(v('--height')), int(v('--width')), 3), np.uint8)).save(v('--capture'))\n")
    fake.chmod(0o755)
    manifest = ms.render_gt(str(out), str(fake))
    a = as_dict(scene_args.train_args(str(out)))
    pixels = sum(c["width"] * c["height"] for c in manifest["cameras"])
    assert a == {"--splats": os.path.join(out, "splats.bin"), "--cameras": os.path.join(out, "cameras.bin"),
                 "--targets": os.path.join(out, "images_0.bin"), "--splat-count": "40",
                 "--camera-count": str(manifest["cameraCount"]), "--sh-degree": "3",
                 "--max-width": str(max(c["width"] for c in manifest["cameras"])),
                 "--max-height": str(max(c["height"] for c in manifest["cameras"])),
                 "--target-pixels": str(pixels), "--extent": repr(float(manifest["extent"]))}
    assert a["--max-width"] == "160" and a["--max-height"] == "128"


def test_train_args_checks_the_split(tmp_path):
    """A scene from before format version 2, or whose buffer disagrees with its manifest, is refused:
    splat_train would otherwise train on the test cameras."""
    out = tmp_path / "s3"
    ms.main(["s3", str(out), "--count", "40"])
    path = out / "manifest.json"
    manifest = json.loads(path.read_text())
    assert manifest["test"] == [0, 8, 16]
    path.write_text(json.dumps(dict(manifest, version=1)))
    with pytest.raises(SystemExit, match="format version 1.*re-convert"):
        scene_args.train_args(str(out))
    path.write_text(json.dumps(dict(manifest, test=[0, 8])))
    with pytest.raises(SystemExit, match=r"holds out cameras \[0, 8, 16\]"):
        scene_args.train_args(str(out))
    path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="0 image batches"):  # past the split check
        scene_args.train_args(str(out))


def test_view_args(tmp_path):
    """--view: the stream, the header's splat count and the file's size; a short file is refused."""
    write_fixture(str(tmp_path / "fixture"), 40, 7)
    convert.main(["stream", str(tmp_path / "fixture" / "splats.bin"), str(tmp_path / "out"), "--config",
                  os.path.join(REPO, "tools", "stream-configs", "web.json")])
    path = str(tmp_path / "out" / "scene.stream")
    args = scene_args.view_args(path)
    assert args[0] == path
    assert as_dict(args[1:]) == {"--splat-count": "40", "--stream-bytes": str(os.path.getsize(path))}
    short = tmp_path / "short.stream"
    short.write_bytes(b"SPLS\0\0\0\0")
    with pytest.raises(SystemExit, match="too short"):
        scene_args.view_args(str(short))
    with pytest.raises(SystemExit, match="whitespace"):
        scene_args.view_args(str(tmp_path / "a b.stream"))


def test_query(tmp_path):
    """--query: options as name=value, positionals as arg=value, paths as <url dir>/<file name>."""
    golden = os.path.join(REPO, "programs", "splat_render", "golden", "s3")
    assert scene_args.query("/scene", scene_args.scene_args(golden, "0")) == (
        "splats=/scene/splats.bin&cameras=/scene/cameras.bin&splat-count=2000&sh-degree=3&camera=0"
        "&width=128&height=128")
    write_fixture(str(tmp_path / "fixture"), 40, 7)
    convert.main(["stream", str(tmp_path / "fixture" / "splats.bin"), str(tmp_path / "out"), "--config",
                  os.path.join(REPO, "tools", "stream-configs", "web.json")])
    path = str(tmp_path / "out" / "scene.stream")
    assert scene_args.query("/data/", scene_args.view_args(path)) == (
        f"arg=/data/scene.stream&splat-count=40&stream-bytes={os.path.getsize(path)}")
    # A value that isn't a path is quoted, not mapped
    assert scene_args.query("/d", ["--name", "a&b"]) == "name=a%26b"


def test_query_main(capsys):
    golden = os.path.join(REPO, "programs", "splat_render", "golden", "s3")
    scene_args.main(["--query", "/scene", golden, "cam_24_wide"])
    out = capsys.readouterr().out.strip()
    assert out == scene_args.query("/scene", scene_args.scene_args(golden, "24"))
    assert "camera=24&width=160&height=96" in out
    with pytest.raises(SystemExit):
        scene_args.main(["--query", "/scene"])
