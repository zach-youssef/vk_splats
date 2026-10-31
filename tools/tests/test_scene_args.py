"""scene_args.py: splat_render's arguments from a scene's manifest."""

import os

import pytest

import make_synthetic as ms
import scene_args
from conftest import REPO


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
