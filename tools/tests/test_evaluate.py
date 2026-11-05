"""evaluate.py: scores a PLY's held-out renders against the scene's images, with a fake renderer."""

import json
import math
import os
import re
import sys

import numpy as np
import pytest

import evaluate
import make_synthetic as ms
from splatdata import scene, splats

# Captures camera i as a PNG of its size: red 10 i, green x, blue 200 with --validation (else 0);
# it records the splat count it was given in <capture>.args
FAKE_RENDERER = """\
import sys
import numpy as np
from PIL import Image
args = sys.argv[1:]
value = lambda name: args[args.index(name) + 1]
open(value("--capture") + ".args", "w").write(" ".join(args))
width, height, camera = int(value("--width")), int(value("--height")), int(value("--camera"))
image = np.zeros((height, width, 3), np.uint8)
image[..., 0] = 10 * camera
image[..., 1] = np.arange(width)[None, :] % 256
image[..., 2] = 200 if "--validation" in args else 0
Image.fromarray(image).save(value("--capture"))
"""


@pytest.fixture
def renderer(tmp_path):
    path = tmp_path / "fake_renderer"
    path.write_text(f"#!{sys.executable}\n" + FAKE_RENDERER)
    path.chmod(0o755)
    return str(path)


@pytest.fixture
def gt(tmp_path, renderer):
    """S3 (40 splats) with the fake renderer's validated captures as its images, and a PLY of its
    first 30 splats"""
    out = tmp_path / "gt"
    ms.main(["s3", str(out), "--count", "40"])
    ms.main(["render-gt", str(out), renderer, "--validation"])
    with open(out / "splats.bin", "rb") as f:
        s = splats.from_std430(f.read())
    ply = tmp_path / "trained.ply"
    splats.write_ply(str(ply), splats.Splats(s.position[:30], s.log_scale[:30], s.rotation[:30],
                                             s.opacity_logit[:30], s.sh[:30]))
    return out, str(ply)


def test_equal_renders_score_perfectly(tmp_path, renderer, gt):
    scene_dir, ply = gt
    out = tmp_path / "eval"
    summary = evaluate.main([str(scene_dir), ply, renderer, str(out), "--validation"])
    manifest = json.loads((scene_dir / "manifest.json").read_text())
    assert [v["camera"] for v in summary["views"]] == manifest["test"] == [0, 8, 16]
    assert summary["psnr"] == math.inf and summary["ssim"] == pytest.approx(1)
    assert summary["splats"] == 30
    # The renders read the PLY's splats, not the scene's
    rendered = json.loads((out / "scene" / "manifest.json").read_text())
    assert rendered["splatCount"] == 30 and rendered["files"]["images"] == []
    assert len(splats.from_std430((out / "scene" / "splats.bin").read_bytes())) == 30
    args = (out / "test-8.png.args").read_text().split()
    assert args[args.index("--splat-count") + 1] == "30" and args[args.index("--camera") + 1] == "8"
    assert json.loads((out / "metrics.json").read_text()) == json.loads(json.dumps(summary))


def test_scores_the_difference(tmp_path, renderer, gt):
    scene_dir, ply = gt
    summary = evaluate.main([str(scene_dir), ply, renderer, str(tmp_path / "eval")])
    # Only blue differs, by 200 of 255: the MSE over the three channels is (200/255)^2 / 3
    expected = 10 * math.log10(3 / (200 / 255) ** 2)
    assert [v["psnr"] for v in summary["views"]] == pytest.approx([expected] * 3)
    assert 0 < summary["ssim"] < 1


def test_side_by_side(tmp_path, renderer, gt):
    scene_dir, ply = gt
    out = tmp_path / "eval"
    evaluate.main([str(scene_dir), ply, renderer, str(out), "--validation", "--side-by-side", "8", "24",
                   "--thumbnail", "64"])
    wide = scene.load_rgb(str(out / "side-by-side-24.png"))  # 160x96 -> 64x38, twice
    assert wide.shape == (38, 128, 3)
    square = scene.load_rgb(str(out / "side-by-side-8.png"))
    assert square.shape == (64, 128, 3)
    assert np.array_equal(square[:, :64], square[:, 64:])  # the same image both sides
    assert (out / "camera-24.png").exists()
    with pytest.raises(SystemExit):
        evaluate.main([str(scene_dir), ply, renderer, str(out), "--side-by-side", "99"])


def test_a_failing_renderer_fails(tmp_path, gt):
    scene_dir, ply = gt
    failing = tmp_path / "failing"
    failing.write_text(f"#!{sys.executable}\nprint('no device')\nraise SystemExit(3)\n")
    failing.chmod(0o755)
    with pytest.raises(RuntimeError, match="no device"):
        evaluate.main([str(scene_dir), ply, str(failing), str(tmp_path / "eval")])


def growing_renderer(tmp_path, always, grown=512):
    """The fake renderer, saying a frame overflowed its tile entries (as splat_render does) unless
    started at --initial-entries 512 or more (with `always`, never satisfied)"""
    path = tmp_path / "growing"
    path.write_text(f"#!{sys.executable}\n" + FAKE_RENDERER + f"""\
entries = int(value("--initial-entries")) if "--initial-entries" in args else 100
if entries < 512 or {always}:
    print(f"splat_render: a frame needed 300 tile entries, more than its capacity of {{entries}}; growing it to {grown}")
""")
    path.chmod(0o755)
    return str(path)


def test_an_overflowing_render_is_rendered_again_from_the_grown_capacity(tmp_path, gt):
    scene_dir, ply = gt
    out = tmp_path / "eval"
    evaluate.main([str(scene_dir), ply, growing_renderer(tmp_path, False), str(out)])
    assert "--initial-entries 512" in (out / "test-8.png.args").read_text()


def test_a_render_that_overflows_again_fails(tmp_path, gt):
    scene_dir, ply = gt
    with pytest.raises(RuntimeError, match="overflowed its tile entries even starting at 512"):
        evaluate.main([str(scene_dir), ply, growing_renderer(tmp_path, True), str(tmp_path / "eval")])


def test_growth_to_the_maximum_fails(tmp_path, gt):
    scene_dir, ply = gt
    renderer = growing_renderer(tmp_path, True, grown=33554432)
    with pytest.raises(RuntimeError, match="its maximum"):
        evaluate.main([str(scene_dir), ply, renderer, str(tmp_path / "eval")])


def test_the_growth_pattern_matches_splat_renders_message():
    source = open(os.path.join(os.path.dirname(__file__), "..", "..", "programs", "splat_render",
                               "splat_render.zgal")).read()
    fragments = re.search(r'\(print! ("splat_render: a frame needed ".*?)\)\)', source, re.S).group(1)
    pieces = re.findall(r'"([^"]*)"', fragments)
    assert pieces == ["splat_render: a frame needed ", " tile entries, more than its capacity of ", "; growing it to "]
    assert evaluate.GROWTH.search(f"{pieces[0]}7{pieces[1]}3{pieces[2]}9").group(1) == "9"
    assert str(evaluate.MAX_ENTRIES) == "33554432"
