#!/usr/bin/env python3
"""Writes synthetic test scenes in the converter's format (docs/data-format.md), without images.
docs/building-and-testing.md describes each scene and what it's for.

  make_synthetic.py s1 <out dir> [--variant <name>]
      One splat on the axis (or two, or a rotated camera), matching the setups of the closed-form
      oracles' tests (tests/oracle/closed_form.h, tests/cpp/closed_form_test.cpp): one scene per
      variant in <out dir>/<variant>, with the oracle's parameters in the manifest's "source".
  make_synthetic.py s2 <out dir> [--seed <n>]
      5 overlapping anisotropic splats at close depths, 2 off-screen and 1 behind the cameras, with
      all SH bands; 2 cameras.
  make_synthetic.py s3 <out dir> [--seed <n>] [--count <n>]
      About 2k random Gaussians (SH degree 3) in a unit ball, 24 cameras at 128x128 on a sphere
      around it and one at 160x96; every 8th sphere camera held out, the 160x96 one always
      trains.
  make_synthetic.py perturb <scene dir> <out dir> [--seed <n>] [--jitter <fraction>]
      The scene with degraded splats: positions jittered, scales doubled or halved, random colours
      and opacities.
  make_synthetic.py sparse <scene dir> <out dir> [--seed <n>] [--fraction <f>]
      The scene with its splats replaced by ones initialized from a random subset of their
      positions, as the converter initializes from COLMAP points.

Every scene is a pure function of its arguments: the same seed writes the same bytes, with the
same numpy version on the same platform (docs/building-and-testing.md).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from splatdata import quaternion, scene, splats as splat_module  # noqa: E402

GENERATOR = "make_synthetic.py"


# --- Building blocks ---------------------------------------------------------------------------------

def quat_about_z(theta: float) -> List[float]:
    """The unit quaternion (w, x, y, z) of a rotation by theta about z (closed_form.h's
    quat_about_z)."""
    return [math.cos(0.5 * theta), 0.0, 0.0, math.sin(0.5 * theta)]


def look_at(name: str, width: int, height: int, f: float, center, target, down) -> scene.PinholeCamera:
    """A pinhole camera at center looking at target, with the image's +y as close to down as the
    view allows (COLMAP's convention: +x right, +y down, +z forward), principal point at the image
    centre."""
    center, target, down = (np.asarray(v, dtype=np.float64) for v in (center, target, down))
    forward = target - center
    forward /= np.linalg.norm(forward)
    right = np.cross(down, forward)
    if np.linalg.norm(right) < 1e-6:
        raise ValueError(f"{name}: the view direction is parallel to down")
    right /= np.linalg.norm(right)
    rotation = np.stack([right, np.cross(forward, right), forward])  # rows: the camera's axes
    return scene.PinholeCamera(name, width, height, f, f, 0.5 * width, 0.5 * height, rotation,
                               -rotation @ center)


def random_unit_quaternions(rng: np.random.Generator, n: int) -> np.ndarray:
    """Uniformly distributed rotations (normalized Gaussian 4-vectors), w >= 0."""
    q = quaternion.normalize(rng.normal(size=(n, 4)))
    return np.where(q[:, :1] < 0, -q, q)


def colour_of_dc(dc: np.ndarray) -> np.ndarray:
    """The colour of a DC coefficient, max(C0·dc + 0.5, 0) (the kernels' degree-0 colour)."""
    return np.maximum(splat_module.SH_C0 * np.asarray(dc, dtype=np.float64) + 0.5, 0.0)


def write(out_dir: str, splats: splat_module.Splats, cameras: Sequence[scene.PinholeCamera],
          source: dict, held_out: Optional[np.ndarray] = None, extent: Optional[float] = None) -> dict:
    """scene.write_scene, with the generator in the source and, if given, extent in place of the
    one computed from the camera centres."""
    manifest = scene.write_scene(out_dir, splats, cameras, held_out=held_out,
                                 source={"generator": GENERATOR, **source})
    if extent is not None:
        manifest["extent"] = extent
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)
            f.write("\n")
    return manifest


# --- S1: the closed-form oracles' setups -------------------------------------------------------------
# Each variant is one setup of tests/cpp/closed_form_test.cpp, with the same numbers, so a kernel
# test reads the scene, renders it and compares with the oracle on the parameters recorded in the
# manifest. The cameras sit at the origin, rotated only about z (R = Rz(phi), t = 0), so the
# oracle's canonical camera applies directly (phi = 0) or through world_to_camera_about_z(phi).

@dataclass
class OracleSplat:
    """closed_form.h's Splat: mean (x, y, z), log-scales, rotation theta about z, opacity logit,
    DC coefficients."""
    x: float
    y: float
    z: float
    scale: Sequence[float]  # s = exp(log-scale), as the tests write them
    theta: float = 0.0
    opacity_logit: float = 0.0
    dc: Sequence[float] = (1.0, 0.0, -1.0)  # the tests' on_axis() default

    def to_json(self) -> dict:
        return {"x": self.x, "y": self.y, "z": self.z, "logScale": [math.log(s) for s in self.scale],
                "theta": self.theta, "opacityLogit": self.opacity_logit, "dc": list(self.dc)}


@dataclass
class S1Variant:
    f: float
    width: int
    height: int
    splats: List[OracleSplat]
    background: Sequence[float] = (0.0, 0.0, 0.0)  # black where the test sets none
    target: Optional[Sequence[float]] = None  # the gradient tests' constant L2 target
    phis: Sequence[float] = (0.0,)  # one camera per rotation about z
    note: str = ""


def on_axis(z, sx, sy, sz, theta=0.0, opacity=0.0, dc=(1.0, 0.0, -1.0)) -> OracleSplat:
    return OracleSplat(0.0, 0.0, z, (sx, sy, sz), theta, opacity, dc)


S1_VARIANTS: Dict[str, S1Variant] = {
    # "2D covariance: closed-form cases": isotropic on the axis, (f s / z)^2 + 0.3 for any rotation
    "isotropic": S1Variant(50.0, 33, 33, [on_axis(4.0, 0.2, 0.2, 0.2, theta=0.3)],
                           note="isotropic on the axis: a = c = (f s / z)^2 + 0.3, b = 0, for any theta"),
    # ... anisotropic and axis-aligned, its 90-degree rotation (axes swap) and 45 degrees (b != 0)
    "anisotropic": S1Variant(50.0, 33, 33, [on_axis(4.0, 0.3, 0.1, 0.5)],
                             note="axis-aligned: a = (f sx / z)^2 + 0.3, c = (f sy / z)^2 + 0.3"),
    "anisotropic-rot90": S1Variant(50.0, 33, 33, [on_axis(4.0, 0.3, 0.1, 0.5, theta=math.pi / 2)],
                                   note="rotated 90 degrees about z: a and c of 'anisotropic' swap"),
    "anisotropic-rot45": S1Variant(50.0, 33, 33, [on_axis(4.0, 0.3, 0.1, 0.5, theta=math.pi / 4)],
                                   note="rotated 45 degrees: same trace and determinant, b = (a0 - c0) / 2"),
    # ... matches Eqs. 5, 6 and 10 off the axis (one of the test's grid points)
    "off-axis": S1Variant(50.0, 33, 33, [OracleSplat(0.4, -0.5, 4.0, (0.25, 0.08, 0.15), theta=0.9)],
                          note="off the axis, where J's third column and sz matter"),
    # "one splat: per-pixel value"; the odd width puts pixel (16, 16)'s centre on the principal point
    "one-splat": S1Variant(50.0, 33, 33, [on_axis(4.0, 0.3, 0.1, 0.2, opacity=0.5)], background=(0.1, 0.2, 0.3),
                           note="per-pixel value c alpha + bg (1 - alpha), 1/255 cutoff"),
    "one-splat-clamped": S1Variant(50.0, 33, 33, [on_axis(4.0, 0.3, 0.1, 0.2, opacity=12.0)],
                                   background=(0.1, 0.2, 0.3), note="alpha clamped at 0.99 near the mean"),
    # "single-splat L2 gradients": kGradCam, grad_splat(), kBg, kTarget; every pixel blended, none clamped
    "gradient": S1Variant(20.0, 15, 15,
                          [OracleSplat(0.15, -0.1, 2.0, (0.45, 0.35, 0.3), theta=0.4, opacity_logit=0.8,
                                       dc=(0.8, -0.6, 0.2))],
                          background=(0.1, 0.2, 0.3), target=(0.7, 0.4, 0.05),
                          note="L2 gradients against target; footprint covers the image"),
    # "two splats on the axis composite in depth order"; stored back first, so the kernels must sort
    "two-splat": S1Variant(50.0, 33, 33,
                           [on_axis(6.0, 0.5, 0.3, 0.2, opacity=1.2, dc=(-1.0, 1.5, 0.5)),
                            on_axis(3.0, 0.2, 0.2, 0.2, opacity=0.4, dc=(1.5, -1.0, 0.0))],
                           background=(0.0, 0.1, 0.2), note="c1 a1 + c2 a2 (1 - a1) + bg (1 - a1)(1 - a2), "
                                                            "the nearer splat (stored second) first"),
    # "a camera rotated about z": the world splat seen by cameras Rz(phi); the oracle's splat is
    # world_to_camera_about_z(splat, phi). Anisotropic, so W against W^T shows.
    "rotated-camera": S1Variant(50.0, 33, 33,
                                [OracleSplat(0.3, -0.2, 4.0, (0.3, 0.08, 0.15), theta=0.7, dc=(0.0, 0.0, 0.0))],
                                phis=(0.4, -1.1, 2.5), note="camera k is Rz(phi_k); compare with "
                                                            "world_to_camera_about_z(splat, phi_k)"),
    "rotated-camera-aligned": S1Variant(50.0, 33, 33,
                                        [OracleSplat(0.3, -0.2, 4.0, (0.3, 0.08, 0.15), dc=(0.0, 0.0, 0.0))],
                                        phis=(0.4, -1.1, 2.5), note="as rotated-camera with theta = 0"),
}


def s1_scene(name: str):
    """The splats, cameras and manifest source of one S1 variant."""
    v = S1_VARIANTS[name]
    n = len(v.splats)
    sh = np.zeros((n, splat_module.SH_COEFFICIENTS, 3))
    sh[:, 0, :] = [s.dc for s in v.splats]
    splats = splat_module.Splats(
        position=[[s.x, s.y, s.z] for s in v.splats],
        log_scale=[[math.log(c) for c in s.scale] for s in v.splats],
        rotation=[quat_about_z(s.theta) for s in v.splats],
        opacity_logit=[s.opacity_logit for s in v.splats],
        sh=sh)
    cameras = [scene.PinholeCamera(f"cam_{k}", v.width, v.height, v.f, v.f, 0.5 * v.width, 0.5 * v.height,
                                   quaternion.to_matrix(quat_about_z(phi)), np.zeros(3))
               for k, phi in enumerate(v.phis)]
    oracle = {"f": v.f, "width": v.width, "height": v.height, "background": list(v.background),
              "cameraPhi": list(v.phis), "splats": [s.to_json() for s in v.splats], "note": v.note}
    if v.target is not None:
        oracle["target"] = list(v.target)
    return splats, cameras, {"scene": "s1", "variant": name, "oracle": oracle}


def make_s1(out_dir: str, names: Sequence[str]) -> List[dict]:
    manifests = []
    for name in names:
        splats, cameras, source = s1_scene(name)
        # The cameras all sit at the origin, so the camera-based extent would be 0, and a trainer
        # scaling its position learning rate by it would never move a splat
        extent = 1.1 * float(np.max(np.linalg.norm(splats.position.astype(np.float64), axis=1)))
        manifests.append(write(os.path.join(out_dir, name), splats, cameras, source, extent=extent))
    return manifests


# --- S2: a handful of overlapping splats ---------------------------------------------------------------
# For the tile pipeline's exact tile ranges and the finite-difference tests: few splats, every SH
# band nonzero, general rotations, and the cases the preprocess must cull or clip. Positions are
# fixed (so the cases hold by construction); rotations, scales, opacities and SH come from the seed.

S2_WIDTH, S2_HEIGHT, S2_F = 64, 64, 60.0
S2_OVERLAPPING = 5
# Camera-0 space (the world): the overlapping splats sit near the axis at depths 4.00-4.08, a
# 0.5% spacing that the depth key must still order
S2_POSITIONS = np.array([
    [0.00, 0.00, 4.00], [0.15, 0.05, 4.02], [-0.10, 0.12, 4.04], [0.05, -0.15, 4.06], [-0.08, -0.05, 4.08],
    [2.32, 0.0, 4.0],   # off-screen: mean at x = 66.8 px (past the right edge), footprint reaching in
    [3.0, 0.0, 4.0],    # off-screen: mean at x = 77 px, footprint entirely outside
    [0.0, 0.0, -2.0],   # behind both cameras
])
# What each splat is for, as seen by cam_0 (cam_1 sees the overlapping ones too, and the last behind it)
S2_CASES = ["overlapping"] * S2_OVERLAPPING + ["off-screen-partial", "off-screen", "behind"]


def s2_scene(seed: int):
    rng = np.random.default_rng(seed)
    n = len(S2_POSITIONS)
    log_scale = np.log(rng.uniform(0.08, 0.35, size=(n, 3)))
    log_scale[5] = np.log([0.3, 0.3, 0.3])   # mean 2.8 px past the edge, radius 16 px: reaches ~13 px in
    log_scale[6] = np.log([0.05, 0.05, 0.05])  # radius 4 px, its mean 13 px past the edge
    # Unnormalized quaternions: the kernels normalize
    rotation = random_unit_quaternions(rng, n) * rng.uniform(0.5, 2.0, size=(n, 1))
    opacity = splat_module.logit(rng.uniform(0.3, 0.9, size=n))
    sh = np.zeros((n, splat_module.SH_COEFFICIENTS, 3))
    sh[:, 0, :] = splat_module.rgb_to_sh_dc(rng.uniform(0.15, 0.85, size=(n, 3)))
    sh[:, 1:, :] = rng.normal(0.0, 0.15, size=(n, splat_module.SH_COEFFICIENTS - 1, 3))
    splats = splat_module.Splats(position=S2_POSITIONS, log_scale=log_scale, rotation=rotation,
                                 opacity_logit=opacity, sh=sh)
    cameras = [
        look_at("cam_0", S2_WIDTH, S2_HEIGHT, S2_F, [0, 0, 0], [0, 0, 1], [0, 1, 0]),  # R = I, t = 0
        # Off the axis, looking at the cluster, non-square: a general rotation and translation
        look_at("cam_1", 80, 48, 70.0, [1.0, -0.3, 0.0], [0.0, 0.0, 4.04], [0, 1, 0]),
    ]
    return splats, cameras, {"scene": "s2", "seed": seed, "cases": S2_CASES}


def make_s2(out_dir: str, seed: int) -> dict:
    splats, cameras, source = s2_scene(seed)
    return write(out_dir, splats, cameras, source)


# --- S3: a random scene ------------------------------------------------------------------------------------
# For tile ranges at scale, the regression capture and re-fitting: Gaussians in a unit ball at the
# origin, seen from a sphere of radius 4 by cameras whose view holds the whole ball.

S3_SIZE = 128
S3_WIDE = (160, 96)  # the non-square camera
S3_F = 140.0  # half field of view atan(64/140) = 24.6 degrees (18.9 vertically at 96 px), over the ball's 14.5
S3_CAMERA_RADIUS = 4.0
S3_CAMERAS = 24
S3_DOWN = (0.0, 0.0, -1.0)  # world +z is up


def fibonacci_sphere(n: int) -> np.ndarray:
    """n nearly evenly spread unit vectors (none at a pole)."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = math.pi * (3.0 - math.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def s3_scene(seed: int, count: int):
    rng = np.random.default_rng(seed)
    # Uniform in the unit ball
    direction = quaternion.normalize(rng.normal(size=(count, 3)))
    position = direction * rng.uniform(0.0, 1.0, size=(count, 1)) ** (1.0 / 3.0)
    # About the spacing of count points in the ball ((4.19 / 2000)^(1/3) = 0.13), anisotropic
    log_scale = rng.uniform(math.log(0.01), math.log(0.06), size=(count, 3))
    rotation = random_unit_quaternions(rng, count) * rng.uniform(0.5, 2.0, size=(count, 1))
    opacity = splat_module.logit(rng.uniform(0.3, 0.95, size=count))
    sh = np.zeros((count, splat_module.SH_COEFFICIENTS, 3))
    sh[:, 0, :] = splat_module.rgb_to_sh_dc(rng.uniform(0.1, 0.9, size=(count, 3)))
    # Higher bands small enough that colours stay mostly within [0, 1]
    sh[:, 1:, :] = rng.normal(0.0, 0.1, size=(count, splat_module.SH_COEFFICIENTS - 1, 3))
    splats = splat_module.Splats(position=position, log_scale=log_scale, rotation=rotation,
                                 opacity_logit=opacity, sh=sh)
    cameras = [look_at(f"cam_{k:02}", S3_SIZE, S3_SIZE, S3_F, S3_CAMERA_RADIUS * d, [0, 0, 0], S3_DOWN)
               for k, d in enumerate(fibonacci_sphere(S3_CAMERAS))]
    wide_dir = quaternion.normalize(np.array([1.0, 1.0, 0.3]))
    cameras.append(look_at(f"cam_{S3_CAMERAS:02}_wide", *S3_WIDE, S3_F, S3_CAMERA_RADIUS * wide_dir, [0, 0, 0],
                           S3_DOWN))
    # Every 8th sphere camera is held out; the non-square camera always trains, so re-fitting runs
    # the backward pass on a W != H image
    held_out = np.append(scene.test_split([c.name for c in cameras[:-1]]), False)
    return splats, cameras, held_out, {"scene": "s3", "seed": seed, "count": count}


def make_s3(out_dir: str, seed: int, count: int) -> dict:
    splats, cameras, held_out, source = s3_scene(seed, count)
    return write(out_dir, splats, cameras, source, held_out)


# --- Degraded starting points ---------------------------------------------------------------------------
# Both keep the scene's cameras, images and split, copying every file but splats.bin, and record
# what they did in the manifest's source.

def read_splats(scene_dir: str):
    with open(os.path.join(scene_dir, "manifest.json")) as f:
        manifest = json.load(f)
    with open(os.path.join(scene_dir, manifest["files"]["splats"]), "rb") as f:
        return manifest, splat_module.from_std430(f.read())


def write_degraded(scene_dir: str, out_dir: str, manifest: dict, splats: splat_module.Splats, how: dict) -> dict:
    scene_dir, out_dir = os.path.realpath(scene_dir), os.path.realpath(out_dir)
    if os.path.commonpath([scene_dir, out_dir]) == scene_dir:
        raise ValueError(f"{out_dir}: the output directory can't be the scene's or inside it")
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise ValueError(f"{out_dir}: the output directory isn't empty (its files would mix with the scene's)")
    shutil.copytree(scene_dir, out_dir, dirs_exist_ok=True)
    with open(os.path.join(out_dir, manifest["files"]["splats"]), "wb") as f:
        f.write(splat_module.to_std430(splats))
    manifest = dict(manifest, splatCount=len(splats), source={**how, "from": manifest.get("source", {})})
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    return manifest


def perturbed(splats: splat_module.Splats, seed: int, jitter: float) -> splat_module.Splats:
    """A degraded copy: positions plus Gaussian noise of sigma jitter times the RMS distance of the
    positions from their centroid, every splat's scale doubled or halved (on all axes, at random),
    DC colours uniform in [0, 1] with the higher SH bands zeroed, opacities uniform in [0.05, 0.95];
    rotations kept."""
    rng = np.random.default_rng(seed)
    n = len(splats)
    position = splats.position.astype(np.float64)
    spread = math.sqrt(np.mean(np.sum((position - position.mean(0)) ** 2, axis=1))) if n else 0.0
    sh = np.zeros((n, splat_module.SH_COEFFICIENTS, 3))
    sh[:, 0, :] = splat_module.rgb_to_sh_dc(rng.uniform(0.0, 1.0, size=(n, 3)))
    return splat_module.Splats(
        position=position + rng.normal(0.0, jitter * spread, size=(n, 3)),
        log_scale=splats.log_scale + rng.choice([-math.log(2.0), math.log(2.0)], size=(n, 1)),
        rotation=splats.rotation,
        opacity_logit=splat_module.logit(rng.uniform(0.05, 0.95, size=n)),
        sh=sh)


def sparse(splats: splat_module.Splats, seed: int, fraction: float) -> splat_module.Splats:
    """Splats initialized as the converter initializes COLMAP points (splats.from_points) from a
    random subset of round(fraction·N) positions (at least 4, for the nearest-neighbour scale),
    kept in their original order, coloured with the splats' DC colours clamped to [0, 1]."""
    rng = np.random.default_rng(seed)
    n = len(splats)
    keep = max(4, int(round(fraction * n)))
    if keep > n:
        raise ValueError(f"the scene has {n} splats; a sparse subset needs at least 4")
    chosen = np.sort(rng.choice(n, size=keep, replace=False))
    rgb = np.clip(colour_of_dc(splats.sh[chosen, 0, :]), 0.0, 1.0)
    return splat_module.from_points(splats.position[chosen].astype(np.float64), rgb)


def make_perturb(scene_dir: str, out_dir: str, seed: int, jitter: float) -> dict:
    manifest, splats = read_splats(scene_dir)
    return write_degraded(scene_dir, out_dir, manifest, perturbed(splats, seed, jitter),
                          {"generator": GENERATOR, "perturb": {"seed": seed, "jitter": jitter}})


def make_sparse(scene_dir: str, out_dir: str, seed: int, fraction: float) -> dict:
    manifest, splats = read_splats(scene_dir)
    return write_degraded(scene_dir, out_dir, manifest, sparse(splats, seed, fraction),
                          {"generator": GENERATOR, "sparse": {"seed": seed, "fraction": fraction}})


# --- Command line --------------------------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    c = commands.add_parser("s1", help="the closed-form oracles' one- and two-splat setups")
    c.add_argument("out", help="output directory; each variant goes in <out>/<variant>")
    c.add_argument("--variant", choices=sorted(S1_VARIANTS), help="only this variant (default: all)")
    c.set_defaults(run=lambda a: make_s1(a.out, [a.variant] if a.variant else list(S1_VARIANTS)))

    c = commands.add_parser("s2", help="a few overlapping, off-screen and behind-the-camera splats")
    c.add_argument("out")
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(run=lambda a: make_s2(a.out, a.seed))

    c = commands.add_parser("s3", help="random Gaussians seen by cameras on a sphere")
    c.add_argument("out")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--count", type=int, default=2000, help="number of Gaussians (default 2000)")
    c.set_defaults(run=lambda a: make_s3(a.out, a.seed, a.count))

    c = commands.add_parser("perturb", help="a scene with degraded splats, as a training start")
    c.add_argument("scene")
    c.add_argument("out")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--jitter", type=float, default=0.05,
                   help="position noise, relative to the positions' RMS spread (default 0.05)")
    c.set_defaults(run=lambda a: make_perturb(a.scene, a.out, a.seed, a.jitter))

    c = commands.add_parser("sparse", help="a scene initialized from a subset of its splats' positions")
    c.add_argument("scene")
    c.add_argument("out")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--fraction", type=float, default=0.1, help="share of positions kept (default 0.1)")
    c.set_defaults(run=lambda a: make_sparse(a.scene, a.out, a.seed, a.fraction))

    args = parser.parse_args(argv)
    if getattr(args, "count", 4) < 4:
        parser.error("--count must be at least 4")
    if getattr(args, "jitter", 0.0) < 0:
        parser.error("--jitter must not be negative")
    if not 0 < getattr(args, "fraction", 1.0) <= 1:
        parser.error("--fraction must be in (0, 1]")
    args.run(args)


if __name__ == "__main__":
    main()
