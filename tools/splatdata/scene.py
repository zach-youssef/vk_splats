"""Scenes as the programs read them: pinhole cameras, undistorted and resized images, the
train/test split, and the output directory (buffer-files and manifest.json; docs/data-format.md).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy import ndimage

from . import colmap, quaternion, splats as splat_module

FORMAT_VERSION = 1

# Every 8th image, in name order, is held out for testing (the reference code's llffhold)
TEST_EVERY = 8

# No buffer binds more than this (lavapipe's maxStorageBufferRange, WebGPU's default
# maxStorageBufferBindingSize), so the packed images are split into files of at most this size
MAX_BUFFER_BYTES = 128 * 1024 * 1024

NO_IMAGE = 0xFFFFFFFF


@dataclass
class PinholeCamera:
    """An undistorted camera: COLMAP's convention (world to camera, +z forward, +y down, the
    top-left pixel's centre at (0.5, 0.5))."""
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    rotation: np.ndarray  # (3, 3), world to camera
    translation: np.ndarray  # (3,)
    image: Optional[np.ndarray] = field(default=None, repr=False)  # (height, width, 3) uint8

    @property
    def center(self) -> np.ndarray:
        return -np.einsum("ji,j->i", self.rotation, self.translation)

    def world_to_camera_matrix(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.rotation
        m[:3, 3] = self.translation
        return m

    def project(self, points: np.ndarray) -> np.ndarray:
        p = np.einsum("ij,nj->ni", self.rotation, points) + self.translation
        return np.stack([self.fx * p[:, 0] / p[:, 2] + self.cx, self.fy * p[:, 1] / p[:, 2] + self.cy], 1)


# --- Undistortion and resizing ----------------------------------------------------------------------

def undistorted_focal_scale(camera: colmap.Camera) -> float:
    """The smallest factor s such that a pinhole camera of the same size and principal point with
    focal lengths s·(fx, fy) sees only pixels inside the distorted image: every border pixel of
    the undistorted image maps inside the source's pixel centres, and the distortion doesn't fold
    over the view (its Jacobian stays positive, so no two output pixels show the same place). 1
    without distortion. Raises ValueError when no s in [0.25, 4] works."""
    if not camera.has_distortion:
        return 1.0
    w, h = camera.width, camera.height
    xs, ys = np.arange(w) + 0.5, np.arange(h) + 0.5
    border = np.concatenate([np.stack([xs, np.full(w, 0.5)], 1), np.stack([xs, np.full(w, h - 0.5)], 1),
                             np.stack([np.full(h, 0.5), ys], 1), np.stack([np.full(h, w - 0.5), ys], 1)])
    # Where folding is checked: a grid over the view, its edges and corners included. Usually the
    # Jacobian is smallest at the corners, but a radial lens with k1 < 0 < k2 can fold in a narrow
    # ring inside the view, which the grid can step over
    gx, gy = np.meshgrid(np.linspace(0.5, w - 0.5, 65), np.linspace(0.5, h - 0.5, 65))
    grid = np.stack([gx, gy], -1).reshape(-1, 2)

    def inside(s):
        src = _source_pixels(camera, border, s)
        return bool(np.all((src >= 0.5 - 1e-9) & (src <= np.array([w, h]) - 0.5 + 1e-9)))

    # Both predicates only get easier as s grows (the view shrinks to a subset of itself), so
    # their conjunction can be bisected; inside() alone isn't monotone once the map folds, since
    # pixels past the fold come back inside
    def valid(s):
        return not folds(camera, grid, s) and inside(s)

    lo, hi = 0.25, 4.0
    if folds(camera, grid, hi):
        raise ValueError(f"camera {camera.id}: the distortion folds over the image (not monotonic over its "
                         f"radius), so it can't be undistorted")
    if not inside(hi):
        raise ValueError(f"camera {camera.id}: distortion too strong to undistort")
    if valid(lo):
        return lo
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if valid(mid) else (mid, hi)
    return hi


def folds(camera: colmap.Camera, pixels: np.ndarray, scale: float) -> bool:
    """Whether the distortion folds at any of the undistorted pixels (focal lengths scaled by
    scale): the Jacobian of distort() there isn't positive."""
    (fx, fy), (cx, cy) = camera.focal, camera.principal_point
    n = np.stack([(pixels[..., 0] - cx) / (fx * scale), (pixels[..., 1] - cy) / (fy * scale)], -1)
    eps = 1e-6
    ju = (camera.distort(n + [eps, 0]) - camera.distort(n - [eps, 0])) / (2 * eps)
    jv = (camera.distort(n + [0, eps]) - camera.distort(n - [0, eps])) / (2 * eps)
    return bool(np.any(ju[..., 0] * jv[..., 1] - jv[..., 0] * ju[..., 1] <= 0))


def _source_pixels(camera: colmap.Camera, pixels: np.ndarray, scale: float) -> np.ndarray:
    """Where undistorted pixels (with focal lengths scaled by scale) come from in the distorted
    image."""
    (fx, fy), (cx, cy) = camera.focal, camera.principal_point
    normalized = np.stack([(pixels[..., 0] - cx) / (fx * scale), (pixels[..., 1] - cy) / (fy * scale)], -1)
    return camera.to_pixels(normalized)


def undistort_image(camera: colmap.Camera, image: np.ndarray, scale: float) -> np.ndarray:
    """Resamples a distorted image (H, W, C) to the pinhole camera with focal lengths scaled by
    scale (bilinear; copied unchanged without distortion)."""
    if not camera.has_distortion and scale == 1.0:
        return image.copy()
    h, w = image.shape[:2]
    if (w, h) != (camera.width, camera.height):
        raise ValueError(f"image is {w}x{h}, camera {camera.id} is {camera.width}x{camera.height}")
    ys, xs = np.mgrid[0:h, 0:w]
    src = _source_pixels(camera, np.stack([xs + 0.5, ys + 0.5], -1).astype(np.float64), scale)
    # map_coordinates indexes pixels by their centres at integers
    coords = [src[..., 1] - 0.5, src[..., 0] - 0.5]
    out = np.empty_like(image)
    for c in range(image.shape[2]):
        channel = ndimage.map_coordinates(image[..., c].astype(np.float64), coords, order=1, mode="nearest")
        out[..., c] = np.clip(np.rint(channel), 0, 255).astype(image.dtype)
    return out


def resized_size(width: int, height: int, max_size: Optional[int]):
    """The size after scaling the longer side down to max_size (aspect ratio kept, rounded)."""
    if max_size is None or max(width, height) <= max_size:
        return width, height
    s = max_size / max(width, height)
    return max(1, round(width * s)), max(1, round(height * s))


def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resizes with Pillow's Lanczos filter, which maps the image's extent onto the new one, so
    intrinsics scale by width'/width and height'/height exactly."""
    from PIL import Image as PILImage
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return np.asarray(PILImage.fromarray(image).resize((width, height), PILImage.LANCZOS))


def load_rgb(path: str) -> np.ndarray:
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        return np.asarray(im.convert("RGB"))


def save_png(path: str, image: np.ndarray) -> None:
    from PIL import Image as PILImage
    PILImage.fromarray(image).save(path)


def pinhole_from_colmap(model: colmap.Model, image: colmap.Image, pixels: Optional[np.ndarray] = None,
                        max_size: Optional[int] = None) -> PinholeCamera:
    """The undistorted, resized pinhole camera of a COLMAP image, with its image if pixels (the
    distorted image, as COLMAP saw it) are given."""
    camera = model.cameras[image.camera_id]
    if pixels is not None and pixels.shape[:2] != (camera.height, camera.width):
        raise ValueError(f"{image.name}: image is {pixels.shape[1]}x{pixels.shape[0]}, camera {camera.id} is "
                         f"{camera.width}x{camera.height}")
    scale = undistorted_focal_scale(camera)
    (fx, fy), (cx, cy) = camera.focal, camera.principal_point
    width, height = resized_size(camera.width, camera.height, max_size)
    sx, sy = width / camera.width, height / camera.height
    undistorted = None
    if pixels is not None:
        undistorted = resize_image(undistort_image(camera, pixels, scale), width, height)
    return PinholeCamera(name=image.name, width=width, height=height,
                         fx=fx * scale * sx, fy=fy * scale * sy, cx=cx * sx, cy=cy * sy,
                         rotation=quaternion.to_matrix(image.qvec), translation=np.array(image.tvec),
                         image=undistorted)


# --- Split and extent -------------------------------------------------------------------------------

def test_split(names: Sequence[str]) -> np.ndarray:
    """Whether each image is held out: every TEST_EVERY-th in name order, starting with the first."""
    order = sorted(range(len(names)), key=lambda i: names[i])
    held_out = np.zeros(len(names), dtype=bool)
    held_out[order[::TEST_EVERY]] = True
    return held_out


def camera_extent(cameras: Sequence[PinholeCamera]) -> float:
    """The scene extent the reference code scales the position learning rate by: 1.1 times the
    largest distance from a camera centre to their mean (getNerfppNorm). The reference passes
    the training cameras only."""
    centers = np.array([c.center for c in cameras])
    return float(1.1 * np.max(np.linalg.norm(centers - centers.mean(0), axis=1)))


# --- std430 buffers -----------------------------------------------------------------------------------
# (struct Camera (worldToCamera mat4) (intrinsics vec4) (center vec3) (pixelOffset uint)
#                (size uvec2) (imageBatch uint))
# Offsets: worldToCamera 0 (column-major), intrinsics 64 (fx, fy, cx, cy), center 80,
# pixelOffset 92, size 96, imageBatch 104, then 4 bytes padding; 112 bytes.
CAMERA_DTYPE = np.dtype({
    "names": ["worldToCamera", "intrinsics", "center", "pixelOffset", "size", "imageBatch"],
    "formats": [("<f4", 16), ("<f4", 4), ("<f4", 3), "<u4", ("<u4", 2), "<u4"],
    "offsets": [0, 64, 80, 92, 96, 104],
    "itemsize": 112,
})


def pack_images(cameras: Sequence[PinholeCamera], max_bytes: int = MAX_BUFFER_BYTES):
    """Packs the cameras' images into RGBA8 buffers (one little-endian uint per pixel: R in the
    low byte, alpha 255; rows top to bottom), each at most max_bytes, filled in camera order.
    Returns the buffers' bytes and each camera's (batch, pixel offset), (NO_IMAGE, 0) without an
    image."""
    batches: List[List[bytes]] = []
    sizes: List[int] = []
    placement = []
    for camera in cameras:
        if camera.image is None:
            placement.append((NO_IMAGE, 0))
            continue
        if camera.image.shape != (camera.height, camera.width, 3) or camera.image.dtype != np.uint8:
            raise ValueError(f"{camera.name}: image is {camera.image.shape} {camera.image.dtype}, expected "
                             f"({camera.height}, {camera.width}, 3) uint8")
        rgba = np.concatenate([camera.image, np.full((camera.height, camera.width, 1), 255, np.uint8)], 2)
        data = rgba.tobytes()
        if len(data) > max_bytes:
            raise ValueError(f"{camera.name}: one image is {len(data)} bytes, over the {max_bytes}-byte limit")
        if not batches or sizes[-1] + len(data) > max_bytes:
            batches.append([])
            sizes.append(0)
        placement.append((len(batches) - 1, sizes[-1] // 4))
        batches[-1].append(data)
        sizes[-1] += len(data)
    return [b"".join(b) for b in batches], placement


def cameras_to_std430(cameras: Sequence[PinholeCamera], placement) -> bytes:
    out = np.zeros(len(cameras), CAMERA_DTYPE)
    for i, (camera, (batch, offset)) in enumerate(zip(cameras, placement)):
        out[i]["worldToCamera"] = camera.world_to_camera_matrix().T.reshape(16)  # column-major
        out[i]["intrinsics"] = [camera.fx, camera.fy, camera.cx, camera.cy]
        out[i]["center"] = camera.center
        out[i]["pixelOffset"] = offset
        out[i]["size"] = [camera.width, camera.height]
        out[i]["imageBatch"] = batch
    return out.tobytes()


def cameras_from_std430(data: bytes) -> np.ndarray:
    """The Camera structs of a buffer-file, as a numpy structured array."""
    if len(data) % CAMERA_DTYPE.itemsize:
        raise ValueError(f"camera buffer of {len(data)} bytes isn't a whole number of 112-byte Cameras")
    return np.frombuffer(data, CAMERA_DTYPE)


# --- The output directory -----------------------------------------------------------------------------

def write_scene(out_dir: str, splats: splat_module.Splats, cameras: Sequence[PinholeCamera],
                held_out: Optional[np.ndarray] = None, source: Optional[dict] = None,
                max_image_bytes: int = MAX_BUFFER_BYTES) -> dict:
    """Writes a scene: splats.bin, cameras.bin, images_<k>.bin and images/<name>.png (when the
    cameras have images), and manifest.json, which it returns. held_out marks the test cameras
    (default: none)."""
    os.makedirs(out_dir, exist_ok=True)
    held_out = np.zeros(len(cameras), dtype=bool) if held_out is None else np.asarray(held_out, dtype=bool)
    if len(held_out) != len(cameras):
        raise ValueError("held_out needs one entry per camera")
    pngs = [os.path.join("images", os.path.splitext(os.path.basename(c.name))[0] + ".png") for c in cameras]
    with_images = [png for png, c in zip(pngs, cameras) if c.image is not None]
    if len(set(with_images)) != len(with_images):
        raise ValueError("two images would be written to the same PNG (same name without extension)")
    buffers, placement = pack_images(cameras, max_image_bytes)
    # The reference computes the extent from the training cameras (all of them if none are)
    train_cameras = [c for c, h in zip(cameras, held_out) if not h] or list(cameras)

    with open(os.path.join(out_dir, "splats.bin"), "wb") as f:
        f.write(splat_module.to_std430(splats))
    with open(os.path.join(out_dir, "cameras.bin"), "wb") as f:
        f.write(cameras_to_std430(cameras, placement))
    image_files = []
    for k, data in enumerate(buffers):
        image_files.append(f"images_{k}.bin")
        with open(os.path.join(out_dir, image_files[-1]), "wb") as f:
            f.write(data)

    entries = []
    for i, (camera, (batch, offset)) in enumerate(zip(cameras, placement)):
        entry = {"index": i, "name": camera.name, "split": "test" if held_out[i] else "train",
                 "width": camera.width, "height": camera.height,
                 "fx": camera.fx, "fy": camera.fy, "cx": camera.cx, "cy": camera.cy}
        if batch != NO_IMAGE:
            entry.update({"imageBatch": batch, "pixelOffset": offset, "png": pngs[i]})
            os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
            save_png(os.path.join(out_dir, pngs[i]), camera.image)
        entries.append(entry)

    manifest = {
        "version": FORMAT_VERSION,
        "splatCount": len(splats),
        "shDegree": splat_module.SH_DEGREE,
        "cameraCount": len(cameras),
        "extent": camera_extent(train_cameras) if train_cameras else 0.0,
        "files": {"splats": "splats.bin", "cameras": "cameras.bin", "images": image_files},
        "train": [i for i in range(len(cameras)) if not held_out[i]],
        "test": [i for i in range(len(cameras)) if held_out[i]],
        "cameras": entries,
        "source": source or {},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    return manifest
