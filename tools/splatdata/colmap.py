"""COLMAP's text export (cameras.txt, images.txt, points3D.txt), parsed as COLMAP writes it.

- images.txt has two lines per image, the second (its 2D points) possibly empty.
- points3D.txt lines have any number of track entries.
- Quaternions are stored w first, and (R, t) maps world to camera: X_cam = R X + t.
- Files may have CRLF line endings.

Pixel coordinates are COLMAP's: the top-left pixel's centre is (0.5, 0.5).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from . import quaternion

# Each supported model's parameter names, in COLMAP's order
CAMERA_MODELS = {
    "SIMPLE_PINHOLE": ("f", "cx", "cy"),
    "PINHOLE": ("fx", "fy", "cx", "cy"),
    "SIMPLE_RADIAL": ("f", "cx", "cy", "k"),
    "RADIAL": ("f", "cx", "cy", "k1", "k2"),
    "OPENCV": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"),
}


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray  # float64, as named in CAMERA_MODELS[model]

    def __post_init__(self):
        if self.model not in CAMERA_MODELS:
            raise ValueError(f"camera {self.id}: unsupported model {self.model} "
                             f"(supported: {', '.join(CAMERA_MODELS)})")
        if len(self.params) != len(CAMERA_MODELS[self.model]):
            raise ValueError(f"camera {self.id}: {self.model} takes {len(CAMERA_MODELS[self.model])} "
                             f"parameters, got {len(self.params)}")

    def param(self, name: str) -> float:
        return float(self.params[CAMERA_MODELS[self.model].index(name)])

    @property
    def focal(self):
        """(fx, fy)."""
        if "f" in CAMERA_MODELS[self.model]:
            return self.param("f"), self.param("f")
        return self.param("fx"), self.param("fy")

    @property
    def principal_point(self):
        """(cx, cy)."""
        return self.param("cx"), self.param("cy")

    @property
    def has_distortion(self) -> bool:
        return self.model not in ("SIMPLE_PINHOLE", "PINHOLE") and bool(np.any(self.distortion_params() != 0))

    def distortion_params(self) -> np.ndarray:
        """(k1, k2, p1, p2): the OPENCV model's terms, zero where this model has none."""
        names = CAMERA_MODELS[self.model]
        get = lambda n: self.param(n) if n in names else 0.0  # noqa: E731
        return np.array([get("k1") if "k1" in names else get("k"), get("k2"), get("p1"), get("p2")])

    def distort(self, normalized: np.ndarray) -> np.ndarray:
        """Applies the lens distortion to normalized image coordinates (x/z, y/z), shape (..., 2)."""
        k1, k2, p1, p2 = self.distortion_params()
        u, v = normalized[..., 0], normalized[..., 1]
        r2 = u * u + v * v
        radial = k1 * r2 + k2 * r2 * r2
        du = u * radial + 2 * p1 * u * v + p2 * (r2 + 2 * u * u)
        dv = v * radial + 2 * p2 * u * v + p1 * (r2 + 2 * v * v)
        return np.stack([u + du, v + dv], -1)

    def undistort(self, distorted: np.ndarray, iterations: int = 100) -> np.ndarray:
        """Inverts distort() by Newton's method (numerical Jacobian, as COLMAP does)."""
        distorted = np.asarray(distorted, dtype=np.float64)
        x = distorted.copy()
        if not self.has_distortion:
            return x
        eps = 1e-9
        for _ in range(iterations):
            f = self.distort(x) - distorted
            ju = (self.distort(x + [eps, 0]) - self.distort(x - [eps, 0])) / (2 * eps)
            jv = (self.distort(x + [0, eps]) - self.distort(x - [0, eps])) / (2 * eps)
            a, c = ju[..., 0], ju[..., 1]  # d(distorted)/du
            b, d = jv[..., 0], jv[..., 1]  # d(distorted)/dv
            det = a * d - b * c
            step = np.stack([(d * f[..., 0] - b * f[..., 1]) / det, (a * f[..., 1] - c * f[..., 0]) / det], -1)
            x = x - step
            if np.max(np.abs(step), initial=0) < 1e-12:
                break
        return x

    def to_pixels(self, normalized: np.ndarray) -> np.ndarray:
        """Distorts normalized coordinates and maps them to pixels."""
        (fx, fy), (cx, cy) = self.focal, self.principal_point
        d = self.distort(normalized)
        return np.stack([fx * d[..., 0] + cx, fy * d[..., 1] + cy], -1)

    def to_normalized(self, pixels: np.ndarray) -> np.ndarray:
        """Inverts to_pixels()."""
        (fx, fy), (cx, cy) = self.focal, self.principal_point
        d = np.stack([(pixels[..., 0] - cx) / fx, (pixels[..., 1] - cy) / fy], -1)
        return self.undistort(d)


@dataclass
class Image:
    id: int
    qvec: np.ndarray  # (w, x, y, z), world to camera
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # POINTS2D positions
    point3D_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))  # -1: none

    @property
    def rotation(self) -> np.ndarray:
        """R, world to camera."""
        return quaternion.to_matrix(self.qvec)

    @property
    def center(self) -> np.ndarray:
        """The camera centre in world space, −Rᵀt."""
        return -np.einsum("ji,j->i", self.rotation, self.tvec)

    def world_to_camera(self, points: np.ndarray) -> np.ndarray:
        # einsum, not @: numpy's matmul on macOS's Accelerate raises spurious FP warnings
        return np.einsum("ij,nj->ni", self.rotation, points) + self.tvec


@dataclass
class Points3D:
    ids: np.ndarray  # int64, (N,)
    xyz: np.ndarray  # float64, (N, 3)
    rgb: np.ndarray  # uint8, (N, 3)
    error: np.ndarray  # float64, (N,): COLMAP's mean reprojection error, in pixels
    # Tracks, flattened: observation i is point track_point[i] (an index into ids), seen in image
    # track_image_id[i] as that image's POINTS2D[track_point2D_idx[i]]
    track_point: np.ndarray
    track_image_id: np.ndarray
    track_point2D_idx: np.ndarray


@dataclass
class Model:
    cameras: Dict[int, Camera]
    images: Dict[int, Image]
    points: Points3D


def _data_lines(path: str) -> List[str]:
    """The file's lines without comments or line endings; blank lines are kept."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\r\n") for line in f if not line.startswith("#")]


def read_cameras(path: str) -> Dict[int, Camera]:
    cameras = {}
    for line in _data_lines(path):
        tokens = line.split()
        if not tokens:
            continue
        camera = Camera(int(tokens[0]), tokens[1], int(tokens[2]), int(tokens[3]),
                        np.array([float(t) for t in tokens[4:]]))
        cameras[camera.id] = camera
    return cameras


def read_images(path: str) -> Dict[int, Image]:
    lines = _data_lines(path)
    images = {}
    i = 0
    while i < len(lines):
        tokens = lines[i].split()
        if not tokens:  # A blank line where a record would start (e.g. at the end)
            i += 1
            continue
        if len(tokens) < 10:
            raise ValueError(f"{path}: image record with {len(tokens)} fields: {lines[i]!r}")
        points = lines[i + 1].split() if i + 1 < len(lines) else []
        if len(points) % 3:
            raise ValueError(f"{path}: image {tokens[0]}: POINTS2D has {len(points)} values")
        p = np.array(points, dtype=np.float64).reshape(-1, 3)
        image = Image(
            id=int(tokens[0]),
            qvec=np.array([float(t) for t in tokens[1:5]]),
            tvec=np.array([float(t) for t in tokens[5:8]]),
            camera_id=int(tokens[8]),
            name=" ".join(tokens[9:]),
            xys=p[:, :2],
            point3D_ids=p[:, 2].astype(np.int64),
        )
        images[image.id] = image
        i += 2
    return images


def read_points3D(path: str) -> Points3D:
    ids, xyz, rgb, error = [], [], [], []
    track_point, track_image, track_idx = [], [], []
    for line in _data_lines(path):
        tokens = line.split()
        if not tokens:
            continue
        if len(tokens) < 8 or (len(tokens) - 8) % 2:
            raise ValueError(f"{path}: malformed point line: {line[:80]!r}")
        index = len(ids)
        ids.append(int(tokens[0]))
        xyz.append([float(t) for t in tokens[1:4]])
        rgb.append([int(t) for t in tokens[4:7]])
        error.append(float(tokens[7]))
        track = np.array(tokens[8:], dtype=np.int64).reshape(-1, 2)
        track_point.append(np.full(len(track), index, dtype=np.int64))
        track_image.append(track[:, 0])
        track_idx.append(track[:, 1])
    cat = lambda parts: np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)  # noqa: E731
    return Points3D(
        ids=np.array(ids, dtype=np.int64),
        xyz=np.array(xyz, dtype=np.float64).reshape(-1, 3),
        rgb=np.array(rgb, dtype=np.uint8).reshape(-1, 3),
        error=np.array(error, dtype=np.float64),
        track_point=cat(track_point),
        track_image_id=cat(track_image),
        track_point2D_idx=cat(track_idx),
    )


def read_model(directory: str) -> Model:
    """Reads cameras.txt, images.txt and points3D.txt from a COLMAP text export."""
    model = Model(
        cameras=read_cameras(os.path.join(directory, "cameras.txt")),
        images=read_images(os.path.join(directory, "images.txt")),
        points=read_points3D(os.path.join(directory, "points3D.txt")),
    )
    for image in model.images.values():
        if image.camera_id not in model.cameras:
            raise ValueError(f"image {image.id} ({image.name}) refers to missing camera {image.camera_id}")
    return model


def project(camera: Camera, image: Image, points: np.ndarray) -> np.ndarray:
    """Projects world points (N, 3) through the image's pose and its camera, distortion included,
    to pixel coordinates (N, 2). Points must be in front of the camera."""
    p = image.world_to_camera(points)
    return camera.to_pixels(p[:, :2] / p[:, 2:3])
