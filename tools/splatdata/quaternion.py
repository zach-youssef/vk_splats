"""Unit quaternions, stored w first (COLMAP's and the INRIA PLY's order), and rotation matrices.

Arrays of quaternions have shape (..., 4) as (w, x, y, z); matrices (..., 3, 3), acting on column
vectors (v' = R v).
"""

from __future__ import annotations

import numpy as np


def normalize(q: np.ndarray) -> np.ndarray:
    """Returns q / |q| along the last axis; a zero quaternion is an error."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(n == 0):
        raise ValueError("zero quaternion")
    return q / n


def to_matrix(q: np.ndarray) -> np.ndarray:
    """The rotation matrix of each quaternion (w, x, y, z), normalized first."""
    w, x, y, z = np.moveaxis(normalize(q), -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def from_matrix(r: np.ndarray) -> np.ndarray:
    """The unit quaternion (w, x, y, z) of each rotation matrix, with w >= 0.

    Shepperd's method: divides by the largest of 4w², 4x², 4y², 4z², so it's stable for every
    rotation, including half turns (w = 0).
    """
    r = np.asarray(r, dtype=np.float64)
    m00, m01, m02 = r[..., 0, 0], r[..., 0, 1], r[..., 0, 2]
    m10, m11, m12 = r[..., 1, 0], r[..., 1, 1], r[..., 1, 2]
    m20, m21, m22 = r[..., 2, 0], r[..., 2, 1], r[..., 2, 2]
    # Each row: a candidate quaternion scaled by 4 times one of its components
    candidates = np.stack(
        [
            np.stack([1 + m00 + m11 + m22, m21 - m12, m02 - m20, m10 - m01], -1),
            np.stack([m21 - m12, 1 + m00 - m11 - m22, m01 + m10, m02 + m20], -1),
            np.stack([m02 - m20, m01 + m10, 1 - m00 + m11 - m22, m12 + m21], -1),
            np.stack([m10 - m01, m02 + m20, m12 + m21, 1 - m00 - m11 + m22], -1),
        ],
        -2,
    )
    diag = np.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1)
    best = np.argmax(diag, axis=-1)
    q = np.take_along_axis(candidates, best[..., None, None], axis=-2)[..., 0, :]
    q = normalize(q)
    return np.where(q[..., :1] < 0, -q, q)
