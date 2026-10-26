"""Gaussian splats: the stored parameters, their initialization from points, INRIA PLY files, and
the std430 `Splat` buffer the programs read (layout in docs/data-format.md).

Stored parameters are pre-activation, as in the paper and the INRIA PLY: position, log-scale,
quaternion (w, x, y, z; not necessarily unit), opacity logit, and SH coefficients of degree 3
(16 per colour channel, the DC term first).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

SH_DEGREE = 3
SH_COEFFICIENTS = (SH_DEGREE + 1) ** 2  # 16
SH_C0 = 0.28209479177387814  # Y_0^0 = 1 / (2 sqrt(pi))

# Initial opacity (reference code: inverse_sigmoid(0.1))
INITIAL_OPACITY = 0.1


@dataclass
class Splats:
    position: np.ndarray  # (N, 3)
    log_scale: np.ndarray  # (N, 3)
    rotation: np.ndarray  # (N, 4), (w, x, y, z)
    opacity_logit: np.ndarray  # (N,)
    sh: np.ndarray  # (N, 16, 3): coefficient k of channel c is sh[:, k, c]

    def __post_init__(self):
        n = len(self.position)
        shapes = {"position": (n, 3), "log_scale": (n, 3), "rotation": (n, 4),
                  "opacity_logit": (n,), "sh": (n, SH_COEFFICIENTS, 3)}
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.shape != shape:
                raise ValueError(f"splats: {name} has shape {value.shape}, expected {shape}")
            setattr(self, name, value)

    def __len__(self):
        return len(self.position)


def logit(p):
    p = np.asarray(p, dtype=np.float64)
    return np.log(p / (1 - p))


def rgb_to_sh_dc(rgb: np.ndarray) -> np.ndarray:
    """The DC coefficient whose colour is rgb (0–1): colour = SH_C0·dc + 0.5."""
    return (np.asarray(rgb, dtype=np.float64) - 0.5) / SH_C0


def nearest_neighbour_log_scale(points: np.ndarray, k: int = 3) -> np.ndarray:
    """The initial isotropic log-scale of each point, as the reference code computes it
    (simple-knn's distCUDA2): log of the root of the mean squared distance to its k nearest
    other points, the mean clamped to at least 1e-7. Points must be distinct enough to have k
    neighbours; duplicates count as neighbours at distance 0."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= k:
        raise ValueError(f"need more than {k} points for the nearest-neighbour scale, got {len(points)}")
    distances, _ = cKDTree(points).query(points, k=k + 1)  # The first is the point itself
    mean_sq = np.maximum(np.mean(distances[:, 1:] ** 2, axis=1), 1e-7)
    return 0.5 * np.log(mean_sq)


def from_points(points: np.ndarray, rgb: np.ndarray) -> Splats:
    """Initial splats at points (N, 3) with colours rgb (N, 3; uint8 or 0–1 floats), as in the
    paper (Sec. 5.1) and the reference code: isotropic scales from the 3 nearest neighbours,
    identity rotations, opacity 0.1, and the colour in the DC term with the other SH zero."""
    rgb = np.asarray(rgb)
    colour = rgb / 255.0 if rgb.dtype == np.uint8 else rgb.astype(np.float64)
    n = len(points)
    sh = np.zeros((n, SH_COEFFICIENTS, 3))
    sh[:, 0, :] = rgb_to_sh_dc(colour)
    return Splats(
        position=points,
        log_scale=np.repeat(nearest_neighbour_log_scale(points)[:, None], 3, axis=1),
        rotation=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        opacity_logit=np.full(n, logit(INITIAL_OPACITY)),
        sh=sh,
    )


# --- The std430 Splat struct ----------------------------------------------------------------------
# (struct Splat (position vec3) (opacity float) (logScale vec3) (rotation vec4) (sh (array float 48)))
# Offsets: position 0, opacity 12, logScale 16, (4 bytes padding), rotation 32, sh 48; 240 bytes.
SPLAT_DTYPE = np.dtype({
    "names": ["position", "opacity", "logScale", "rotation", "sh"],
    "formats": [("<f4", 3), "<f4", ("<f4", 3), ("<f4", 4), ("<f4", 48)],
    "offsets": [0, 12, 16, 32, 48],
    "itemsize": 240,
})


def to_std430(splats: Splats) -> bytes:
    """The splats as a buffer-file of std430 Splat structs (padding zeroed)."""
    out = np.zeros(len(splats), SPLAT_DTYPE)
    out["position"] = splats.position
    out["opacity"] = splats.opacity_logit
    out["logScale"] = splats.log_scale
    out["rotation"] = splats.rotation
    out["sh"] = splats.sh.reshape(len(splats), 48)
    return out.tobytes()


def from_std430(data: bytes) -> Splats:
    if len(data) % SPLAT_DTYPE.itemsize:
        raise ValueError(f"splat buffer of {len(data)} bytes isn't a whole number of "
                         f"{SPLAT_DTYPE.itemsize}-byte Splats")
    a = np.frombuffer(data, SPLAT_DTYPE)
    return Splats(position=a["position"], log_scale=a["logScale"], rotation=a["rotation"],
                  opacity_logit=a["opacity"], sh=a["sh"].reshape(len(a), SH_COEFFICIENTS, 3))


# --- INRIA PLY ------------------------------------------------------------------------------------
# The reference code's format: binary little-endian, one vertex element of float properties
# x y z nx ny nz f_dc_0..2 f_rest_0..44 opacity scale_0..2 rot_0..3. f_rest is channel-major: the
# coefficient k (1–15) of channel c is f_rest_{c·15 + k − 1}. Scales are logs, opacity a logit and
# rot (w, x, y, z), unnormalized.

_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def ply_property_names(rest_per_channel: int = SH_COEFFICIENTS - 1):
    return (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
            + [f"f_rest_{i}" for i in range(3 * rest_per_channel)]
            + ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])


def write_ply(path: str, splats: Splats) -> None:
    """Writes the splats in the INRIA format (SH degree 3, normals zero)."""
    names = ply_property_names()
    n = len(splats)
    rest = splats.sh[:, 1:, :].transpose(0, 2, 1).reshape(n, 45)  # channel-major
    columns = np.concatenate([splats.position, np.zeros((n, 3)), splats.sh[:, 0, :], rest,
                              splats.opacity_logit[:, None], splats.log_scale, splats.rotation], axis=1)
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property float {name}" for name in names] + ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(columns.astype("<f4").tobytes())


def read_ply(path: str) -> Splats:
    """Reads an INRIA-format PLY: binary little-endian, with its vertex element first. Properties
    may be in any order and any numeric type; SH of degree below 3 (fewer f_rest) is padded with
    zeros, and normals are ignored."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        fmt, count, props, element = None, None, [], None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header")
            tokens = line.decode("ascii").split()
            if not tokens or tokens[0] in ("comment", "obj_info"):
                continue
            if tokens[0] == "end_header":
                break
            if tokens[0] == "format":
                fmt = tokens[1]
            elif tokens[0] == "element":
                if element == "vertex":  # Later elements (faces) are ignored
                    element = "after"
                elif element is None:
                    if tokens[1] != "vertex":
                        raise ValueError(f"{path}: the first element is {tokens[1]}, not vertex")
                    element, count = "vertex", int(tokens[2])
            elif tokens[0] == "property" and element == "vertex":
                if tokens[1] == "list":
                    raise ValueError(f"{path}: list property {tokens[-1]} in vertex")
                if tokens[1] not in _PLY_TYPES:
                    raise ValueError(f"{path}: property {tokens[2]} has unknown type {tokens[1]}")
                props.append((tokens[2], "<" + _PLY_TYPES[tokens[1]]))
        if fmt != "binary_little_endian":
            raise ValueError(f"{path}: format {fmt}; only binary_little_endian is supported")
        if count is None:
            raise ValueError(f"{path}: no vertex element")
        dtype = np.dtype(props)
        data = np.frombuffer(f.read(count * dtype.itemsize), dtype)
        if len(data) != count:
            raise ValueError(f"{path}: {count} vertices declared, {len(data)} present")

    names = set(dtype.names)
    required = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"] + \
        [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"{path}: missing properties {', '.join(missing)}")
    rest_names = sorted((name for name in names if name.startswith("f_rest_")), key=lambda s: int(s[7:]))
    if [int(name[7:]) for name in rest_names] != list(range(len(rest_names))):
        raise ValueError(f"{path}: f_rest properties aren't numbered 0..{len(rest_names) - 1}")
    per_channel = len(rest_names) // 3
    if len(rest_names) % 3 or per_channel + 1 not in (1, 4, 9, 16):
        raise ValueError(f"{path}: {len(rest_names)} f_rest properties: not SH of degree 0–3")

    col = lambda name: data[name].astype(np.float64)  # noqa: E731
    sh = np.zeros((count, SH_COEFFICIENTS, 3))
    for c in range(3):
        sh[:, 0, c] = col(f"f_dc_{c}")
        for k in range(1, per_channel + 1):
            sh[:, k, c] = col(f"f_rest_{c * per_channel + k - 1}")
    return Splats(
        position=np.stack([col("x"), col("y"), col("z")], 1),
        log_scale=np.stack([col(f"scale_{i}") for i in range(3)], 1),
        rotation=np.stack([col(f"rot_{i}") for i in range(4)], 1),
        opacity_logit=col("opacity"),
        sh=sh,
    )
