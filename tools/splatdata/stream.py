"""Stream files (docs/data-format.md, "Stream files"): the JSON config `convert.py stream` takes,
checked here (where its errors read best) and passed to the shared C++ encoder
(lib/native/stream_encoder.h, through splatdata.native) as its plain EncodeConfig, and the
encoder's per-group report, formatted for printing. The encoding is the C++ one; nothing here
encodes or decodes a value.

A config gives the stream's SH degree and every group's encoding; none has a default, since each
trades size for precision:

  {"shDegree": "data",                          0-3, or "data": the input's degree
   "position": {"encoding": "f32"},             f32 or f16: no range
   "rotation": {"encoding": "u8", "range": [-1, 1]},
   "opacity": {"encoding": "u8", "range": "data"},
   ...}                                          every group: position, scale, rotation, opacity,
                                                 dc and sh

A quantized group (u16 or u8) takes a "range": "data" (the data's minimum and maximum, per
component, or over every component for SH), [lo, hi] for every component, or, for a group of up to
four components, one [lo, hi] per component. Values outside an explicit range are clamped.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from . import native, scene

ENCODINGS = {"f32": native.ENCODING_F32, "f16": native.ENCODING_F16,
             "u16": native.ENCODING_U16, "u8": native.ENCODING_U8}
ENCODING_NAMES = {v: k for k, v in ENCODINGS.items()}
QUANTIZED = {"u16": 16, "u8": 8}  # their bits
GROUPS = native.GROUP_NAMES
# Each group's components (SH's depend on the degree and share one range)
COMPONENTS = {"position": 3, "scale": 3, "rotation": 4, "opacity": 1, "dc": 3}
# What a group's activated error measures (the encoder's report)
ACTIVATED = {"position": "value", "scale": "exp", "rotation": "angle (rad)", "opacity": "sigmoid",
             "dc": "value", "sh": "value"}

# The StreamHeader (lib/splat_stream.zgal), to read a stream's layout and ranges back
FIELD_DTYPE = np.dtype({
    "names": ["encoding", "components", "offset", "sharedRange", "low", "stepSize"],
    "formats": ["<u4", "<u4", "<u4", "<u4", ("<f4", 4), ("<f4", 4)],
    "offsets": [0, 4, 8, 12, 16, 32],
    "itemsize": 48,
})
HEADER_DTYPE = np.dtype({
    "names": ["magic", "version", "splatCount", "shDegree", "baseStride", "shStride", "baseOffset",
              "shOffset", "camera", "fields"],
    "formats": ["<u4"] * 8 + [scene.CAMERA_DTYPE, (FIELD_DTYPE, len(GROUPS))],
    "offsets": [0, 4, 8, 12, 16, 20, 24, 28, 32, 144],
    "itemsize": native.STREAM_HEADER_SIZE,
})


class ConfigError(ValueError):
    """A config that isn't one: the message names the file and the key."""


Range = Union[str, List[Tuple[float, float]]]  # "data", or (lo, hi) per range


@dataclass
class GroupConfig:
    encoding: str
    range: Range | None  # None for f32 and f16


@dataclass
class StreamConfig:
    sh_degree: Union[int, str]  # 0-3 or "data"
    groups: Dict[str, GroupConfig]


def _range_pair(value, where: str) -> Tuple[float, float]:
    if (not isinstance(value, list) or len(value) != 2
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)):
        raise ConfigError(f"{where} must be [lo, hi] (two numbers)")
    try:
        with np.errstate(over="ignore"):  # a value beyond float32 becomes inf, refused below
            lo, hi = (float(np.float32(float(v))) for v in value)
    except OverflowError:  # a JSON integer beyond even float64
        lo = hi = math.inf
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ConfigError(f"{where} isn't finite in float32")
    if lo > hi:
        raise ConfigError(f"{where} [{value[0]}, {value[1]}] has lo above hi")
    return lo, hi


def _group(name: str, value, where: str) -> GroupConfig:
    if not isinstance(value, dict):
        raise ConfigError(f'{where} must be an object, e.g. {{"encoding": "f32"}}')
    unknown = sorted(set(value) - {"encoding", "range"})
    if unknown:
        raise ConfigError(f"{where}: unknown keys {', '.join(unknown)} (a group has encoding and range)")
    encoding = value.get("encoding")
    if encoding not in ENCODINGS:
        raise ConfigError(f"{where}.encoding must be one of {', '.join(ENCODINGS)}, not {json.dumps(encoding)}")
    if encoding not in QUANTIZED:
        if "range" in value:
            raise ConfigError(f"{where}: {encoding} takes no range (only u16 and u8 do)")
        return GroupConfig(encoding, None)
    if "range" not in value:
        raise ConfigError(f'{where}: {encoding} needs a range: "data" or [lo, hi]')
    r = value["range"]
    if r == "data":
        return GroupConfig(encoding, "data")
    if isinstance(r, list) and r and all(isinstance(p, list) for p in r):
        components = COMPONENTS.get(name)
        if components is None:
            raise ConfigError(f'{where}.range: sh shares one range: "data" or [lo, hi]')
        if len(r) != components:
            raise ConfigError(f"{where}.range: {len(r)} [lo, hi] pairs for {components} components")
        return GroupConfig(encoding, [_range_pair(p, f"{where}.range[{i}]") for i, p in enumerate(r)])
    if not isinstance(r, list):
        raise ConfigError(f'{where}.range must be "data" or [lo, hi], not {json.dumps(r)}')
    return GroupConfig(encoding, [_range_pair(r, f"{where}.range")])


def parse_config(value, source: str) -> StreamConfig:
    """Checks a parsed JSON config (`source` names it in errors): raises ConfigError for a missing
    or unknown key or a bad value."""
    if not isinstance(value, dict):
        raise ConfigError(f"{source}: a config must be a JSON object")
    keys = ("shDegree",) + GROUPS
    unknown = sorted(set(value) - set(keys))
    if unknown:
        raise ConfigError(f"{source}: unknown keys {', '.join(unknown)} (a config has {', '.join(keys)})")
    missing = [k for k in keys if k not in value]
    if missing:
        raise ConfigError(f"{source}: missing {', '.join(missing)} (every key is required)")
    degree = value["shDegree"]
    if degree != "data" and (not isinstance(degree, int) or isinstance(degree, bool) or not 0 <= degree <= 3):
        raise ConfigError(f'{source}: shDegree must be 0 to 3 or "data", not {json.dumps(degree)}')
    return StreamConfig(degree, {g: _group(g, value[g], f"{source}: {g}") for g in GROUPS})


def load_config(path: str) -> StreamConfig:
    """Reads and checks a JSON config file (ConfigError for a bad one, OSError if unreadable)."""
    with open(path, encoding="utf-8") as f:
        try:
            value = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigError(f"{path}: not JSON: {e}") from None
    return parse_config(value, path)


def encode_config(config: StreamConfig, data_degree: int, source: str) -> native.EncodeConfig:
    """The encoder's EncodeConfig for input splats of SH degree `data_degree`; ConfigError when the
    config asks for more SH bands than the input holds."""
    degree = data_degree if config.sh_degree == "data" else config.sh_degree
    if degree > data_degree:
        raise ConfigError(f"{source}: shDegree {degree} is above the input's SH degree, {data_degree}")
    out = native.EncodeConfig()
    out.shDegree = degree
    for g, name in enumerate(GROUPS):
        group = config.groups[name]
        out.groups[g].encoding = ENCODINGS[group.encoding]
        if group.range is None:
            out.groups[g].rangeMode = native.RANGE_NONE
        elif group.range == "data":
            out.groups[g].rangeMode = native.RANGE_DATA
        else:
            out.groups[g].rangeMode = native.RANGE_EXPLICIT
            pairs = group.range * 4 if len(group.range) == 1 else group.range
            for c, (lo, hi) in enumerate(pairs[:4]):
                out.groups[g].lo[c], out.groups[g].hi[c] = lo, hi
    return out


def read_header(stream: bytes) -> np.void:
    """A stream's StreamHeader as a numpy record (HEADER_DTYPE)."""
    return np.frombuffer(stream, HEADER_DTYPE, 1)[0]


def _format_range(field) -> str:
    encoding = ENCODING_NAMES[int(field["encoding"])]
    if encoding not in QUANTIZED or field["components"] == 0:
        return "-"
    ranges = 1 if field["sharedRange"] else int(field["components"])
    codes = np.float32((1 << QUANTIZED[encoding]) - 1)
    lows = field["low"][:ranges]
    highs = lows + field["stepSize"][:ranges] * codes
    pairs = [f"[{lo:.4g}, {hi:.4g}]" for lo, hi in zip(lows, highs)]
    return pairs[0] if len(set(pairs)) == 1 else ", ".join(pairs)


def format_report(stream: bytes, report: native.EncodeReport, range_modes: Sequence[str]) -> str:
    """The stream's layout and each group's encoding, largest error (raw, and activated: of what
    the renderer uses, ACTIVATED) and range (`range_modes[g]`, then the range the header holds:
    low to low + stepSize (2^bits - 1), once when every component's is the same)."""
    h = read_header(stream)
    n, base, sh = int(h["splatCount"]), int(h["baseStride"]), int(h["shStride"])
    lines = [f"{n} splats, SH degree {int(h['shDegree'])}, {base + sh} B per splat (base {base}, SH {sh}): "
             f"{len(stream)} B, the SH records from byte {int(h['shOffset'])}",
             f"  {'group':<9} {'encoding':<8} {'max error':>10} {'activated':>10}  {'(of)':<12} range"]
    for g, name in enumerate(GROUPS):
        field = h["fields"][g]
        r = report.groups[g]
        components = int(field["components"])
        encoding = ENCODING_NAMES[int(field["encoding"])] if components else "(none)"
        mode = range_modes[g] if components and range_modes[g] != "-" else ""
        lines.append(f"  {name:<9} {encoding:<8} {r.maxError:>10.3g} {r.maxActivatedError:>10.3g}  "
                     f"{ACTIVATED[name]:<12} {(mode + ' ' if mode else '') + _format_range(field)}")
    return "\n".join(lines)


def range_modes(config: StreamConfig) -> List[str]:
    """Each group's range as the config gives it: "-", "data" or "explicit"."""
    return ["-" if c.range is None else "data" if c.range == "data" else "explicit"
            for c in (config.groups[g] for g in GROUPS)]
