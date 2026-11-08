"""The ctypes binding to libsplat_native (lib/native/c_api.cpp): the logic the data tools share
with the C++ programs, which lives once in C++ (lib/native/). make builds the library into
build/native/ (`make native`; `make test-tools` builds it first); it's loaded on first use, so the
tools that don't call it run without it. $SPLAT_NATIVE_LIB, set by make, names another build.

ctypes doesn't check the signatures below against the library, so every function is exercised by
tools/tests/test_native.py, and loading refuses a library whose structs' sizes differ from these.
"""

from __future__ import annotations

import ctypes
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# make's build (it sets $SPLAT_NATIVE_LIB for its recipes, so a build with another OUT is the one
# loaded), or build/native/'s when run outside make
LIBRARY = os.environ.get("SPLAT_NATIVE_LIB") or os.path.join(
    REPO, "build", "native", "libsplat_native." + ("dylib" if sys.platform == "darwin" else "so"))

SPLAT_SIZE = 240  # The std430 Splat (lib/splat_math.zgal), as splats.SPLAT_DTYPE
CAMERA_SIZE = 112  # The std430 Camera (lib/splat_math.zgal), as scene.CAMERA_DTYPE
STREAM_HEADER_SIZE = 432  # The StreamHeader (lib/splat_stream.zgal)
STREAM_MAX_SPLATS = 559240  # streamMaxSplats

# lib/splat_stream.zgal's encodings and field groups (the header's order)
ENCODING_F32 = 0
ENCODING_F16 = 1
ENCODING_U16 = 2
ENCODING_U8 = 3
GROUP_NAMES = ("position", "scale", "rotation", "opacity", "dc", "sh")
GROUP_COUNT = len(GROUP_NAMES)

# lib/native/stream_encoder.h's range modes and statuses
RANGE_NONE = 0
RANGE_DATA = 1
RANGE_EXPLICIT = 2
ENCODE_OK = 0
ENCODE_BAD_CONFIG = 1
ENCODE_TOO_MANY_SPLATS = 2
ENCODE_BAD_OUTPUT_SIZE = 3
ENCODE_NON_FINITE = 4
ENCODE_ZERO_ROTATION = 5
ENCODE_F16_OVERFLOW = 6
ENCODE_OUT_OF_MEMORY = 7
ENCODE_RANGE_OVERFLOW = 8

# The readers' results (lib/native/read_status.h)
READ_OK = 0
READ_CANNOT_READ = 1
READ_INVALID = 2
READ_OUT_OF_MEMORY = 3
READ_MESSAGE_SIZE = 256


class _Splats(ctypes.Structure):
    """c_api.cpp's SplatNativeSplats"""
    _fields_ = [
        ("splats", ctypes.c_void_p),
        ("count", ctypes.c_uint64),
        ("shDegree", ctypes.c_uint32),
        ("systemError", ctypes.c_int32),
        ("owner", ctypes.c_void_p),
        ("message", ctypes.c_char * READ_MESSAGE_SIZE),
    ]


class GroupConfig(ctypes.Structure):
    """stream_encoder.h's GroupConfig: a group's encoding, its range mode and an explicit range
    (lo[c], hi[c] per component, or [0] for SH)."""
    _fields_ = [
        ("encoding", ctypes.c_uint32),
        ("rangeMode", ctypes.c_uint32),
        ("lo", ctypes.c_float * 4),
        ("hi", ctypes.c_float * 4),
    ]


class EncodeConfig(ctypes.Structure):
    """stream_encoder.h's EncodeConfig: the stream's SH degree and each group's config, in the
    header's group order."""
    _fields_ = [
        ("shDegree", ctypes.c_uint32),
        ("groups", GroupConfig * GROUP_COUNT),
    ]


class GroupReport(ctypes.Structure):
    """stream_encoder.h's GroupReport: a group's largest error, raw and activated."""
    _fields_ = [
        ("maxError", ctypes.c_double),
        ("maxActivatedError", ctypes.c_double),
    ]


class EncodeReport(ctypes.Structure):
    """stream_encoder.h's EncodeReport: the status, the group and splat it concerns, and each
    group's errors."""
    _fields_ = [
        ("status", ctypes.c_uint32),
        ("group", ctypes.c_uint32),
        ("splat", ctypes.c_uint32),
        ("groups", GroupReport * GROUP_COUNT),
    ]


class EncodeError(ValueError):
    """The encoder refused its input: `status` (an ENCODE_* value), and the `group` and `splat` it
    concerns (when it concerns them)."""

    def __init__(self, message: str, status: int, group: int, splat: int):
        super().__init__(message)
        self.status = status
        self.group = group
        self.splat = splat


_library = None


def library() -> ctypes.CDLL:
    """The library, loaded and checked on the first call."""
    global _library
    if _library is None:
        if not os.path.exists(LIBRARY):
            raise RuntimeError(f"{LIBRARY} isn't built: run `make native` in {REPO}")
        lib = ctypes.CDLL(LIBRARY)
        lib.splat_native_splat_size.argtypes = []
        lib.splat_native_splat_size.restype = ctypes.c_uint32
        lib.splat_native_splats_struct_size.argtypes = []
        lib.splat_native_splats_struct_size.restype = ctypes.c_uint32
        for reader in (lib.splat_native_read_ply, lib.splat_native_read_splats_bin):
            reader.argtypes = [ctypes.c_char_p, ctypes.POINTER(_Splats)]
            reader.restype = ctypes.c_uint32
        lib.splat_native_free_splats.argtypes = [ctypes.POINTER(_Splats)]
        lib.splat_native_free_splats.restype = None
        for f in [getattr(lib, f"splat_native_{name}_size")
                  for name in ("encode_config", "encode_report", "stream_header", "camera")] \
                + [lib.splat_native_stream_max_splats]:
            f.argtypes = []
            f.restype = ctypes.c_uint32
        lib.splat_native_stream_size.argtypes = [ctypes.POINTER(EncodeConfig), ctypes.c_uint64,
                                                 ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32)]
        lib.splat_native_stream_size.restype = ctypes.c_uint32
        lib.splat_native_encode_stream.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(EncodeConfig),
                                                   ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(EncodeReport)]
        lib.splat_native_encode_stream.restype = ctypes.c_uint32
        lib.splat_native_default_camera.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
        lib.splat_native_default_camera.restype = ctypes.c_uint32
        # The structs' sizes in bytes, and streamMaxSplats
        sizes = {"Splat": (lib.splat_native_splat_size(), SPLAT_SIZE),
                 "SplatNativeSplats": (lib.splat_native_splats_struct_size(), ctypes.sizeof(_Splats)),
                 "EncodeConfig": (lib.splat_native_encode_config_size(), ctypes.sizeof(EncodeConfig)),
                 "EncodeReport": (lib.splat_native_encode_report_size(), ctypes.sizeof(EncodeReport)),
                 "StreamHeader": (lib.splat_native_stream_header_size(), STREAM_HEADER_SIZE),
                 "Camera": (lib.splat_native_camera_size(), CAMERA_SIZE),
                 "streamMaxSplats": (lib.splat_native_stream_max_splats(), STREAM_MAX_SPLATS)}
        for name, (theirs, ours) in sizes.items():
            if theirs != ours:
                raise RuntimeError(f"{LIBRARY}: its {name} is {theirs}, the binding's {ours}: "
                                   "rebuild it, or update tools/splatdata/native.py")
        _library = lib
    return _library


def _read(reader, path: str) -> tuple[bytes, int]:
    """Runs one of the library's readers on `path`: its splats as std430 bytes and its SH degree."""
    encoded = os.fsencode(path)
    if b"\0" in encoded:  # c_char_p would cut the path there
        raise ValueError(f"{path!r}: the path has a NUL byte")
    lib = library()
    out = _Splats()
    status = getattr(lib, reader)(encoded, ctypes.byref(out))
    if status != READ_OK:
        message = f"{path}: {out.message.decode('utf-8', 'replace')}"
        if status == READ_CANNOT_READ:
            raise OSError(out.systemError, f"{message} ({os.strerror(out.systemError)})", path)
        if status == READ_INVALID:
            raise ValueError(message)
        if status == READ_OUT_OF_MEMORY:
            raise MemoryError(message)
        raise RuntimeError(f"{message} (unknown status {status})")
    try:
        return ctypes.string_at(out.splats, out.count * SPLAT_SIZE) if out.count else b"", out.shDegree
    finally:
        lib.splat_native_free_splats(ctypes.byref(out))


def read_ply(path: str) -> tuple[bytes, int]:
    """Reads an INRIA PLY (lib/native/ply_reader.h): its splats as std430 Splat structs (SPLAT_SIZE
    bytes each) and the SH degree (0–3) the file has. Raises OSError when the file can't be read,
    ValueError when it isn't a PLY the reader supports and MemoryError when the splats don't fit."""
    return _read("splat_native_read_ply", path)


def read_splats_bin(path: str) -> tuple[bytes, int]:
    """Reads a splats.bin (lib/native/splats_bin_reader.h): its std430 Splat structs as stored, and
    SH degree 3 (a Splat holds every band). Raises as read_ply does; ValueError when the size isn't
    a whole number of Splats."""
    return _read("splat_native_read_splats_bin", path)


def _splat_count(splats: bytes) -> int:
    if len(splats) % SPLAT_SIZE:
        raise ValueError(f"{len(splats)} bytes isn't a whole number of {SPLAT_SIZE}-byte Splats")
    return len(splats) // SPLAT_SIZE


def _encode_error(status: int, group: int, splat: int) -> EncodeError:
    """The encoder's status as an error, naming the group and the splat."""
    name = GROUP_NAMES[group] if group < GROUP_COUNT else "shDegree"
    messages = {
        ENCODE_BAD_CONFIG: f"the config of {name} is invalid",
        ENCODE_TOO_MANY_SPLATS: f"more than {STREAM_MAX_SPLATS} splats (the f32 Splat buffer's 128 MiB)",
        ENCODE_BAD_OUTPUT_SIZE: "the output isn't the stream's size",
        ENCODE_NON_FINITE: f"splat {splat}'s {name} isn't finite",
        ENCODE_ZERO_ROTATION: f"splat {splat}'s rotation has length 0, which a quantized rotation can't normalize",
        ENCODE_F16_OVERFLOW: f"splat {splat}'s {name} is too large for f16 (|v| >= 65520)",
        ENCODE_OUT_OF_MEMORY: "out of memory",
        ENCODE_RANGE_OVERFLOW: f"the range of {name} is too wide for a float32 step",
    }
    return EncodeError(messages.get(status, f"unknown status {status}"), status, group, splat)


def stream_size(config: EncodeConfig, count: int) -> int:
    """The bytes of the stream of `count` splats under `config` (lib/native/stream_encoder.h's
    streamLayout). Raises EncodeError for a bad config or too many splats."""
    lib = library()
    size = ctypes.c_uint64()
    group = ctypes.c_uint32()
    status = lib.splat_native_stream_size(ctypes.byref(config), count, ctypes.byref(size), ctypes.byref(group))
    if status != ENCODE_OK:
        raise _encode_error(status, group.value, 0)
    return size.value


def encode_stream(splats: bytes, config: EncodeConfig) -> tuple[bytes, EncodeReport]:
    """Encodes std430 Splats (SPLAT_SIZE bytes each) under `config` (lib/native/stream_encoder.h's
    encodeStream): the stream's bytes and the report of each group's error. Raises EncodeError
    when the encoder refuses (naming the group and the splat)."""
    count = _splat_count(splats)
    size = stream_size(config, count)
    out = bytearray(size)
    report = EncodeReport()
    buffer = (ctypes.c_char * size).from_buffer(out) if size else None
    status = library().splat_native_encode_stream(splats, count, ctypes.byref(config), buffer, size,
                                                  ctypes.byref(report))
    if status != ENCODE_OK:
        raise _encode_error(status, report.group, report.splat)
    return bytes(out), report


def default_camera(splats: bytes) -> bytes:
    """The default camera framing std430 Splats (lib/native/default_camera.h): a std430 Camera
    (CAMERA_SIZE bytes), as the encoder writes into a stream's header."""
    count = _splat_count(splats)
    camera = ctypes.create_string_buffer(CAMERA_SIZE)
    if library().splat_native_default_camera(splats, count, camera) != 0:
        raise MemoryError("the default camera: out of memory")
    return camera.raw
