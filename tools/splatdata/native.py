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

# lib/native/ply_reader.h's results
PLY_OK = 0
PLY_CANNOT_READ = 1
PLY_INVALID = 2
PLY_OUT_OF_MEMORY = 3
PLY_MESSAGE_SIZE = 256


class _Ply(ctypes.Structure):
    """c_api.cpp's SplatNativePly"""
    _fields_ = [
        ("splats", ctypes.c_void_p),
        ("count", ctypes.c_uint64),
        ("shDegree", ctypes.c_uint32),
        ("systemError", ctypes.c_int32),
        ("owner", ctypes.c_void_p),
        ("message", ctypes.c_char * PLY_MESSAGE_SIZE),
    ]


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
        lib.splat_native_ply_size.argtypes = []
        lib.splat_native_ply_size.restype = ctypes.c_uint32
        lib.splat_native_read_ply.argtypes = [ctypes.c_char_p, ctypes.POINTER(_Ply)]
        lib.splat_native_read_ply.restype = ctypes.c_uint32
        lib.splat_native_free_ply.argtypes = [ctypes.POINTER(_Ply)]
        lib.splat_native_free_ply.restype = None
        sizes = {"Splat": (lib.splat_native_splat_size(), SPLAT_SIZE),
                 "SplatNativePly": (lib.splat_native_ply_size(), ctypes.sizeof(_Ply))}
        for name, (theirs, ours) in sizes.items():
            if theirs != ours:
                raise RuntimeError(f"{LIBRARY}: its {name} is {theirs} bytes, the binding's {ours}: "
                                   "rebuild it, or update tools/splatdata/native.py")
        _library = lib
    return _library


def read_ply(path: str) -> tuple[bytes, int]:
    """Reads an INRIA PLY (lib/native/ply_reader.h): its splats as std430 Splat structs (SPLAT_SIZE
    bytes each) and the SH degree (0–3) the file has. Raises OSError when the file can't be read,
    ValueError when it isn't a PLY the reader supports and MemoryError when the splats don't fit."""
    encoded = os.fsencode(path)
    if b"\0" in encoded:  # c_char_p would cut the path there
        raise ValueError(f"{path!r}: the path has a NUL byte")
    lib = library()
    ply = _Ply()
    status = lib.splat_native_read_ply(encoded, ctypes.byref(ply))
    if status != PLY_OK:
        message = f"{path}: {ply.message.decode('utf-8', 'replace')}"
        if status == PLY_CANNOT_READ:
            raise OSError(ply.systemError, f"{message} ({os.strerror(ply.systemError)})", path)
        if status == PLY_INVALID:
            raise ValueError(message)
        if status == PLY_OUT_OF_MEMORY:
            raise MemoryError(message)
        raise RuntimeError(f"{message} (unknown status {status})")
    try:
        return ctypes.string_at(ply.splats, ply.count * SPLAT_SIZE) if ply.count else b"", ply.shDegree
    finally:
        lib.splat_native_free_ply(ctypes.byref(ply))
