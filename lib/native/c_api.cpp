// libsplat_native: a C ABI over lib/native/'s headers, for callers that can't include C++ (the
// converter's ctypes binding, tools/splatdata/native.py). Plain structs and status codes; nothing
// throws across it. Every function here is called by the binding's tests
// (tools/tests/test_native.py), since ctypes doesn't check signatures.
#include "c_api.zgal.h"

namespace splat_native {
namespace gen = ::c_api;
}

#include "ply_reader.h"

#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

#define SPLAT_NATIVE_EXPORT extern "C" __attribute__((visibility("default")))

// A PLY read by splat_native_read_ply: `count` Splats (240-byte std430 structs, as the programs'
// buffers hold them) at `splats`, owned by the library until splat_native_free_ply
struct SplatNativePly {
    const void* splats;
    uint64_t count;
    uint32_t shDegree;    // 0-3: the bands the file has (the others are zero)
    int32_t systemError;  // splat_native_read_ply's kPlyCannotRead: the errno
    void* owner;          // the library's: the vector holding the splats
    char message[splat_native::kPlyMessageSize];  // any error: what went wrong
};

// The sizes of the structs the binding mirrors, so it can refuse a library that doesn't match it
SPLAT_NATIVE_EXPORT uint32_t splat_native_splat_size(void) { return sizeof(c_api::Splat); }
SPLAT_NATIVE_EXPORT uint32_t splat_native_ply_size(void) { return sizeof(SplatNativePly); }

// Reads the PLY at `path` into `ply` (lib/native/ply_reader.h): returns kPlyOk (0) with `ply`
// holding the splats, which the caller frees with splat_native_free_ply, or the error (kPlyCannotRead
// 1, kPlyInvalid 2, kPlyOutOfMemory 3) with `ply`'s message (and systemError) set and nothing to free
SPLAT_NATIVE_EXPORT uint32_t splat_native_read_ply(const char* path, SplatNativePly* ply) {
    *ply = SplatNativePly{};
    auto* splats = new (std::nothrow) std::vector<c_api::Splat>();
    if (!splats) {
        std::strcpy(ply->message, "out of memory");
        return splat_native::kPlyOutOfMemory;
    }
    uint32_t shDegree = 0;
    const splat_native::PlyStatus status = splat_native::readPly(path, *splats, shDegree);
    if (status.code != splat_native::kPlyOk) {
        delete splats;
        ply->systemError = status.systemError;
        std::memcpy(ply->message, status.message, sizeof(ply->message));
        return status.code;
    }
    ply->splats = splats->data();
    ply->count = splats->size();
    ply->shDegree = shDegree;
    ply->owner = splats;
    return splat_native::kPlyOk;
}

// Frees what splat_native_read_ply returned, and empties `ply` (so freeing twice is harmless)
SPLAT_NATIVE_EXPORT void splat_native_free_ply(SplatNativePly* ply) {
    delete static_cast<std::vector<c_api::Splat>*>(ply->owner);
    *ply = SplatNativePly{};
}
