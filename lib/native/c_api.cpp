// libsplat_native: a C ABI over lib/native/'s headers, for callers that can't include C++ (the
// converter's ctypes binding, tools/splatdata/native.py). Plain structs and status codes; nothing
// throws across it. Every function here is called by the binding's tests
// (tools/tests/test_native.py), since ctypes doesn't check signatures.
#include "c_api.zgal.h"

namespace splat_native {
namespace gen = ::c_api;
}

#include "ply_reader.h"
#include "splats_bin_reader.h"

#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

#define SPLAT_NATIVE_EXPORT extern "C" __attribute__((visibility("default")))

// Splats read by splat_native_read_ply or splat_native_read_splats_bin: `count` Splats (240-byte
// std430 structs, as the programs' buffers hold them) at `splats`, owned by the library until
// splat_native_free_splats
struct SplatNativeSplats {
    const void* splats;
    uint64_t count;
    uint32_t shDegree;    // 0-3: the bands the file has (the others are zero); 3 for a splats.bin
    int32_t systemError;  // kReadCannotRead: the errno
    void* owner;          // the library's: the vector holding the splats
    char message[splat_native::kReadMessageSize];  // any error: what went wrong
};

// The sizes of the structs the binding mirrors, so it can refuse a library that doesn't match it
SPLAT_NATIVE_EXPORT uint32_t splat_native_splat_size(void) { return sizeof(c_api::Splat); }
SPLAT_NATIVE_EXPORT uint32_t splat_native_splats_struct_size(void) { return sizeof(SplatNativeSplats); }

namespace {

// Runs `read` (a reader's call filling a vector of Splats and its SH degree) into `out`
template <typename Read>
uint32_t readSplats(SplatNativeSplats* out, Read read) {
    *out = SplatNativeSplats{};
    auto* splats = new (std::nothrow) std::vector<c_api::Splat>();
    if (!splats) {
        std::strcpy(out->message, "out of memory");
        return splat_native::kReadOutOfMemory;
    }
    uint32_t shDegree = 0;
    const splat_native::ReadStatus status = read(*splats, shDegree);
    if (status.code != splat_native::kReadOk) {
        delete splats;
        out->systemError = status.systemError;
        std::memcpy(out->message, status.message, sizeof(out->message));
        return status.code;
    }
    out->splats = splats->data();
    out->count = splats->size();
    out->shDegree = shDegree;
    out->owner = splats;
    return splat_native::kReadOk;
}

}  // namespace

// Reads the PLY at `path` (lib/native/ply_reader.h) into `out`: returns kReadOk (0) with `out`
// holding the splats, which the caller frees with splat_native_free_splats, or the error
// (kReadCannotRead 1, kReadInvalid 2, kReadOutOfMemory 3) with `out`'s message (and systemError)
// set and nothing to free
SPLAT_NATIVE_EXPORT uint32_t splat_native_read_ply(const char* path, SplatNativeSplats* out) {
    return readSplats(out, [&](std::vector<c_api::Splat>& splats, uint32_t& shDegree) {
        return splat_native::readPly(path, splats, shDegree);
    });
}

// Reads the splats.bin at `path` (lib/native/splats_bin_reader.h) into `out`, as
// splat_native_read_ply does; its shDegree is 3
SPLAT_NATIVE_EXPORT uint32_t splat_native_read_splats_bin(const char* path, SplatNativeSplats* out) {
    return readSplats(out, [&](std::vector<c_api::Splat>& splats, uint32_t& shDegree) {
        shDegree = splat_native::kSplatsBinShDegree;
        return splat_native::readSplatsBin(path, splats);
    });
}

// Frees what a reader returned, and empties `out` (so freeing twice is harmless)
SPLAT_NATIVE_EXPORT void splat_native_free_splats(SplatNativeSplats* out) {
    delete static_cast<std::vector<c_api::Splat>*>(out->owner);
    *out = SplatNativeSplats{};
}
