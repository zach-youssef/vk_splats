// The splats.bin reader (docs/data-format.md, "Output directory"): a file of std430 Splat structs
// (240 bytes each, as the converter writes and the programs' buffers hold them) to Splat structs.
// The file has no header, so its size must be a whole number of Splats; the values are copied as
// they are (the padding too), and a splats.bin holds every SH band (degree 3).
//
// The entry point takes plain arguments and returns a status (read_status.h), never throwing, so a
// C ABI wraps it as it is. It's inline in `splat_native`, so a linked binary must alias `gen` to one
// namespace in every translation unit (otherwise it's an ODR violation).
//
// Header-only, on top of the C++ that ZGAL generates for a module importing lib/splat_math.zgal
// (its Splat). The generated code lands in the namespace of the .zgal file importing it, so the
// includer names that namespace first:
//   #include "my_program.zgal.h"
//   namespace splat_native { namespace gen = ::my_program; }
//   #include "<relative path>/lib/native/splats_bin_reader.h"
#pragma once

#include "read_status.h"

#include <bit>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <new>
#include <stdexcept>
#include <vector>

#include <sys/stat.h>

namespace splat_native {

static_assert(std::endian::native == std::endian::little,
              "the reader copies the file's little-endian Splats as they are in memory");
static_assert(sizeof(gen::Splat) == 240, "a splats.bin holds 240-byte std430 Splats");

// The SH degree every splats.bin has: the Splat struct holds bands 0 to 3
inline constexpr uint32_t kSplatsBinShDegree = 3;

// Reads the splats.bin at `path` into `splats` (replacing what it held). Never throws: kReadOk, or
// an error with `splats` unspecified (kReadCannotRead with the errno, kReadInvalid when the path
// isn't a regular file or the size isn't a whole number of Splats, kReadOutOfMemory) and the
// message set.
inline ReadStatus readSplatsBin(const char* path, std::vector<gen::Splat>& splats) noexcept {
    ReadStatus status;
    auto fail = [&](uint32_t code, int systemError, const char* message) {
        status.code = code;
        status.systemError = systemError;
        std::snprintf(status.message, sizeof(status.message), "%s", message);
        return status;
    };
    struct File {
        std::FILE* f;
        ~File() {
            if (f) std::fclose(f);
        }
    } file{std::fopen(path, "rb")};
    if (!file.f) return fail(kReadCannotRead, errno, "opening the file failed");
    // A directory opens on some systems (macOS), and its size would read as a bad Splat count
    struct stat info;
    if (::fstat(::fileno(file.f), &info) != 0) return fail(kReadCannotRead, errno, "reading the file failed");
    if (!S_ISREG(info.st_mode)) return fail(kReadInvalid, 0, "not a regular file");
    if (std::fseek(file.f, 0, SEEK_END) != 0) return fail(kReadCannotRead, errno, "seeking in the file failed");
    const long size = std::ftell(file.f);
    if (size < 0 || std::fseek(file.f, 0, SEEK_SET) != 0)
        return fail(kReadCannotRead, errno, "seeking in the file failed");
    const auto bytes = static_cast<uint64_t>(size);
    if (bytes % sizeof(gen::Splat) != 0) {
        char message[kReadMessageSize];
        std::snprintf(message, sizeof message, "%llu bytes isn't a whole number of %zu-byte Splats",
                      static_cast<unsigned long long>(bytes), sizeof(gen::Splat));
        return fail(kReadInvalid, 0, message);
    }
    const uint64_t count = bytes / sizeof(gen::Splat);
    try {
        if (count > splats.max_size()) return fail(kReadOutOfMemory, 0, "out of memory");
        splats.assign(static_cast<std::size_t>(count), gen::Splat{});
    } catch (const std::bad_alloc&) {
        return fail(kReadOutOfMemory, 0, "out of memory");
    } catch (const std::length_error&) {
        return fail(kReadOutOfMemory, 0, "out of memory");
    }
    if (std::fread(splats.data(), sizeof(gen::Splat), splats.size(), file.f) != splats.size()) {
        const bool error = std::ferror(file.f) != 0;
        return fail(error ? kReadCannotRead : kReadInvalid, error ? errno : 0, "reading the splats failed");
    }
    return status;
}

}  // namespace splat_native
