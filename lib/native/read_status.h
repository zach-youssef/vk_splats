// The result of lib/native/'s file readers (ply_reader.h, splats_bin_reader.h): a code, the errno
// when the file couldn't be read, and a message. A plain struct, so a C ABI passes it on as it is.
#pragma once

#include <cstddef>
#include <cstdint>

namespace splat_native {

inline constexpr uint32_t kReadOk = 0;
inline constexpr uint32_t kReadCannotRead = 1;    // opening or reading the file failed (systemError)
inline constexpr uint32_t kReadInvalid = 2;       // not a file this reader supports (message says why)
inline constexpr uint32_t kReadOutOfMemory = 3;   // the splats couldn't be allocated

inline constexpr std::size_t kReadMessageSize = 256;

struct ReadStatus {
    uint32_t code = kReadOk;
    int systemError = 0;               // kReadCannotRead: the errno of the call that failed
    char message[kReadMessageSize]{};  // anything but kReadOk: what went wrong (truncated to fit)
};

}  // namespace splat_native
