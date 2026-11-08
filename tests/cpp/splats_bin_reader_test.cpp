// lib/native/splats_bin_reader.h, the splats.bin reader: the bytes come back as they were written
// (padding too), an empty file is no splats, and each status. tools/tests/test_native.py tests the
// same reader through the C ABI and ctypes.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include "splats_bin_reader_test.zgal.h"

namespace splat_native {
namespace gen = ::splats_bin_reader_test;
}

#include "../../lib/native/splats_bin_reader.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

namespace fs = std::filesystem;
using Splat = splats_bin_reader_test::Splat;

// A file in the temporary directory, removed when it goes
struct TempFile {
    fs::path path;
    explicit TempFile(const std::string& name)
        : path(fs::temp_directory_path() / ("splats_bin_reader_test_" + std::to_string(::getpid()) + "_" + name)) {}
    ~TempFile() {
        std::error_code ignored;
        fs::remove(path, ignored);
    }
    void write(const std::string& bytes) const {
        std::ofstream out(path, std::ios::binary);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
};

struct Read {
    splat_native::ReadStatus status;
    std::vector<Splat> splats;
};

Read read(const TempFile& file) {
    Read r;
    r.splats.resize(3);  // replaced by what the file holds
    r.status = splat_native::readSplatsBin(file.path.c_str(), r.splats);
    return r;
}

}  // namespace

TEST_CASE("a splats.bin's bytes come back as written") {
    std::string bytes(2 * sizeof(Splat), '\0');
    for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<char>(i * 7 + 3);
    TempFile file("two.bin");
    file.write(bytes);
    const Read r = read(file);
    REQUIRE(r.status.code == splat_native::kReadOk);
    REQUIRE(r.splats.size() == 2);
    CHECK(std::memcmp(r.splats.data(), bytes.data(), bytes.size()) == 0);
}

TEST_CASE("an empty splats.bin holds no splats") {
    TempFile file("empty.bin");
    file.write("");
    const Read r = read(file);
    CHECK(r.status.code == splat_native::kReadOk);
    CHECK(r.splats.empty());
}

TEST_CASE("the reader refuses a partial Splat and a missing file") {
    TempFile file("partial.bin");
    file.write(std::string(sizeof(Splat) + 4, '\0'));
    const Read r = read(file);
    CHECK(r.status.code == splat_native::kReadInvalid);
    CHECK(std::string(r.status.message) == "244 bytes isn't a whole number of 240-byte Splats");

    TempFile missing("missing.bin");
    const Read m = read(missing);
    CHECK(m.status.code == splat_native::kReadCannotRead);
    CHECK(m.status.systemError == ENOENT);
}

TEST_CASE("the reader refuses a directory") {
    TempFile directory("directory.bin");
    fs::create_directory(directory.path);
    const Read r = read(directory);
    // It opens on macOS (then it isn't a regular file) and fails to open on Linux (EISDIR)
    if (r.status.code == splat_native::kReadInvalid)
        CHECK(std::string(r.status.message) == "not a regular file");
    else
        CHECK(r.status.code == splat_native::kReadCannotRead);
}
