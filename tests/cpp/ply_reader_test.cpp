// lib/native/ply_reader.h, the INRIA PLY reader: the layout (f_rest channel-major to sh[3k + c],
// rot_0..3 to rotation's x..w), lower SH degrees, other property types and orders, and each status.
// tools/tests/test_native.py tests the same reader through the C ABI and ctypes, more widely.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include "ply_reader_test.zgal.h"

namespace splat_native {
namespace gen = ::ply_reader_test;
}

#include "../../lib/native/ply_reader.h"

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

// A file in the temporary directory, removed when it goes
struct TempFile {
    fs::path path;
    explicit TempFile(const std::string& name)
        : path(fs::temp_directory_path() / ("ply_reader_test_" + std::to_string(::getpid()) + "_" + name)) {}
    ~TempFile() {
        std::error_code ignored;
        fs::remove(path, ignored);
    }
    void write(const std::string& bytes) const {
        std::ofstream out(path, std::ios::binary);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
};

// The INRIA properties with `perChannel` f_rest per channel, in the reference's order
std::vector<std::string> inriaNames(int perChannel) {
    std::vector<std::string> names = {"x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"};
    for (int i = 0; i < 3 * perChannel; ++i) names.push_back("f_rest_" + std::to_string(i));
    for (const char* n : {"opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"})
        names.push_back(n);
    return names;
}

std::string floatHeader(const std::vector<std::string>& names, uint64_t count,
                        const std::string& after = "") {
    std::string h = "ply\nformat binary_little_endian 1.0\ncomment test\nelement vertex " +
                    std::to_string(count) + "\n";
    for (const auto& n : names) h += "property float " + n + "\n";
    return h + after + "end_header\n";
}

template <typename T>
void append(std::string& bytes, T value) {
    bytes.append(reinterpret_cast<const char*>(&value), sizeof(T));
}

struct Read {
    splat_native::ReadStatus status;
    std::vector<ply_reader_test::Splat> splats;
    uint32_t shDegree = 99;
};

Read read(const TempFile& file) {
    Read r;
    r.status = splat_native::readPly(file.path.c_str(), r.splats, r.shDegree);
    return r;
}

}  // namespace

TEST_CASE("degree 3 in the reference's order: f_rest is channel-major, rot_0..3 is rotation's x..w") {
    const auto names = inriaNames(15);
    TempFile file("degree3.ply");
    std::string bytes = floatHeader(names, 2);
    for (int v = 0; v < 2; ++v)
        for (std::size_t i = 0; i < names.size(); ++i) append(bytes, 1000.0f * v + static_cast<float>(i));
    file.write(bytes);
    const Read r = read(file);
    REQUIRE(r.status.code == splat_native::kReadOk);
    CHECK(r.shDegree == 3);
    REQUIRE(r.splats.size() == 2);
    for (int v = 0; v < 2; ++v) {
        const auto& s = r.splats[v];
        const float o = 1000.0f * v;
        CHECK(s.position == glm::vec3(o + 0, o + 1, o + 2));  // 3-5 are the normals, ignored
        for (int c = 0; c < 3; ++c) {
            CHECK(s.sh[c] == o + 6 + c);
            for (int k = 1; k < 16; ++k) CHECK(s.sh[3 * k + c] == o + 9 + 15 * c + k - 1);
        }
        CHECK(s.opacity == o + 54);
        CHECK(s.logScale == glm::vec3(o + 55, o + 56, o + 57));
        CHECK(s.rotation == glm::vec4(o + 58, o + 59, o + 60, o + 61));
        CHECK(s.zgalPadding0_[0] == 0);
    }
}

TEST_CASE("degree 1, reordered, other types, a face element after: the other bands are zero") {
    // The properties reversed, each a double but f_dc_1 a uchar and rot_3 a short
    auto names = inriaNames(3);
    std::vector<std::string> reversed(names.rbegin(), names.rend());
    std::string header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\n";
    std::string body;
    for (const auto& n : reversed) {
        const double value = static_cast<double>(&n - reversed.data()) + 0.25;
        if (n == "f_dc_1") {
            header += "property uchar " + n + "\n";
            append<uint8_t>(body, 200);
        } else if (n == "rot_3") {
            header += "property short " + n + "\n";
            append<int16_t>(body, -300);
        } else {
            header += "property double " + n + "\n";
            append<double>(body, value);
        }
    }
    header += "element face 0\nproperty list uchar int vertex_indices\nend_header\n";
    TempFile file("degree1.ply");
    file.write(header + body);
    const Read r = read(file);
    REQUIRE(r.status.code == splat_native::kReadOk);
    CHECK(r.shDegree == 1);
    REQUIRE(r.splats.size() == 1);
    const auto& s = r.splats[0];
    // reversed[i] has the value i + 0.25: name j of the 26 in order has 25 - j + 0.25
    auto valueOf = [&](int j) { return static_cast<float>(25 - j + 0.25); };
    CHECK(s.position == glm::vec3(valueOf(0), valueOf(1), valueOf(2)));
    CHECK(s.sh[0] == valueOf(6));
    CHECK(s.sh[1] == 200.0f);
    CHECK(s.sh[2] == valueOf(8));
    for (int c = 0; c < 3; ++c)
        for (int k = 1; k <= 3; ++k) CHECK(s.sh[3 * k + c] == valueOf(9 + 3 * c + k - 1));
    for (int i = 12; i < 48; ++i) CHECK(s.sh[i] == 0.0f);
    CHECK(s.rotation.w == -300.0f);
}

TEST_CASE("degree 0 and no vertices") {
    TempFile file("empty.ply");
    file.write(floatHeader(inriaNames(0), 0));
    const Read r = read(file);
    CHECK(r.status.code == splat_native::kReadOk);
    CHECK(r.shDegree == 0);
    CHECK(r.splats.empty());
}

TEST_CASE("errors: each status, with a message") {
    auto invalid = [](const std::string& name, const std::string& bytes, const char* message) {
        TempFile file(name);
        file.write(bytes);
        const Read r = read(file);
        CAPTURE(name);
        CHECK(r.status.code == splat_native::kReadInvalid);
        CHECK(std::strstr(r.status.message, message) != nullptr);
    };
    auto names = inriaNames(15);
    invalid("ascii.ply", "ply\nformat ascii 1.0\nelement vertex 0\nend_header\n", "binary_little_endian");
    invalid("not.ply", "PLY\n", "not a PLY file");
    invalid("noend.ply", "ply\nformat binary_little_endian 1.0\n", "no end_header");
    std::vector<std::string> noOpacity;
    for (const auto& n : names)
        if (n != "opacity") noOpacity.push_back(n);
    invalid("missing.ply", floatHeader(noOpacity, 0), "missing properties opacity");
    invalid("degree.ply", floatHeader(inriaNames(2), 0), "6 f_rest properties");
    invalid("truncated.ply", floatHeader(names, 2) + std::string(62 * 4 + 3, '\0'), "2 vertices declared, 1 present");

    TempFile missing("missing-file.ply");
    const Read r = read(missing);
    CHECK(r.status.code == splat_native::kReadCannotRead);
    CHECK(r.status.systemError == ENOENT);
}
