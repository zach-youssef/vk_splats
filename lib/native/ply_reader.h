// The INRIA PLY reader (docs/data-format.md, "INRIA PLY"): a PLY file to Splat structs, as the
// programs' buffers hold them. The converter reads PLYs through it (lib/native/c_api.cpp, called by
// tools/splatdata/native.py), and so can any C++ program.
//
// The format is the reference code's: binary little-endian, with the vertex element first and its
// properties x y z (nx ny nz) f_dc_0..2 f_rest_0..(3m - 1) opacity scale_0..2 rot_0..3, where m is
// 0, 3, 8 or 15 (SH degree 0-3). Properties may be in any order and of any scalar type (each is
// converted to double, then to float); other vertex properties (the normals) and the elements after
// the vertices (faces) are ignored. f_rest is channel-major: coefficient k (1 to m) of channel c is
// f_rest_{c m + k - 1}, which lands in Splat::sh[3k + c]; the bands the file doesn't have are zero.
// The rotation (w, x, y, z) = rot_0..3 lands in Splat::rotation's (.x, .y, .z, .w), as stored.
//
// The entry point takes plain arguments and returns a status, never throwing, so a C ABI wraps it
// as it is. It's inline in `splat_native`, so a linked binary must alias `gen` to one namespace in
// every translation unit (otherwise it's an ODR violation).
//
// Header-only, on top of the C++ that ZGAL generates for a module importing lib/splat_math.zgal
// (its Splat). The generated code lands in the namespace of the .zgal file importing it, so the
// includer names that namespace first:
//   #include "my_program.zgal.h"
//   namespace splat_native { namespace gen = ::my_program; }
//   #include "<relative path>/lib/native/ply_reader.h"
#pragma once

#include "read_status.h"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace splat_native {

static_assert(std::endian::native == std::endian::little,
              "the reader loads the PLY's little-endian values as they are in memory");

namespace ply_detail {

// A PLY scalar type: its size and how a value converts (to double, then float)
enum class Scalar { I8, U8, I16, U16, I32, U32, F32, F64 };

inline bool scalarType(std::string_view name, Scalar& type) {
    struct Name {
        std::string_view name;
        Scalar type;
    };
    static constexpr Name kNames[] = {
        {"char", Scalar::I8},   {"int8", Scalar::I8},     {"uchar", Scalar::U8},  {"uint8", Scalar::U8},
        {"short", Scalar::I16}, {"int16", Scalar::I16},   {"ushort", Scalar::U16}, {"uint16", Scalar::U16},
        {"int", Scalar::I32},   {"int32", Scalar::I32},   {"uint", Scalar::U32},  {"uint32", Scalar::U32},
        {"float", Scalar::F32}, {"float32", Scalar::F32}, {"double", Scalar::F64}, {"float64", Scalar::F64},
    };
    for (const Name& n : kNames)
        if (n.name == name) {
            type = n.type;
            return true;
        }
    return false;
}

inline std::size_t scalarSize(Scalar type) {
    switch (type) {
        case Scalar::I8: case Scalar::U8: return 1;
        case Scalar::I16: case Scalar::U16: return 2;
        case Scalar::I32: case Scalar::U32: case Scalar::F32: return 4;
        case Scalar::F64: return 8;
    }
    return 0;
}

template <typename T>
inline double load(const unsigned char* p) {
    T value;
    std::memcpy(&value, p, sizeof(T));
    return static_cast<double>(value);
}

inline float loadScalar(const unsigned char* p, Scalar type) {
    double value = 0.0;
    switch (type) {
        case Scalar::I8: value = load<int8_t>(p); break;
        case Scalar::U8: value = load<uint8_t>(p); break;
        case Scalar::I16: value = load<int16_t>(p); break;
        case Scalar::U16: value = load<uint16_t>(p); break;
        case Scalar::I32: value = load<int32_t>(p); break;
        case Scalar::U32: value = load<uint32_t>(p); break;
        case Scalar::F32: value = load<float>(p); break;
        case Scalar::F64: value = load<double>(p); break;
    }
    return static_cast<float>(value);
}

struct Property {
    std::string name;
    Scalar type;
    std::size_t offset;  // in the vertex
};

// The properties a Splat is made of, in this order (kSlots): x y z f_dc_0..2 opacity scale_0..2
// rot_0..3, then the f_rest
inline constexpr int kRequired = 14;
inline constexpr const char* kRequiredNames[kRequired] = {
    "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
    "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
};
inline constexpr int kMaxRest = 45;
// The most bytes of vertices read at once
inline constexpr std::size_t kPlyChunkBytes = std::size_t{1} << 20;

struct Failure {
    uint32_t code;
    int systemError;
    std::string message;
};

// A FILE that closes itself
struct File {
    std::FILE* f = nullptr;
    ~File() {
        if (f) std::fclose(f);
    }
};

inline std::vector<std::string_view> tokens(std::string_view line) {
    std::vector<std::string_view> out;
    std::size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t' || line[i] == '\r')) ++i;
        std::size_t j = i;
        while (j < line.size() && !(line[j] == ' ' || line[j] == '\t' || line[j] == '\r')) ++j;
        if (j > i) out.push_back(line.substr(i, j - i));
        i = j;
    }
    return out;
}

// One header line, without its '\n': false at the end of the file. Fails on a read error or a byte
// that isn't ASCII (the header is text).
inline bool readLine(std::FILE* f, std::string& line, Failure& failure) {
    line.clear();
    for (;;) {
        int c = std::fgetc(f);
        if (c == EOF) {
            if (std::ferror(f)) {
                failure = {kReadCannotRead, errno, "reading the header failed"};
                return false;
            }
            return !line.empty();
        }
        if (c == '\n') return true;
        if (c > 0x7F) {
            failure = {kReadInvalid, 0, "the header isn't ASCII"};
            return false;
        }
        line.push_back(static_cast<char>(c));
    }
}

inline std::string joined(std::string_view a, std::string_view b) {
    std::string s(a);
    s += b;
    return s;
}

// readPly's body: throws only std::bad_alloc or std::length_error (out of memory)
inline bool readPlyOrFail(const char* path, std::vector<gen::Splat>& splats, uint32_t& shDegree,
                          Failure& failure) {
    File file;
    errno = 0;
    file.f = std::fopen(path, "rb");
    if (!file.f) {
        failure = {kReadCannotRead, errno, "opening the file failed"};
        return false;
    }
    std::FILE* f = file.f;

    std::string line;
    failure = {kReadInvalid, 0, ""};
    if (!readLine(f, line, failure)) {
        if (failure.message.empty()) failure.message = "not a PLY file";
        return false;
    }
    {
        auto t = tokens(line);
        if (t.size() != 1 || t[0] != "ply") {
            failure.message = "not a PLY file";
            return false;
        }
    }

    // The header: the format, the vertex count and the vertex element's properties
    enum class Element { None, Vertex, After } element = Element::None;
    std::string format;
    bool haveCount = false;
    uint64_t count = 0;
    std::vector<Property> properties;
    std::unordered_set<std::string> names;  // the properties' names, so a long header checks fast
    std::size_t stride = 0;
    for (;;) {
        failure.message.clear();
        if (!readLine(f, line, failure)) {
            if (failure.message.empty()) failure.message = "no end_header";
            return false;
        }
        auto t = tokens(line);
        if (t.empty() || t[0] == "comment" || t[0] == "obj_info") continue;
        if (t[0] == "end_header") break;
        if (t[0] == "format") {
            if (t.size() < 2) {
                failure.message = "a format line without a format";
                return false;
            }
            format = t[1];
        } else if (t[0] == "element") {
            if (t.size() < 3) {
                failure.message = joined("a malformed element line: ", line);
                return false;
            }
            if (element == Element::Vertex) {
                element = Element::After;  // Later elements (faces) are ignored
            } else if (element == Element::None) {
                if (t[1] != "vertex") {
                    failure.message = joined(joined("the first element is ", t[1]), ", not vertex");
                    return false;
                }
                auto [end, error] = std::from_chars(t[2].data(), t[2].data() + t[2].size(), count);
                if (error != std::errc() || end != t[2].data() + t[2].size()) {
                    failure.message = joined(joined("the vertex count ", t[2]), " isn't a count");
                    return false;
                }
                element = Element::Vertex;
                haveCount = true;
            }
        } else if (t[0] == "property" && element == Element::Vertex) {
            if (t.size() >= 2 && t[1] == "list") {
                failure.message = joined(joined("list property ", t.back()), " in vertex");
                return false;
            }
            if (t.size() < 3) {
                failure.message = joined("a malformed property line: ", line);
                return false;
            }
            Scalar type;
            if (!scalarType(t[1], type)) {
                failure.message = joined(joined(joined("property ", t[2]), " has unknown type "), t[1]);
                return false;
            }
            if (!names.insert(std::string(t[2])).second) {
                failure.message = joined(joined("property ", t[2]), " appears twice");
                return false;
            }
            properties.push_back({std::string(t[2]), type, stride});
            stride += scalarSize(type);
        }
    }
    if (format != "binary_little_endian") {
        failure.message = joined(joined("format ", format.empty() ? "missing" : format),
                                 "; only binary_little_endian is supported");
        return false;
    }
    if (!haveCount) {
        failure.message = "no vertex element";
        return false;
    }

    // The properties a Splat needs: slots 0-13 the required ones (kRequiredNames), 14 on f_rest
    const Property* slots[kRequired + kMaxRest] = {};
    std::string missing;
    for (int i = 0; i < kRequired; ++i) {
        for (const Property& p : properties)
            if (p.name == kRequiredNames[i]) slots[i] = &p;
        if (!slots[i]) missing += (missing.empty() ? "" : ", ") + std::string(kRequiredNames[i]);
    }
    if (!missing.empty()) {
        failure.message = "missing properties " + missing;
        return false;
    }
    int restCount = 0;
    for (const Property& p : properties) {
        if (p.name.rfind("f_rest_", 0) != 0) continue;
        std::string_view digits = std::string_view(p.name).substr(7);
        int k = -1;
        auto [end, error] = std::from_chars(digits.data(), digits.data() + digits.size(), k);
        // f_rest_<k> with k in 0..44 written plainly; anything else counts as a gap in the numbering
        if (error != std::errc() || end != digits.data() + digits.size() || digits.empty() ||
            (digits.size() > 1 && digits[0] == '0') || k < 0 || k >= kMaxRest) {
            failure.message = "f_rest properties aren't numbered 0 to n - 1";
            return false;
        }
        slots[kRequired + k] = &p;
        ++restCount;
    }
    for (int k = 0; k < restCount; ++k)
        if (!slots[kRequired + k]) {
            failure.message = "f_rest properties aren't numbered 0 to n - 1";
            return false;
        }
    const int perChannel = restCount / 3;
    if (restCount % 3 != 0 || (perChannel != 0 && perChannel != 3 && perChannel != 8 && perChannel != 15)) {
        failure.message = std::to_string(restCount) + " f_rest properties: not SH of degree 0-3";
        return false;
    }
    shDegree = perChannel == 0 ? 0 : perChannel == 3 ? 1 : perChannel == 8 ? 2 : 3;

    // The vertices must all be there before they're allocated, so a header can't ask for more
    // memory than the file holds
    const long dataStart = std::ftell(f);
    if (dataStart < 0 || std::fseek(f, 0, SEEK_END) != 0) {
        failure = {kReadCannotRead, errno, "seeking in the file failed"};
        return false;
    }
    const long fileEnd = std::ftell(f);
    if (fileEnd < 0 || std::fseek(f, dataStart, SEEK_SET) != 0) {
        failure = {kReadCannotRead, errno, "seeking in the file failed"};
        return false;
    }
    const uint64_t available = static_cast<uint64_t>(fileEnd - dataStart);
    const uint64_t present = stride == 0 ? count : available / stride;
    if (present < count) {
        failure.message = std::to_string(count) + " vertices declared, " + std::to_string(present) + " present";
        return false;
    }
    if (count > splats.max_size()) throw std::length_error("too many splats");
    splats.assign(static_cast<std::size_t>(count), gen::Splat{});

    // In chunks of at most 1 MiB (or one vertex), so the file's bytes are never all in memory and a
    // file of few vertices but many properties allocates no more than it holds
    const std::size_t kChunk =
        std::min<std::size_t>(splats.size(), std::max<std::size_t>(1, kPlyChunkBytes / stride));
    std::vector<unsigned char> chunk(kChunk * stride);
    for (std::size_t first = 0; first < splats.size(); first += kChunk) {
        const std::size_t n = std::min(kChunk, splats.size() - first);
        if (std::fread(chunk.data(), stride, n, f) != n) {
            const bool error = std::ferror(f) != 0;
            failure = {error ? kReadCannotRead : kReadInvalid, error ? errno : 0, "reading the vertices failed"};
            return false;
        }
        for (std::size_t i = 0; i < n; ++i) {
            const unsigned char* v = chunk.data() + i * stride;
            auto value = [&](int slot) { return loadScalar(v + slots[slot]->offset, slots[slot]->type); };
            gen::Splat& s = splats[first + i];
            s.position = {value(0), value(1), value(2)};
            for (int c = 0; c < 3; ++c) s.sh[c] = value(3 + c);
            s.opacity = value(6);
            s.logScale = {value(7), value(8), value(9)};
            s.rotation = {value(10), value(11), value(12), value(13)};
            for (int c = 0; c < 3; ++c)
                for (int k = 1; k <= perChannel; ++k)
                    s.sh[3 * k + c] = value(kRequired + c * perChannel + k - 1);
        }
    }
    return true;
}

}  // namespace ply_detail

// Reads the PLY at `path` (see the top of this file) into `splats` (replacing what it held) and its
// SH degree (0-3) into `shDegree`. Never throws: kReadOk, or an error with `splats` and `shDegree`
// unspecified (kReadCannotRead with the errno, kReadInvalid, kReadOutOfMemory) and the message set.
inline ReadStatus readPly(const char* path, std::vector<gen::Splat>& splats, uint32_t& shDegree) noexcept {
    ReadStatus status;
    auto fail = [&](uint32_t code, int systemError, const char* message) {
        status.code = code;
        status.systemError = systemError;
        std::snprintf(status.message, sizeof(status.message), "%s", message);
        return status;
    };
    try {
        ply_detail::Failure failure{kReadOk, 0, {}};
        if (!ply_detail::readPlyOrFail(path, splats, shDegree, failure))
            return fail(failure.code, failure.systemError, failure.message.c_str());
        return status;
    } catch (const std::bad_alloc&) {
        return fail(kReadOutOfMemory, 0, "out of memory");
    } catch (const std::length_error&) {
        return fail(kReadOutOfMemory, 0, "out of memory");
    }
}

}  // namespace splat_native
