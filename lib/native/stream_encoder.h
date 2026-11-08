// The stream encoder (docs/data-format.md, "Stream files"): Splat arrays and a per-group config to
// the bytes of a stream, [StreamHeader][base record x N][SH record x N], with a per-group report of
// the encoding's error. The format, its header check and its decode are lib/splat_stream.zgal's;
// the error report decodes through that same ZGAL function, so it measures what the shaders read.
//
// The entry points take plain structs and spans and return status codes, never throwing, so a C
// ABI wraps them as they are (convert.py's binding, the native viewer's PLY opener).
// The includer aliases `gen` to the same namespace in every translation unit of a binary: these
// functions are inline, so two aliases would be an ODR violation.
//
// Header-only, on top of the C++ that ZGAL generates for lib/splat_stream.zgal. The generated code
// lands in the namespace of the .zgal file importing it, so the includer names that namespace first:
//   #include "my_program.zgal.h"
//   namespace splat_native { namespace gen = ::my_program; }
//   #include "<relative path>/lib/native/stream_encoder.h"
#pragma once

#include "default_camera.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <span>

namespace splat_native {

static_assert(std::endian::native == std::endian::little,
              "the stream is little-endian, and the header is copied as it is in memory");
static_assert(sizeof(gen::StreamHeader) == gen::streamHeaderSize);

// ---------------------------------------------------------------------------------------------
// The config
// ---------------------------------------------------------------------------------------------
// How a quantized group's range is chosen: the data's own minimum and maximum (per component, or
// over every component for SH), or lo/hi as given (values outside are clamped, and the report shows
// the error). f32 and f16 have no range, and must say so.
inline constexpr uint32_t kRangeNone = 0;
inline constexpr uint32_t kRangeData = 1;
inline constexpr uint32_t kRangeExplicit = 2;

struct GroupConfig {
    uint32_t encoding;   // gen::encodingF32, encodingF16, encodingU16 or encodingU8
    uint32_t rangeMode;  // kRangeNone for f32 and f16; kRangeData or kRangeExplicit when quantized
    float lo[4];         // kRangeExplicit: per component, or lo[0] for SH; otherwise unused
    float hi[4];
};

struct EncodeConfig {
    uint32_t shDegree;                       // 0-3: the stream keeps bands 1 to shDegree
    GroupConfig groups[gen::groupCount];     // indexed by gen::groupPosition ... groupSh
};

// ---------------------------------------------------------------------------------------------
// The report
// ---------------------------------------------------------------------------------------------
inline constexpr uint32_t kEncodeOk = 0;
inline constexpr uint32_t kEncodeBadConfig = 1;      // `group` (groupCount for shDegree)
inline constexpr uint32_t kEncodeTooManySplats = 2;  // above gen::streamMaxSplats
inline constexpr uint32_t kEncodeBadOutputSize = 3;  // not streamSize of the layout
inline constexpr uint32_t kEncodeNonFinite = 4;      // an input value of `group` at `splat`
inline constexpr uint32_t kEncodeZeroRotation = 5;   // a quantized rotation of length 0 at `splat`
inline constexpr uint32_t kEncodeF16Overflow = 6;    // |value| rounds past f16's largest, 65504
inline constexpr uint32_t kEncodeOutOfMemory = 7;
inline constexpr uint32_t kEncodeRangeOverflow = 8;  // a range of `group` too wide for a float step (no splat)

// A group's largest errors over every splat and component, against the values encoded (for a
// quantized rotation, the normalized quaternion). "Activated" is in the space the renderer uses:
// exp of the scale, sigmoid of the opacity, the rotation's angle in radians (the angle between the
// two rotations), and the value itself for position, DC and SH.
struct GroupReport {
    double maxError;
    double maxActivatedError;
};

struct EncodeReport {
    uint32_t status;   // kEncodeOk or the first problem
    uint32_t group;    // the group it concerns, when it concerns one
    uint32_t splat;    // the splat it concerns, when it concerns one
    GroupReport groups[gen::groupCount];
};

// ---------------------------------------------------------------------------------------------
// Encodings
// ---------------------------------------------------------------------------------------------
// The IEEE half nearest to a finite float, ties to even; values from 65520 on (past the half way
// between 65504 and the next binade) become infinities, which the encoder refuses
inline uint32_t floatToHalf(float value) {
    const uint32_t x = std::bit_cast<uint32_t>(value);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t ax = x & 0x7FFFFFFFu;
    if (ax > 0x7F800000u) return sign | 0x7E00u;    // NaN
    if (ax >= 0x477FF000u) return sign | 0x7C00u;   // 65520 and up (and infinities)
    const uint32_t exponent = ax >> 23;
    const uint32_t mantissa = ax & 0x7FFFFFu;
    uint32_t h, rest, halfway;
    if (exponent < 113) {
        // Below 2^-14: a subnormal half, q 2^-24 with q = m 2^(e - 150) / 2^-24 = m / 2^(126 - e)
        const uint32_t shift = 126 - exponent;
        if (shift > 24) return sign;  // below 2^-25, which rounds to zero (ties to even, too)
        const uint32_t m = mantissa | 0x800000u;
        h = m >> shift;
        rest = m & ((1u << shift) - 1);
        halfway = 1u << (shift - 1);
    } else {
        h = ((exponent - 112) << 10) | (mantissa >> 13);
        rest = mantissa & 0x1FFFu;
        halfway = 0x1000u;
    }
    if (rest > halfway || (rest == halfway && (h & 1u))) ++h;  // a carry moves to the next binade
    return sign | h;
}

// The integer whose decode (gen::dequantize, in float32) is nearest the value, of 0 to 2^bits - 1
// (ties to the lower). The nearest in real arithmetic is the candidate; its neighbours are checked
// because the decode rounds.
inline uint32_t quantize(float value, float low, float stepSize, uint32_t bits) {
    const uint32_t qMax = (1u << bits) - 1;
    if (!(stepSize > 0.0f)) return 0;
    const double exact = (static_cast<double>(value) - low) / stepSize;
    const double clamped = std::clamp(std::nearbyint(exact), 0.0, static_cast<double>(qMax));
    const auto guess = static_cast<uint32_t>(clamped);
    uint32_t best = guess;
    double bestError = std::abs(static_cast<double>(gen::dequantize(guess, low, stepSize)) - value);
    for (const uint32_t q : {guess - 1, guess + 1}) {
        if (q > qMax) continue;  // guess - 1 wraps above qMax when guess is 0
        const double error = std::abs(static_cast<double>(gen::dequantize(q, low, stepSize)) - value);
        if (error < bestError || (error == bestError && q < best)) {
            best = q;
            bestError = error;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------------------------
// The layout
// ---------------------------------------------------------------------------------------------
// Checks a config, returning kEncodeOk or kEncodeBadConfig with the group (groupCount for shDegree)
inline uint32_t checkConfig(const EncodeConfig& config, uint32_t& group) {
    group = gen::groupCount;
    if (config.shDegree > 3) return kEncodeBadConfig;
    for (uint32_t g = 0; g < gen::groupCount; ++g) {
        group = g;
        const GroupConfig& c = config.groups[g];
        if (c.encoding >= gen::encodingCount) return kEncodeBadConfig;
        if (!gen::isQuantized(c.encoding)) {
            if (c.rangeMode != kRangeNone) return kEncodeBadConfig;
            continue;
        }
        if (c.rangeMode == kRangeData) continue;
        if (c.rangeMode != kRangeExplicit) return kEncodeBadConfig;
        const uint32_t ranges = g == gen::groupSh ? 1 : gen::groupComponents(g, config.shDegree);
        for (uint32_t r = 0; r < ranges; ++r)
            if (!std::isfinite(c.lo[r]) || !std::isfinite(c.hi[r]) || !(c.lo[r] <= c.hi[r]))
                return kEncodeBadConfig;
    }
    group = 0;
    return kEncodeOk;
}

// The header's layout for a config and splat count: magic to shOffset, and each field's encoding,
// components, offset and sharedRange (low, stepSize and the camera are zero; encodeStream fills
// them). The base record holds its fields largest component first (in group order within a size),
// so only the stride's rounding to 4 bytes pads it. Returns kEncodeOk, kEncodeBadConfig (with
// `group`) or kEncodeTooManySplats.
inline uint32_t streamLayout(const EncodeConfig& config, uint32_t splatCount, gen::StreamHeader& header,
                             uint32_t& group) {
    header = gen::StreamHeader{};
    if (const uint32_t status = checkConfig(config, group); status != kEncodeOk) return status;
    if (splatCount > gen::streamMaxSplats) return kEncodeTooManySplats;
    header.magic = gen::streamMagic;
    header.version = gen::streamVersion;
    header.splatCount = splatCount;
    header.shDegree = config.shDegree;
    for (uint32_t g = 0; g < gen::groupCount; ++g) {
        gen::StreamField& f = header.fields[g];
        f.encoding = config.groups[g].encoding;
        f.components = gen::groupComponents(g, config.shDegree);
        f.sharedRange = g == gen::groupSh ? 1 : 0;
    }
    uint32_t end = 0;
    for (const uint32_t size : {4u, 2u, 1u})
        for (uint32_t g = 0; g < gen::groupSh; ++g)
            if (gen::encodingSize(header.fields[g].encoding) == size) {
                header.fields[g].offset = end;
                end += header.fields[g].components * size;
            }
    header.baseStride = (end + 3) & ~3u;
    header.shStride = (gen::fieldEnd(header.fields[gen::groupSh]) + 3) & ~3u;
    header.baseOffset = gen::streamHeaderSize;
    header.shOffset = header.baseOffset + splatCount * header.baseStride;
    return kEncodeOk;
}

// The stream's size in bytes: the header, then every base record and every SH record
inline std::size_t streamSize(const gen::StreamHeader& header) {
    return static_cast<std::size_t>(header.shOffset) +
           static_cast<std::size_t>(header.splatCount) * header.shStride;
}

// ---------------------------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------------------------
// Splat s's value of a group's component c, the quantized rotation normalized
inline float groupValue(const gen::Splat& s, uint32_t group, uint32_t c, bool normalizeRotation) {
    switch (group) {
        case gen::groupPosition: return s.position[c];
        case gen::groupScale: return s.logScale[c];
        case gen::groupRotation: {
            if (!normalizeRotation) return s.rotation[c];
            const glm::dvec4 q(s.rotation);
            return static_cast<float>(q[c] / std::sqrt(glm::dot(q, q)));
        }
        case gen::groupOpacity: return s.opacity;
        case gen::groupDc: return s.sh[c];
        default: return s.sh[3 + c];
    }
}

inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// The angle in radians between the rotations of two quaternions (pi if either has length 0)
inline double rotationAngle(const glm::dvec4& a, const glm::dvec4& b) {
    const double na = std::sqrt(glm::dot(a, a)), nb = std::sqrt(glm::dot(b, b));
    if (!(na > 0.0) || !(nb > 0.0)) return 3.14159265358979323846;
    // q and -q are the same rotation, so v is b on a's side. With u . v = cos(phi), the rotation
    // angle is 2 phi, and atan2(|u - v|, |u + v|) = phi / 2: unlike 2 acos(u . v), which is
    // ill-conditioned near 1, it gives 0 for identical rotations (acos gave 4e-8 from rounding).
    const glm::dvec4 u = a / na;
    const glm::dvec4 v = (glm::dot(a, b) < 0.0 ? -b : b) / nb;
    return 4.0 * std::atan2(glm::length(u - v), glm::length(u + v));
}

// Encodes the splats into `out`, which must be exactly streamSize of their layout (streamLayout),
// and fills the report: kEncodeOk, or the first problem (and then `out` is unspecified). Values
// must be finite. The header gets the layout, each quantized field's range and the default camera
// (default_camera.h).
inline uint32_t encodeStream(std::span<const gen::Splat> splats, const EncodeConfig& config,
                             std::span<uint8_t> out, EncodeReport& report) noexcept {
    report = EncodeReport{};
    auto fail = [&](uint32_t status, uint32_t group, uint32_t splat) {
        report.status = status;
        report.group = group;
        report.splat = splat;
        return status;
    };
    if (splats.size() > gen::streamMaxSplats) return fail(kEncodeTooManySplats, 0, 0);
    const auto count = static_cast<uint32_t>(splats.size());
    gen::StreamHeader header;
    uint32_t badGroup = 0;
    if (const uint32_t status = streamLayout(config, count, header, badGroup); status != kEncodeOk)
        return fail(status, badGroup, 0);
    if (out.size() != streamSize(header)) return fail(kEncodeBadOutputSize, 0, 0);

    // Every input value finite; a quantized rotation of nonzero length
    for (uint32_t s = 0; s < count; ++s) {
        for (uint32_t g = 0; g < gen::groupCount; ++g)
            for (uint32_t c = 0; c < header.fields[g].components; ++c)
                if (!std::isfinite(groupValue(splats[s], g, c, false))) return fail(kEncodeNonFinite, g, s);
        if (gen::isQuantized(header.fields[gen::groupRotation].encoding)) {
            const glm::dvec4 q(splats[s].rotation);
            if (!(glm::dot(q, q) > 0.0)) return fail(kEncodeZeroRotation, gen::groupRotation, s);
        }
    }
    const bool normalizeRotation = gen::isQuantized(header.fields[gen::groupRotation].encoding);

    // The quantized fields' ranges: step = (high - low) / (2^bits - 1) in float32
    for (uint32_t g = 0; g < gen::groupCount; ++g) {
        gen::StreamField& f = header.fields[g];
        if (!gen::isQuantized(f.encoding)) continue;
        const GroupConfig& gc = config.groups[g];
        const uint32_t ranges = f.sharedRange == 1 ? (f.components > 0 ? 1 : 0) : f.components;
        for (uint32_t r = 0; r < ranges; ++r) {
            float low = gc.lo[r], high = gc.hi[r];
            if (gc.rangeMode == kRangeData) {
                low = INFINITY;
                high = -INFINITY;
                for (uint32_t s = 0; s < count; ++s)
                    for (uint32_t c = f.sharedRange == 1 ? 0 : r; c < (f.sharedRange == 1 ? f.components : r + 1); ++c) {
                        const float v = groupValue(splats[s], g, c, normalizeRotation);
                        low = std::min(low, v);
                        high = std::max(high, v);
                    }
                if (count == 0) low = high = 0.0f;
            }
            f.low[r] = low;
            f.stepSize[r] = (high - low) / static_cast<float>((1u << gen::encodingBits(f.encoding)) - 1);
            if (!std::isfinite(f.stepSize[r])) return fail(kEncodeRangeOverflow, g, 0);
        }
    }
    if (defaultCamera(splats, header.camera) != kDefaultCameraOk) return fail(kEncodeOutOfMemory, 0, 0);
    std::memset(out.data(), 0, out.size());  // padding is zero, so equal inputs give equal bytes
    std::memcpy(out.data(), &header, sizeof header);

    // The records, and the error of each decoded value (through the shaders' decode)
    for (uint32_t s = 0; s < count; ++s) {
        for (uint32_t g = 0; g < gen::groupCount; ++g) {
            const gen::StreamField& f = header.fields[g];
            if (f.components == 0) continue;
            const std::size_t record = g == gen::groupSh
                                           ? header.shOffset + static_cast<std::size_t>(s) * header.shStride
                                           : header.baseOffset + static_cast<std::size_t>(s) * header.baseStride;
            const uint32_t size = gen::encodingSize(f.encoding);
            std::array<float, 48> values{}, decoded{};
            for (uint32_t c = 0; c < f.components; ++c) {
                const float v = groupValue(splats[s], g, c, normalizeRotation);
                const uint32_t r = f.sharedRange == 1 ? 0 : c;
                uint32_t bits;
                if (f.encoding == gen::encodingF32) {
                    bits = std::bit_cast<uint32_t>(v);
                } else if (f.encoding == gen::encodingF16) {
                    bits = floatToHalf(v);
                    if ((bits & 0x7C00u) == 0x7C00u) return fail(kEncodeF16Overflow, g, s);
                } else {
                    bits = quantize(v, f.low[r], f.stepSize[r], gen::encodingBits(f.encoding));
                }
                const uint32_t offset = gen::componentOffset(f, c);
                for (uint32_t b = 0; b < size; ++b) out[record + offset + b] = static_cast<uint8_t>(bits >> (8 * b));
                // Decode as a shader does: the word holding the component, at its byte offset
                uint32_t word;
                std::memcpy(&word, &out[record + (offset & ~3u)], sizeof word);
                values[c] = v;
                decoded[c] = gen::decodeComponent(f, c, word, offset);
            }
            GroupReport& gr = report.groups[g];
            for (uint32_t c = 0; c < f.components; ++c) {
                const double v = values[c], d = decoded[c];
                gr.maxError = std::max(gr.maxError, std::abs(d - v));
                if (g == gen::groupScale) {
                    // exp overflows past 709: equal values have no error, and inf - inf is an infinite
                    // one (std::max would drop a NaN)
                    const double e = d == v ? 0.0 : std::abs(std::exp(d) - std::exp(v));
                    gr.maxActivatedError = std::max(gr.maxActivatedError, std::isnan(e) ? INFINITY : e);
                }
                else if (g == gen::groupOpacity) gr.maxActivatedError = std::max(gr.maxActivatedError, std::abs(sigmoid(d) - sigmoid(v)));
                else if (g != gen::groupRotation) gr.maxActivatedError = std::max(gr.maxActivatedError, std::abs(d - v));
            }
            if (g == gen::groupRotation)
                gr.maxActivatedError = std::max(
                    gr.maxActivatedError,
                    rotationAngle(glm::dvec4(values[0], values[1], values[2], values[3]),
                                  glm::dvec4(decoded[0], decoded[1], decoded[2], decoded[3])));
        }
    }
    return kEncodeOk;
}

}  // namespace splat_native
