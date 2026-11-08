// The stream format (lib/splat_stream.zgal, through the C++ ZGAL generates for it in
// splat_stream_test.zgal.h) and its encoder (lib/native/): the header's layout, checkStreamHeader's
// cases, the f16 conversions, round trips of every encoding and SH degree through a decode written
// independently here, the encoder's report and refusals, and the default camera in the header
// (default_camera_test.cpp tests the camera itself).
//
// Exactness: f32 is a bit copy and f16 a correctly rounded conversion, so both are checked bit for
// bit. A quantized value must decode to low + stepSize q in float32; whether that product and sum
// are fused into one rounding is the compiler's choice (FMA contraction), so either result passes.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include "splat_stream_test.zgal.h"
namespace splat_native { namespace gen = ::splat_stream_test; }
#include "../../lib/native/stream_encoder.h"

#include "test_random.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace ss = splat_stream_test;
namespace sn = splat_native;

namespace {

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------
sn::GroupConfig f32() { return {ss::encodingF32, sn::kRangeNone, {}, {}}; }
sn::GroupConfig f16() { return {ss::encodingF16, sn::kRangeNone, {}, {}}; }
sn::GroupConfig quantizedData(uint32_t encoding) { return {encoding, sn::kRangeData, {}, {}}; }
sn::GroupConfig quantizedExplicit(uint32_t encoding, float lo, float hi) {
    return {encoding, sn::kRangeExplicit, {lo, lo, lo, lo}, {hi, hi, hi, hi}};
}

// The plan's presets (tools/stream-configs will hold them)
sn::EncodeConfig fullConfig(uint32_t shDegree) { return {shDegree, {f32(), f32(), f32(), f32(), f32(), f32()}}; }
sn::EncodeConfig f16Config(uint32_t shDegree) { return {shDegree, {f32(), f16(), f16(), f16(), f16(), f16()}}; }
sn::EncodeConfig webConfig(uint32_t shDegree) {
    return {shDegree,
            {f32(), f16(), quantizedExplicit(ss::encodingU8, -1.0f, 1.0f), quantizedData(ss::encodingU8), f16(),
             quantizedData(ss::encodingU8)}};
}
// Every quantized encoding, with data and explicit ranges
sn::EncodeConfig quantizedConfig(uint32_t shDegree) {
    return {shDegree,
            {quantizedData(ss::encodingU16), quantizedData(ss::encodingU8),
             quantizedExplicit(ss::encodingU16, -1.0f, 1.0f), quantizedExplicit(ss::encodingU8, -6.0f, 6.0f),
             quantizedData(ss::encodingU16), quantizedExplicit(ss::encodingU16, -0.5f, 0.5f)}};
}

std::vector<ss::Splat> randomSplats(uint32_t seed, int n) {
    TestRandom rng(seed);
    std::vector<ss::Splat> splats(n);
    for (ss::Splat& s : splats) {
        s.position = glm::vec3{float(rng.uniform(-20, 20)), float(rng.uniform(-5, 5)), float(rng.uniform(0, 30))};
        s.opacity = float(rng.uniform(-6, 6));
        s.logScale = glm::vec3{float(rng.uniform(-7, 1)), float(rng.uniform(-7, 1)), float(rng.uniform(-7, 1))};
        s.rotation = glm::vec4{float(rng.uniform(-2, 2)), float(rng.uniform(-2, 2)), float(rng.uniform(-2, 2)),
                               float(rng.uniform(-2, 2))};
        for (float& c : s.sh) c = float(rng.uniform(-0.6, 0.6));
        for (int c = 0; c < 3; ++c) s.sh[c] = float(rng.uniform(-2, 2));
    }
    return splats;
}

struct Encoded {
    uint32_t status;
    sn::EncodeReport report;
    ss::StreamHeader header;
    std::vector<uint8_t> bytes;
};

Encoded encode(const std::vector<ss::Splat>& splats, const sn::EncodeConfig& config) {
    Encoded e{};
    uint32_t group = 0;
    REQUIRE(sn::streamLayout(config, uint32_t(splats.size()), e.header, group) == sn::kEncodeOk);
    e.bytes.assign(sn::streamSize(e.header), 0xAB);  // nonzero, so unwritten padding shows
    e.status = sn::encodeStream(splats, config, e.bytes, e.report);
    std::memcpy(&e.header, e.bytes.data(), sizeof e.header);
    return e;
}

// An IEEE half's value, from its definition
float halfValue(uint32_t h) {
    const int e = int(h >> 10) & 31;
    const int m = int(h & 1023);
    const double sign = (h & 0x8000) ? -1.0 : 1.0;
    if (e == 31) return m ? std::numeric_limits<float>::quiet_NaN() : float(sign * INFINITY);
    if (e == 0) return float(sign * std::ldexp(double(m), -24));
    return float(sign * std::ldexp(1.0 + m / 1024.0, e - 15));
}

// Splat s's component c of a group, as the encoder takes it (quantized rotations normalized)
float inputValue(const ss::Splat& s, uint32_t group, uint32_t c, bool normalizeRotation) {
    switch (group) {
        case ss::groupPosition: return s.position[c];
        case ss::groupScale: return s.logScale[c];
        case ss::groupRotation: {
            if (!normalizeRotation) return s.rotation[c];
            const double n = std::sqrt(double(s.rotation.x) * s.rotation.x + double(s.rotation.y) * s.rotation.y +
                                       double(s.rotation.z) * s.rotation.z + double(s.rotation.w) * s.rotation.w);
            return float(s.rotation[c] / n);
        }
        case ss::groupOpacity: return s.opacity;
        case ss::groupDc: return s.sh[c];
        default: return s.sh[3 + c];
    }
}

// A component's bits from the stream's bytes, little-endian, independent of componentBits
uint32_t rawBits(const std::vector<uint8_t>& bytes, std::size_t at, uint32_t size) {
    uint32_t bits = 0;
    for (uint32_t b = 0; b < size; ++b) bits |= uint32_t(bytes[at + b]) << (8 * b);
    return bits;
}

// Checks every component of every splat against the input: f32 and f16 bit for bit, quantized as
// low + stepSize q, and checks decodeComponent (the shaders' decode) agrees
void checkRoundTrip(const std::vector<ss::Splat>& splats, const sn::EncodeConfig& config) {
    const Encoded e = encode(splats, config);
    REQUIRE(e.status == sn::kEncodeOk);
    const ss::StreamHeader& h = e.header;
    CHECK(ss::checkStreamHeader(h) == ss::streamOk);
    CHECK(h.splatCount == splats.size());
    CHECK(h.shDegree == config.shDegree);
    CHECK(e.bytes.size() == std::size_t(h.shOffset) + std::size_t(h.splatCount) * h.shStride);
    const bool normalizeRotation = ss::isQuantized(config.groups[ss::groupRotation].encoding);
    for (uint32_t g = 0; g < ss::groupCount; ++g) {
        const ss::StreamField& f = h.fields[g];
        CAPTURE(g);
        CHECK(f.encoding == config.groups[g].encoding);
        CHECK(f.components == ss::groupComponents(g, config.shDegree));
        double maxStep = 0.0;
        for (uint32_t s = 0; s < splats.size(); ++s) {
            const std::size_t record = g == ss::groupSh ? h.shOffset + std::size_t(s) * h.shStride
                                                        : h.baseOffset + std::size_t(s) * h.baseStride;
            for (uint32_t c = 0; c < f.components; ++c) {
                CAPTURE(s);
                CAPTURE(c);
                const float v = inputValue(splats[s], g, c, normalizeRotation);
                const uint32_t size = ss::encodingSize(f.encoding);
                const std::size_t at = record + f.offset + std::size_t(c) * size;
                const uint32_t bits = rawBits(e.bytes, at, size);
                float expected;
                if (f.encoding == ss::encodingF32) {
                    CHECK(bits == std::bit_cast<uint32_t>(v));
                    expected = v;
                } else if (f.encoding == ss::encodingF16) {
                    // The nearest half (ties aren't drawn by random data; the f16 cases check them)
                    expected = halfValue(bits);
                    const float below = halfValue(bits == 0x8000 || bits == 0 ? bits : bits - 1);
                    const float above = halfValue(bits + 1);
                    CHECK(std::abs(double(expected) - v) <= std::abs(double(below) - v));
                    CHECK(std::abs(double(expected) - v) <= std::abs(double(above) - v));
                } else {
                    const uint32_t r = f.sharedRange ? 0 : c;
                    const float low = f.low[r], stepSize = f.stepSize[r];
                    CHECK(bits <= (1u << ss::encodingBits(f.encoding)) - 1);
                    const float product = stepSize * float(bits);
                    const float unfused = low + product;
                    const float fused = std::fma(stepSize, float(bits), low);
                    expected = ss::dequantize(bits, low, stepSize);
                    CHECK((expected == unfused || expected == fused));
                    maxStep = std::max(maxStep, double(stepSize));
                    // Data ranges and in-range values: within half a step (plus the decode's rounding)
                    const float lo = low, hi = low + stepSize * float((1u << ss::encodingBits(f.encoding)) - 1);
                    if (v >= lo && v <= hi)
                        CHECK(std::abs(double(expected) - v) <= 0.5 * stepSize + 4e-7 * std::max(1.0f, std::abs(v)));
                }
                // The shaders' decode, from the word holding the component
                uint32_t word;
                std::memcpy(&word, &e.bytes[at & ~std::size_t(3)], sizeof word);
                const float decoded = ss::decodeComponent(f, c, word, uint32_t(at));
                CHECK(std::bit_cast<uint32_t>(decoded) == std::bit_cast<uint32_t>(expected));
            }
        }
        if (f.encoding == ss::encodingF32) {
            CHECK(e.report.groups[g].maxError == 0.0);
            CHECK(e.report.groups[g].maxActivatedError == 0.0);
        }
        if (ss::isQuantized(f.encoding) && config.groups[g].rangeMode == sn::kRangeData)
            CHECK(e.report.groups[g].maxError <= 0.5 * maxStep * (1 + 1e-5) + 1e-6);
    }
    // Padding is zero: the bytes no field covers, in every record
    for (uint32_t s = 0; s < splats.size(); ++s) {
        std::vector<bool> covered(h.baseStride, false);
        for (uint32_t g = 0; g < ss::groupSh; ++g)
            for (uint32_t b = h.fields[g].offset; b < ss::fieldEnd(h.fields[g]); ++b) covered[b] = true;
        for (uint32_t b = 0; b < h.baseStride; ++b)
            if (!covered[b]) CHECK(e.bytes[h.baseOffset + std::size_t(s) * h.baseStride + b] == 0);
        for (uint32_t b = ss::fieldEnd(h.fields[ss::groupSh]); b < h.shStride; ++b)
            CHECK(e.bytes[h.shOffset + std::size_t(s) * h.shStride + b] == 0);
    }
}

ss::StreamHeader validHeader(uint32_t shDegree, uint32_t splatCount = 10) {
    ss::StreamHeader h;
    uint32_t group = 0;
    REQUIRE(sn::streamLayout(webConfig(shDegree), splatCount, h, group) == sn::kEncodeOk);
    REQUIRE(ss::checkStreamHeader(h) == ss::streamOk);
    return h;
}

}  // namespace

// -------------------------------------------------------------------------------------------------
TEST_CASE("the header's layout is the format's") {
    CHECK(sizeof(ss::StreamField) == 48);
    CHECK(offsetof(ss::StreamField, encoding) == 0);
    CHECK(offsetof(ss::StreamField, components) == 4);
    CHECK(offsetof(ss::StreamField, offset) == 8);
    CHECK(offsetof(ss::StreamField, sharedRange) == 12);
    CHECK(offsetof(ss::StreamField, low) == 16);
    CHECK(offsetof(ss::StreamField, stepSize) == 32);
    CHECK(sizeof(ss::StreamHeader) == 432);
    CHECK(ss::streamHeaderSize == 432);
    CHECK(offsetof(ss::StreamHeader, magic) == 0);
    CHECK(offsetof(ss::StreamHeader, version) == 4);
    CHECK(offsetof(ss::StreamHeader, splatCount) == 8);
    CHECK(offsetof(ss::StreamHeader, shDegree) == 12);
    CHECK(offsetof(ss::StreamHeader, baseStride) == 16);
    CHECK(offsetof(ss::StreamHeader, shStride) == 20);
    CHECK(offsetof(ss::StreamHeader, baseOffset) == 24);
    CHECK(offsetof(ss::StreamHeader, shOffset) == 28);
    CHECK(offsetof(ss::StreamHeader, camera) == 32);
    CHECK(offsetof(ss::StreamHeader, fields) == 144);
    // "SPLS" in the first four bytes
    const uint32_t magic = ss::streamMagic;
    CHECK(std::memcmp(&magic, "SPLS", 4) == 0);
    // The largest stream fits one 128 MiB binding
    CHECK(uint64_t(ss::streamHeaderSize) + uint64_t(ss::streamMaxSplats) * (ss::maxBaseStride + ss::maxShStride) ==
          131981072ull);
    CHECK(131981072ull <= (128ull << 20));
    CHECK(uint64_t(ss::streamMaxSplats) * sizeof(ss::Splat) <= (128ull << 20));
}

TEST_CASE("the presets' strides are the plan's budget") {
    struct Case { sn::EncodeConfig config; uint32_t base, sh; };
    for (const Case& c : {Case{fullConfig(3), 56, 180}, Case{f16Config(3), 36, 92}, Case{webConfig(3), 32, 48},
                          Case{webConfig(0), 32, 0}, Case{fullConfig(1), 56, 36}, Case{webConfig(2), 32, 24}}) {
        ss::StreamHeader h;
        uint32_t group = 0;
        REQUIRE(sn::streamLayout(c.config, 1000, h, group) == sn::kEncodeOk);
        CHECK(h.baseStride == c.base);
        CHECK(h.shStride == c.sh);
        CHECK(h.baseOffset == 432);
        CHECK(h.shOffset == 432 + 1000 * c.base);
        CHECK(sn::streamSize(h) == 432 + 1000 * (c.base + c.sh));
        CHECK(ss::checkStreamHeader(h) == ss::streamOk);
    }
}

TEST_CASE("checkStreamHeader refuses each kind of bad header") {
    for (uint32_t d = 0; d <= 3; ++d) CHECK(ss::checkStreamHeader(validHeader(d)) == ss::streamOk);
    CHECK(ss::checkStreamHeader(validHeader(3, 0)) == ss::streamOk);
    CHECK(ss::checkStreamHeader(validHeader(3, ss::streamMaxSplats)) == ss::streamOk);
    {
        ss::StreamHeader h;
        uint32_t group = 0;
        REQUIRE(sn::streamLayout(fullConfig(3), ss::streamMaxSplats, h, group) == sn::kEncodeOk);
        CHECK(ss::checkStreamHeader(h) == ss::streamOk);
        CHECK(sn::streamSize(h) == 131981072u);
    }
    auto refused = [](uint32_t code, auto change, uint32_t shDegree = 3) {
        ss::StreamHeader h = validHeader(shDegree);
        change(h);
        CHECK(ss::checkStreamHeader(h) == code);
    };
    refused(ss::streamBadMagic, [](ss::StreamHeader& h) { h.magic ^= 1; });
    refused(ss::streamBadVersion, [](ss::StreamHeader& h) { h.version = 2; });
    refused(ss::streamBadShDegree, [](ss::StreamHeader& h) { h.shDegree = 4; });
    refused(ss::streamTooManySplats, [](ss::StreamHeader& h) {
        h.splatCount = ss::streamMaxSplats + 1;
        h.shOffset = h.baseOffset + h.splatCount * h.baseStride;
    });
    refused(ss::streamBadStride, [](ss::StreamHeader& h) { h.baseStride += 2; });
    refused(ss::streamBadStride, [](ss::StreamHeader& h) { h.shStride += 1; });
    refused(ss::streamBadStride, [](ss::StreamHeader& h) { h.baseStride = ss::maxBaseStride + 4; });
    refused(ss::streamBadStride, [](ss::StreamHeader& h) { h.shStride = ss::maxShStride + 4; });
    refused(ss::streamBadOffset, [](ss::StreamHeader& h) { h.baseOffset += 4; h.shOffset += 4; });
    refused(ss::streamBadOffset, [](ss::StreamHeader& h) { h.shOffset += 4; });
    refused(ss::streamBadEncoding, [](ss::StreamHeader& h) { h.fields[ss::groupOpacity].encoding = 4; });
    refused(ss::streamBadComponents, [](ss::StreamHeader& h) { h.fields[ss::groupRotation].components = 3; });
    refused(ss::streamBadComponents, [](ss::StreamHeader& h) { h.fields[ss::groupSh].components -= 3; });
    // The SH group must match the degree: degree 1's SH field in a degree 2 header
    refused(ss::streamBadComponents, [](ss::StreamHeader& h) { h.shDegree = 2; }, 1);
    refused(ss::streamBadSharedRange, [](ss::StreamHeader& h) { h.fields[ss::groupDc].sharedRange = 2; });
    refused(ss::streamBadSharedRange, [](ss::StreamHeader& h) { h.fields[ss::groupSh].sharedRange = 0; });
    // web: position f32 at 0, scale f16 at 12, DC f16 at 18, rotation u8 at 24, opacity u8 at 28
    {
        const ss::StreamHeader h = validHeader(3);
        CHECK(h.fields[ss::groupPosition].offset == 0);
        CHECK(h.fields[ss::groupScale].offset == 12);
        CHECK(h.fields[ss::groupDc].offset == 18);
        CHECK(h.fields[ss::groupRotation].offset == 24);
        CHECK(h.fields[ss::groupOpacity].offset == 28);
    }
    refused(ss::streamBadFieldLayout, [](ss::StreamHeader& h) { h.fields[ss::groupScale].offset = 13; });
    refused(ss::streamBadFieldLayout, [](ss::StreamHeader& h) { h.fields[ss::groupPosition].offset = 2; });
    refused(ss::streamBadFieldLayout, [](ss::StreamHeader& h) { h.fields[ss::groupOpacity].offset = 32; });
    refused(ss::streamBadFieldLayout, [](ss::StreamHeader& h) { h.fields[ss::groupSh].offset = 4; });
    refused(ss::streamBadFieldLayout, [](ss::StreamHeader& h) { h.fields[ss::groupDc].offset = 14; });
    refused(ss::streamBadFieldLayout, [](ss::StreamHeader& h) { h.fields[ss::groupOpacity].offset = 27; });
    // Aligned offsets near 2^32, whose end would wrap to a small number
    refused(ss::streamBadFieldLayout, [](ss::StreamHeader& h) { h.fields[ss::groupOpacity].offset = 0u - 4u; });
    refused(ss::streamBadFieldLayout, [](ss::StreamHeader& h) { h.fields[ss::groupSh].offset = 0u - 45u; });
    refused(ss::streamBadFieldLayout, [](ss::StreamHeader& h) { h.fields[ss::groupPosition].offset = 0u - 12u; });
    {
        ss::StreamHeader h;
        uint32_t group = 0;
        REQUIRE(sn::streamLayout(fullConfig(3), 10, h, group) == sn::kEncodeOk);
        h.fields[ss::groupSh].offset = 0u - 180u;
        CHECK(ss::checkStreamHeader(h) == ss::streamBadFieldLayout);
        h.fields[ss::groupSh].offset = 0;
        h.fields[ss::groupOpacity].offset = 0u - 4u;
        CHECK(ss::checkStreamHeader(h) == ss::streamBadFieldLayout);
    }
    // A field may end exactly at its stride, and strides may be larger than the fields need
    {
        ss::StreamHeader h = validHeader(3);
        h.baseStride = 40;
        h.shOffset = h.baseOffset + h.splatCount * h.baseStride;
        h.fields[ss::groupOpacity].offset = 39;
        CHECK(ss::checkStreamHeader(h) == ss::streamOk);
    }
    // The SH record is empty at degree 0
    refused(ss::streamBadStride, [](ss::StreamHeader& h) { h.shStride = 4; }, 0);
    // An empty SH field (degree 0) at offset 0 passes (the only offset within its stride, 0)
    {
        ss::StreamHeader h = validHeader(0);
        CHECK(h.shStride == 0);
        h.fields[ss::groupSh].offset = 0;
        CHECK(ss::checkStreamHeader(h) == ss::streamOk);
    }
    // The first problem is the one reported
    refused(ss::streamBadMagic, [](ss::StreamHeader& h) { h.magic = 0; h.version = 0; h.shDegree = 9; });
}

TEST_CASE("decodeHalf is every half's value, and floatToHalf inverts it") {
    for (uint32_t h = 0; h < 65536; ++h) {
        CAPTURE(h);
        const float v = ss::decodeHalf(h);
        const float expected = halfValue(h);
        if (std::isnan(expected)) {
            CHECK(std::isnan(v));
            continue;
        }
        CHECK(std::bit_cast<uint32_t>(v) == std::bit_cast<uint32_t>(expected));
        if (std::isfinite(v)) CHECK(sn::floatToHalf(v) == h);
    }
}

TEST_CASE("floatToHalf rounds to nearest, ties to even") {
    // Every midpoint between neighbouring finite halves of one sign (exact in float32) goes to the
    // even one; a hair either side goes to the nearer
    for (uint32_t h = 0; h < 0x7BFF; ++h) {
        CAPTURE(h);
        const float lo = halfValue(h), hi = halfValue(h + 1);
        const float mid = float((double(lo) + hi) / 2);
        REQUIRE(double(mid) == (double(lo) + hi) / 2);
        CHECK(sn::floatToHalf(mid) == ((h & 1) ? h + 1 : h));
        CHECK(sn::floatToHalf(-mid) == (0x8000 | ((h & 1) ? h + 1 : h)));
        CHECK(sn::floatToHalf(std::nextafter(mid, 0.0f)) == h);
        CHECK(sn::floatToHalf(std::nextafter(mid, INFINITY)) == h + 1);
    }
    CHECK(sn::floatToHalf(65504.0f) == 0x7BFF);
    CHECK(sn::floatToHalf(std::nextafter(65520.0f, 0.0f)) == 0x7BFF);
    CHECK(sn::floatToHalf(65520.0f) == 0x7C00);
    CHECK(sn::floatToHalf(-1e30f) == 0xFC00);
    CHECK(sn::floatToHalf(std::ldexp(1.0f, -25)) == 0);           // half the smallest subnormal: even
    CHECK(sn::floatToHalf(std::nextafter(std::ldexp(1.0f, -25), 1.0f)) == 1);
    CHECK(sn::floatToHalf(1e-30f) == 0);
    CHECK(sn::floatToHalf(-0.0f) == 0x8000);
    CHECK(sn::floatToHalf(std::numeric_limits<float>::denorm_min()) == 0);
}

TEST_CASE("componentBits takes a component from its word") {
    const uint32_t word = 0x89ABCDEFu;
    CHECK(ss::componentBits(word, 8, ss::encodingF32) == word);
    CHECK(ss::componentBits(word, 4, ss::encodingF16) == 0xCDEFu);
    CHECK(ss::componentBits(word, 6, ss::encodingU16) == 0x89ABu);
    CHECK(ss::componentBits(word, 0, ss::encodingU8) == 0xEFu);
    CHECK(ss::componentBits(word, 1, ss::encodingU8) == 0xCDu);
    CHECK(ss::componentBits(word, 2, ss::encodingU8) == 0xABu);
    CHECK(ss::componentBits(word, 435, ss::encodingU8) == 0x89u);
    // Shared ranges use .x for every component
    ss::StreamField f{ss::encodingU8, 9, 0, 1, glm::vec4(-1, 5, 5, 5), glm::vec4(0.5f, 7, 7, 7)};
    CHECK(ss::decodeComponent(f, 7, 0x04000000u, 3) == 1.0f);
    f.sharedRange = 0;
    CHECK(ss::decodeComponent(f, 2, 0x00040000u, 2) == 33.0f);
}

TEST_CASE("every encoding round-trips at every SH degree") {
    const std::vector<ss::Splat> splats = randomSplats(1, 97);
    for (uint32_t d = 0; d <= 3; ++d) {
        CAPTURE(d);
        SUBCASE("full") { checkRoundTrip(splats, fullConfig(d)); }
        SUBCASE("f16") { checkRoundTrip(splats, f16Config(d)); }
        SUBCASE("web") { checkRoundTrip(splats, webConfig(d)); }
        SUBCASE("quantized") { checkRoundTrip(splats, quantizedConfig(d)); }
    }
    SUBCASE("one splat and none") {
        checkRoundTrip({splats[0]}, webConfig(3));
        checkRoundTrip({}, webConfig(3));
        checkRoundTrip({}, fullConfig(0));
    }
}

TEST_CASE("data ranges are the data's extremes, explicit ones clamp") {
    std::vector<ss::Splat> splats = randomSplats(2, 50);
    const Encoded e = encode(splats, quantizedConfig(3));
    REQUIRE(e.status == sn::kEncodeOk);
    // Position, u16 per component: low is the minimum, low + 65535 step about the maximum
    for (int c = 0; c < 3; ++c) {
        float lo = INFINITY, hi = -INFINITY;
        for (const ss::Splat& s : splats) lo = std::min(lo, s.position[c]), hi = std::max(hi, s.position[c]);
        const ss::StreamField& f = e.header.fields[ss::groupPosition];
        CHECK(f.low[c] == lo);
        CHECK(f.stepSize[c] == (hi - lo) / 65535.0f);
    }
    // SH shares one range over every component
    {
        float lo = INFINITY;
        for (const ss::Splat& s : splats)
            for (int k = 3; k < 48; ++k) lo = std::min(lo, s.sh[k]);
        const ss::StreamField& f = e.header.fields[ss::groupSh];
        CHECK(f.sharedRange == 1);
        CHECK(f.low.x == -0.5f);  // explicit
        CHECK(f.stepSize.x == 1.0f / 65535.0f);
        // Values beyond the explicit range clamp to its ends, and the report shows it
        CHECK(lo < -0.5f);
        CHECK(e.report.groups[ss::groupSh].maxError >= (-0.5 - lo) * (1 - 1e-6));
    }
    // f32 and f16 fields carry no range
    for (const uint32_t g : {ss::groupPosition, ss::groupScale}) {
        const Encoded full = encode(splats, f16Config(3));
        CHECK(full.header.fields[g].low == glm::vec4(0));
        CHECK(full.header.fields[g].stepSize == glm::vec4(0));
    }
    // A constant group has a zero step and decodes exactly
    for (ss::Splat& s : splats) s.opacity = 2.5f;
    sn::EncodeConfig config = quantizedConfig(3);
    config.groups[ss::groupOpacity] = quantizedData(ss::encodingU8);
    const Encoded constant = encode(splats, config);
    REQUIRE(constant.status == sn::kEncodeOk);
    CHECK(constant.header.fields[ss::groupOpacity].stepSize.x == 0.0f);
    CHECK(constant.report.groups[ss::groupOpacity].maxError == 0.0);
    checkRoundTrip(splats, config);
}

TEST_CASE("the report measures each group's error, raw and activated") {
    const std::vector<ss::Splat> splats = randomSplats(3, 200);
    const Encoded e = encode(splats, webConfig(3));
    REQUIRE(e.status == sn::kEncodeOk);
    const sn::GroupReport* r = e.report.groups;
    CHECK(r[ss::groupPosition].maxError == 0.0);
    // f16 log-scales in [-7, 1): half a half's ulp, at most 2^-9 (in [4, 8)); exp's error is
    // largest near 1, where it's at most exp(1) 2^-12
    CHECK(r[ss::groupScale].maxError > 0.0);
    CHECK(r[ss::groupScale].maxError <= std::ldexp(1.0, -9));
    CHECK(r[ss::groupScale].maxActivatedError > 0.0);
    CHECK(r[ss::groupScale].maxActivatedError <= std::exp(1.0) * std::ldexp(1.0, -12) * (1 + 1e-6));
    // u8 rotation over [-1, 1]: half a step per component, so a small angle
    CHECK(r[ss::groupRotation].maxError <= 0.5 * 2.0 / 255.0 * (1 + 1e-5));
    CHECK(r[ss::groupRotation].maxActivatedError > 0.0);
    CHECK(r[ss::groupRotation].maxActivatedError < 0.05);
    // u8 opacity logits over the data's [-6, 6]: sigmoid's slope is at most 1/4
    CHECK(r[ss::groupOpacity].maxActivatedError <= r[ss::groupOpacity].maxError / 4 * (1 + 1e-6));
    CHECK(r[ss::groupDc].maxActivatedError == r[ss::groupDc].maxError);
    CHECK(r[ss::groupSh].maxError <= 0.5 * 1.2 / 255.0 * (1 + 1e-5));
}

TEST_CASE("the scale's activated error survives exp's overflow") {
    std::vector<ss::Splat> splats = randomSplats(9, 10);
    splats[3].logScale.x = 800.0f;
    // f32: exact, so no error (not inf - inf)
    const Encoded exact = encode(splats, fullConfig(0));
    REQUIRE(exact.status == sn::kEncodeOk);
    CHECK(exact.report.groups[ss::groupScale].maxActivatedError == 0.0);
    // Quantized over [-7, 800]: a value past 709 that moves has an infinite error, not a dropped NaN
    sn::EncodeConfig config = fullConfig(0);
    config.groups[ss::groupScale] = quantizedData(ss::encodingU8);
    splats[4].logScale.x = 750.0f;
    const Encoded coarse = encode(splats, config);
    REQUIRE(coarse.status == sn::kEncodeOk);
    CHECK(coarse.report.groups[ss::groupScale].maxActivatedError == INFINITY);
}

TEST_CASE("rotationAngle is the angle between two rotations") {
    const double pi = 3.14159265358979323846;
    const glm::dvec4 identity(1, 0, 0, 0);
    for (const double angle : {0.0, 1e-6, 0.3, pi / 2, 3.0}) {
        CAPTURE(angle);
        // (w, x, y, z) about z, unnormalized, and its negation (the same rotation)
        const glm::dvec4 q = 3.0 * glm::dvec4(std::cos(angle / 2), 0, 0, std::sin(angle / 2));
        CHECK(std::abs(sn::rotationAngle(identity, q) - angle) <= 1e-12);
        CHECK(std::abs(sn::rotationAngle(-q, identity) - angle) <= 1e-12);
    }
    CHECK(sn::rotationAngle(identity, identity) == 0.0);
    CHECK(sn::rotationAngle(identity, glm::dvec4(0)) == pi);
}

TEST_CASE("the encoder refuses bad configs and inputs") {
    const std::vector<ss::Splat> splats = randomSplats(4, 20);
    auto status = [&](const std::vector<ss::Splat>& in, const sn::EncodeConfig& config, std::size_t extra = 0) {
        ss::StreamHeader h;
        uint32_t group = 0;
        sn::streamLayout(config, uint32_t(in.size()), h, group);
        std::vector<uint8_t> out(sn::streamSize(h) + extra);
        sn::EncodeReport report;
        CHECK(sn::encodeStream(in, config, out, report) == report.status);
        return report;
    };
    auto badConfig = [&](auto change, uint32_t group) {
        sn::EncodeConfig config = webConfig(3);
        change(config);
        const sn::EncodeReport report = status(splats, config);
        CHECK(report.status == sn::kEncodeBadConfig);
        CHECK(report.group == group);
        ss::StreamHeader h;
        uint32_t layoutGroup = 99;
        CHECK(sn::streamLayout(config, 20, h, layoutGroup) == sn::kEncodeBadConfig);
        CHECK(layoutGroup == group);
    };
    badConfig([](sn::EncodeConfig& c) { c.shDegree = 4; }, ss::groupCount);
    badConfig([](sn::EncodeConfig& c) { c.groups[ss::groupDc].encoding = 4; }, ss::groupDc);
    badConfig([](sn::EncodeConfig& c) { c.groups[ss::groupScale].rangeMode = sn::kRangeData; }, ss::groupScale);
    badConfig([](sn::EncodeConfig& c) { c.groups[ss::groupSh].rangeMode = sn::kRangeNone; }, ss::groupSh);
    badConfig([](sn::EncodeConfig& c) { c.groups[ss::groupSh].rangeMode = 3; }, ss::groupSh);
    badConfig([](sn::EncodeConfig& c) { c.groups[ss::groupRotation].lo[3] = 2.0f; }, ss::groupRotation);
    badConfig([](sn::EncodeConfig& c) { c.groups[ss::groupRotation].hi[1] = NAN; }, ss::groupRotation);
    {
        // SH's explicit range is lo[0], hi[0]; the rest are unused
        sn::EncodeConfig config = quantizedConfig(3);
        config.groups[ss::groupSh].lo[1] = NAN;
        CHECK(status(splats, config).status == sn::kEncodeOk);
    }
    CHECK(status(splats, webConfig(3), 1).status == sn::kEncodeBadOutputSize);
    {
        ss::StreamHeader h;
        uint32_t group = 0;
        CHECK(sn::streamLayout(webConfig(3), ss::streamMaxSplats + 1, h, group) == sn::kEncodeTooManySplats);
    }
    auto badInput = [&](auto change, const sn::EncodeConfig& config, uint32_t code, uint32_t group) {
        std::vector<ss::Splat> in = splats;
        change(in[7]);
        const sn::EncodeReport report = status(in, config);
        CHECK(report.status == code);
        CHECK(report.group == group);
        CHECK(report.splat == 7);
    };
    badInput([](ss::Splat& s) { s.logScale.y = INFINITY; }, fullConfig(3), sn::kEncodeNonFinite, ss::groupScale);
    badInput([](ss::Splat& s) { s.sh[47] = NAN; }, fullConfig(3), sn::kEncodeNonFinite, ss::groupSh);
    badInput([](ss::Splat& s) { s.rotation = glm::vec4(0); }, webConfig(3), sn::kEncodeZeroRotation,
             ss::groupRotation);
    badInput([](ss::Splat& s) { s.sh[1] = 70000.0f; }, webConfig(3), sn::kEncodeF16Overflow, ss::groupDc);
    {
        // Values the stream drops aren't checked: SH above degree 1, a zero rotation kept as f32
        std::vector<ss::Splat> in = splats;
        in[7].sh[47] = NAN;
        in[7].rotation = glm::vec4(0);
        CHECK(status(in, fullConfig(1)).status == sn::kEncodeOk);
    }
    {
        // A data range too wide for a float step
        std::vector<ss::Splat> in = splats;
        in[0].position.x = -3e38f;
        in[1].position.x = 3e38f;
        sn::EncodeConfig config = webConfig(3);
        config.groups[ss::groupPosition] = quantizedData(ss::encodingU16);
        const sn::EncodeReport report = status(in, config);
        CHECK(report.status == sn::kEncodeRangeOverflow);
        CHECK(report.group == ss::groupPosition);
    }
}

TEST_CASE("encoding is deterministic") {
    const std::vector<ss::Splat> splats = randomSplats(5, 120);
    for (const sn::EncodeConfig& config : {fullConfig(3), webConfig(2), quantizedConfig(1)}) {
        const Encoded a = encode(splats, config);
        std::vector<uint8_t> b(a.bytes.size(), 0x5C);
        sn::EncodeReport report;
        REQUIRE(sn::encodeStream(splats, config, b, report) == sn::kEncodeOk);
        CHECK(a.bytes == b);
    }
}

TEST_CASE("the encoder writes the default camera into the header") {
    const std::vector<ss::Splat> splats = randomSplats(8, 64);
    const Encoded e = encode(splats, webConfig(3));
    REQUIRE(e.status == sn::kEncodeOk);
    ss::Camera cam;
    REQUIRE(sn::defaultCamera(splats, cam) == sn::kDefaultCameraOk);
    CHECK(std::memcmp(&e.header.camera, &cam, sizeof cam) == 0);
}
