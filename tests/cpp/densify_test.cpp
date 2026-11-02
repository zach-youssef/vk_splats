// lib/train_graph.zgal's densification functions, as ZGAL compiles them to C++: each splat's action
// (densifyAction: prune, keep, clone or split, the reference's rules and this rewrite's departures),
// the split halves' samples (standardNormals, splitPosition, splitLogScale, splitKey). The kernels
// that apply them (the compaction, the moments, the count) run on the GPU in programs/densify_test.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include "densify_test.zgal.h"

#include <cmath>
#include <cstdint>
#include <set>

namespace dt = densify_test;

namespace {

// A splat with opacity alpha (stored as its logit), every scale `scale` but the first `first`
dt::Splat splat(float alpha, float scale, float first = -1.0f) {
    dt::Splat s{};
    s.position = glm::vec3(1.0f, 2.0f, 3.0f);
    s.opacity = std::log(alpha / (1.0f - alpha));
    s.logScale = glm::vec3(std::log(first > 0.0f ? first : scale), std::log(scale), std::log(scale));
    s.rotation = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
    return s;
}

// Statistics whose mean gradient is `gradient` over 4 frames, with a largest radius
dt::DensifyStat stat(float gradient, uint32_t radius = 5) { return dt::DensifyStat{gradient * 4.0f, 4, radius}; }

constexpr float kThreshold = 0.0002f;
constexpr float kClone = 0.01f;  // percent_dense x an extent of 1
constexpr float kPrune = 0.1f;   // 0.1 x the extent

constexpr uint32_t kScreen = 20;  // the reference's max_screen_size

uint32_t action(const dt::Splat& s, const dt::DensifyStat& d, bool screen, uint32_t maxScreenSize = kScreen) {
    return dt::densifyAction(s, d, kThreshold, kClone, kPrune, screen, maxScreenSize);
}

}  // namespace

TEST_CASE("densifyGradient: the mean norm over the frames that saw the splat, 0 when none did") {
    CHECK(dt::densifyGradient(dt::DensifyStat{0.0f, 0, 0}) == 0.0f);
    CHECK(dt::densifyGradient(dt::DensifyStat{3.0f, 4, 0}) == 0.75f);
}

TEST_CASE("densifyAction: keep, clone or split by the gradient and the largest scale") {
    // Below the threshold: kept, whatever the scale
    CHECK(action(splat(0.5f, 0.005f), stat(0.0001f), false) == dt::densifyKeep);
    CHECK(action(splat(0.5f, 0.05f), stat(0.0001f), false) == dt::densifyKeep);
    // At or past it: cloned when small (at most cloneScale), split when larger
    CHECK(action(splat(0.5f, 0.005f), stat(kThreshold), false) == dt::densifyClone);
    CHECK(action(splat(0.5f, 0.005f), stat(0.01f), false) == dt::densifyClone);
    CHECK(action(splat(0.5f, 0.005f, 0.02f), stat(0.01f), false) == dt::densifySplit);  // the largest axis decides
    CHECK(action(splat(0.5f, 0.05f), stat(0.01f), false) == dt::densifySplit);
    // Never seen: gradient 0, kept
    CHECK(action(splat(0.5f, 0.05f), dt::DensifyStat{0.0f, 0, 0}, false) == dt::densifyKeep);
}

TEST_CASE("densifyAction: pruned below opacity 0.005, and by size once screenPrune holds") {
    // Transparent: pruned, even with a gradient that would densify it, with or without screenPrune
    CHECK(action(splat(0.004f, 0.005f), stat(0.0f), false) == dt::densifyPrune);
    CHECK(action(splat(0.004f, 0.005f), stat(0.01f), false) == dt::densifyPrune);
    CHECK(action(splat(0.004f, 0.05f), stat(0.01f), true) == dt::densifyPrune);
    CHECK(action(splat(0.006f, 0.005f), stat(0.0f), false) == dt::densifyKeep);
    // Large in the world (past pruneScale) or on screen (a radius past 20 px): pruned only with
    // screenPrune
    CHECK(action(splat(0.5f, 0.2f), stat(0.0f), false) == dt::densifyKeep);
    CHECK(action(splat(0.5f, 0.2f), stat(0.0f), true) == dt::densifyPrune);
    CHECK(action(splat(0.5f, 0.005f), stat(0.0f, 21), false) == dt::densifyKeep);
    CHECK(action(splat(0.5f, 0.005f), stat(0.0f, 21), true) == dt::densifyPrune);
    CHECK(action(splat(0.5f, 0.005f), stat(0.0f, 20), true) == dt::densifyKeep);
    // maxScreenSize 0 (the reference's effective behaviour): no screen-space prune, the world-space
    // one still applies
    CHECK(action(splat(0.5f, 0.005f), stat(0.0f, 1000), true, 0) == dt::densifyKeep);
    CHECK(action(splat(0.5f, 0.2f), stat(0.0f, 5), true, 0) == dt::densifyPrune);
    CHECK(action(splat(0.5f, 0.005f), stat(0.0f, 31), true, 30) == dt::densifyPrune);
    CHECK(action(splat(0.5f, 0.005f), stat(0.0f, 30), true, 30) == dt::densifyKeep);
    // A clone is checked as its splat (it's the same)
    CHECK(action(splat(0.5f, 0.005f), stat(0.01f, 21), true) == dt::densifyPrune);
    CHECK(action(splat(0.5f, 0.005f), stat(0.01f, 20), true) == dt::densifyClone);
    // A split's halves are 1.6 times smaller, and their radius isn't known: kept while their
    // scale is at most pruneScale, whatever the splat's radius
    CHECK(action(splat(0.5f, 0.15f), stat(0.01f, 100), true) == dt::densifySplit);
    CHECK(action(splat(0.5f, 0.17f), stat(0.01f, 100), true) == dt::densifyPrune);
    CHECK(action(splat(0.5f, 0.17f), stat(0.01f, 100), false) == dt::densifySplit);
}

TEST_CASE("standardNormals: deterministic, finite, mean 0 and variance 1 per component") {
    const glm::vec3 a = dt::standardNormals(12345u);
    const glm::vec3 b = dt::standardNormals(12345u);
    CHECK(a == b);
    constexpr int n = 100000;
    double sum[3] = {}, squares[3] = {}, product = 0.0;
    for (int i = 0; i < n; ++i) {
        const glm::vec3 v = dt::standardNormals(dt::splitKey(99u, uint32_t(i), uint32_t(i & 1)));
        for (int c = 0; c < 3; ++c) {
            REQUIRE(std::isfinite(v[c]));
            sum[c] += v[c];
            squares[c] += double(v[c]) * v[c];
        }
        product += double(v.x) * v.y;
    }
    for (int c = 0; c < 3; ++c) {
        CHECK(std::abs(sum[c] / n) < 0.02);
        CHECK(std::abs(squares[c] / n - 1.0) < 0.02);
    }
    // Box-Muller's pair is independent
    CHECK(std::abs(product / n) < 0.02);
}

TEST_CASE("splitKey: distinct for each splat's two halves and each seed") {
    std::set<uint32_t> keys;
    for (uint32_t seed : {0u, 1u, 77u}) {
        for (uint32_t i = 0; i < 1000; ++i) {
            keys.insert(dt::splitKey(seed, i, 0));
            keys.insert(dt::splitKey(seed, i, 1));
        }
    }
    CHECK(keys.size() == 6000);
}

TEST_CASE("splitPosition: a sample of the splat's Gaussian, R (scale x N(0, 1)) + position") {
    dt::Splat s = splat(0.5f, 0.1f, 0.4f);
    // A rotation by 90 degrees about z, (cos 45, 0, 0, sin 45), not normalized (quatToMatrix normalizes)
    s.rotation = glm::vec4(2.0f, 0.0f, 0.0f, 2.0f);
    for (uint32_t key : {1u, 2u, 3u, 4000000000u}) {
        const glm::vec3 n = dt::standardNormals(key);
        const glm::vec3 offset = dt::splitPosition(s, key) - s.position;
        // R maps x to y and y to -x: the offset is (-0.1 n.y, 0.4 n.x, 0.1 n.z)
        CHECK(offset.x == doctest::Approx(-0.1f * n.y).epsilon(1e-5));
        CHECK(offset.y == doctest::Approx(0.4f * n.x).epsilon(1e-5));
        CHECK(offset.z == doctest::Approx(0.1f * n.z).epsilon(1e-5));
    }
}

TEST_CASE("splitLogScale: the scale divided by 1.6") {
    const glm::vec3 l(-3.0f, 0.0f, 1.5f);
    const glm::vec3 r = dt::splitLogScale(l);
    for (int c = 0; c < 3; ++c) CHECK(r[c] == doctest::Approx(l[c] - std::log(1.6f)).epsilon(1e-6));
}

TEST_CASE("the reset's opacity logit is logit(0.01)") {
    CHECK(dt::resetOpacityLogit == doctest::Approx(std::log(0.01 / 0.99)).epsilon(1e-6));
    CHECK(dt::sigmoid(dt::resetOpacityLogit) == doctest::Approx(0.01f).epsilon(1e-5));
}
