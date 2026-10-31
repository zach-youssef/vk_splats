// tile_test's check: one frame's readbacks of the tile pipeline (lib/render_graph.zgal) against a
// CPU loop over the splat math's generated C++ (tileRect, tileCountOf, depthKey, depthBitsFor).
//
// The CPU recomputes the stages from the GPU's projections (each splat's mean, radius and depth),
// so everything after the preprocess is compared exactly, with no tolerance:
//   - each splat's tile rectangle and count: tileRect(mean, radius) (floor and ceil);
//   - every entry's key: depthKey(tile, depth) (the depth quantization), and its splat;
//   - the entries' order: by key, ties by splat index (the sort is stable and entries start in
//     splat order: docs/architecture.md), so the keys and splats match position for position;
//   - each tile's range [start, end): its entries, sorted by key, without duplicate splats.
// The preprocess itself is compared with the CPU's projectSplat: whether each splat is culled
// exactly, its mean, depth and radius to float precision (the GPU's exp, sqrt and division
// needn't round as C++'s do). A degenerate splat (a zero quaternion, an overflowing scale; the
// last --culled-tail splats) must be culled on both, and a kept one's radius below 2^31.
//
// A frame that overflowed its capacity is checked too: it must hold exactly the first `capacity`
// entries of the CPU's list in splat order (whole splats, then part of one), sorted, with the
// ranges of those. check() says which kind of frame it checked, so the app can require both.
#include "tile_test.zgal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <set>
#include <utility>
#include <vector>

namespace {

using namespace tile_test;

constexpr int kShown = 8;  // mismatches printed per check before the rest are only counted

bool isCulled(const Projected& p) {
    return p.tileCount == 0;
}

bool allZero(const Projected& p) {
    return p.mean == glm::vec2(0.0f) && p.depth == 0.0f && p.radius == 0 && p.conic == glm::vec3(0.0f) &&
           p.opacity == 0.0f && p.colour == glm::vec3(0.0f) && p.tiles == glm::uvec4(0u);
}

// |a - b| within tol relative to max(1, |b|)
bool close(float a, float b, float tol) {
    return std::abs(a - b) <= tol * std::max(1.0f, std::abs(b));
}

struct Failures {
    int count = 0;
    void operator()(const std::string& what) {
        if (count < kShown) std::cout << "tile_test: MISMATCH " << what << std::endl;
        ++count;
    }
};

std::string str(glm::uvec4 v) {
    return "(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ", " +
           std::to_string(v.w) + ")";
}

uint32_t check(const zgal::Data<Splat>& splats, const zgal::Data<Camera>& cameras,
               const zgal::Data<uint32_t>& counts, const zgal::Data<Projected>& projected,
               const zgal::Data<uint32_t>& entryKeys, const zgal::Data<uint32_t>& entrySplats,
               const zgal::Data<glm::uvec2>& ranges, uint32_t cameraIndex, uint32_t shDegree, float near, float far,
               uint32_t culledTail) {
    if (counts.size() != countSize) return checkedNothing;
    const uint32_t entries = counts[countEntries];
    const uint32_t total = counts[countTotal];
    const uint32_t capacity = counts[countCapacity];
    const bool overflowed = total > capacity;
    // Readbacks of another capacity's frames, or that haven't arrived yet
    if (entryKeys.size() != capacity || entrySplats.size() != capacity) return checkedNothing;
    if (projected.empty() || ranges.empty()) return checkedNothing;

    Failures fail;
    if (cameraIndex >= cameras.size()) {
        std::cout << "tile_test: --camera " << cameraIndex << " is past the file's " << cameras.size() << " cameras"
                  << std::endl;
        std::exit(1);
    }
    if (projected.size() != splats.size()) {
        std::cout << "tile_test: --splat-count " << projected.size() << " isn't the file's " << splats.size()
                  << " splats" << std::endl;
        std::exit(1);
    }
    const Camera& cam = cameras[cameraIndex];
    const glm::uvec2 grid = tileGrid(cam.size);
    const uint32_t tileCount = grid.x * grid.y;
    if (ranges.size() != tileCount) {
        std::cout << "tile_test: --width and --height give " << ranges.size() << " tiles, camera " << cameraIndex
                  << "'s size " << cam.size.x << "x" << cam.size.y << " " << tileCount << std::endl;
        std::exit(1);
    }
    if (culledTail > splats.size()) {
        std::cout << "tile_test: --culled-tail " << culledTail << " is more than the " << splats.size() << " splats"
                  << std::endl;
        std::exit(1);
    }
    const uint32_t depthBits = depthBitsFor(tileCount);
    if (counts[countDepthBits] != depthBits) fail("depth bits: GPU " + std::to_string(counts[countDepthBits]) +
                                                  ", CPU " + std::to_string(depthBits));
    const uint32_t shift = depthShift(depthBits, near, far);
    if (counts[countDepthShift] != shift) fail("depth shift: GPU " + std::to_string(counts[countDepthShift]) +
                                               ", CPU " + std::to_string(shift));
    // min(total, capacity): all of them, or as many as the capacity holds
    if (entries != std::min(total, capacity)) {
        fail("entries " + std::to_string(entries) + " of a total " + std::to_string(total) + " and a capacity " +
             std::to_string(capacity));
    }

    // The preprocess against the CPU's, and each kept splat's entries from the GPU's projection
    struct Entry {
        uint32_t key;
        uint32_t splat;
    };
    std::vector<Entry> expected;
    uint32_t culled = 0;
    for (uint32_t i = 0; i < splats.size(); ++i) {
        const Projected& gpu = projected[i];
        const Projected cpu = projectSplat(splats[i], cam, near, shDegree);
        const std::string at = "splat " + std::to_string(i) + ": ";
        if (isCulled(gpu) != isCulled(cpu)) {
            fail(at + "culled on the " + (isCulled(gpu) ? "GPU" : "CPU") + " only (GPU radius " +
                 std::to_string(gpu.radius) + ", tiles " + str(gpu.tiles) + ")");
            continue;
        }
        if (i >= splats.size() - culledTail && !isCulled(gpu)) {
            fail(at + "is degenerate (--culled-tail), but kept on both");
        }
        if (isCulled(gpu)) {
            ++culled;
            if (!allZero(gpu)) fail(at + "culled, but not all zero");
            continue;
        }
        if (!std::isfinite(gpu.mean.x) || !std::isfinite(gpu.mean.y) || !std::isfinite(gpu.depth) ||
            gpu.radius >= (1u << 31)) {
            fail(at + "kept with a mean, depth or radius out of range");
            continue;
        }
        if (!close(gpu.mean.x, cpu.mean.x, 1e-4f) || !close(gpu.mean.y, cpu.mean.y, 1e-4f) ||
            !close(gpu.depth, cpu.depth, 1e-5f) || std::abs(int64_t(gpu.radius) - int64_t(cpu.radius)) > 1) {
            fail(at + "projected differently: GPU mean (" + std::to_string(gpu.mean.x) + ", " +
                 std::to_string(gpu.mean.y) + ") depth " + std::to_string(gpu.depth) + " radius " +
                 std::to_string(gpu.radius) + ", CPU (" + std::to_string(cpu.mean.x) + ", " +
                 std::to_string(cpu.mean.y) + ") " + std::to_string(cpu.depth) + " " + std::to_string(cpu.radius));
        }
        // Exact from here: the rectangle from the GPU's mean and radius
        const glm::uvec4 rect = tileRect(gpu.mean, gpu.radius, grid);
        if (rect != gpu.tiles || tileCountOf(rect) != gpu.tileCount) {
            fail(at + "tiles " + str(gpu.tiles) + " (" + std::to_string(gpu.tileCount) + "), CPU's tileRect " +
                 str(rect) + " (" + std::to_string(tileCountOf(rect)) + ")");
        }
        for (uint32_t y = rect.y; y < rect.w; ++y) {
            for (uint32_t x = rect.x; x < rect.z; ++x) {
                expected.push_back({depthKey(y * grid.x + x, gpu.depth, depthBits, near, far), i});
            }
        }
    }
    if (expected.size() != total) {
        fail("the GPU's total is " + std::to_string(total) + " entries, the CPU's " + std::to_string(expected.size()));
    }
    // An overflowing frame holds the first `capacity` entries in splat order
    if (overflowed && expected.size() > capacity) expected.resize(capacity);
    // Entries start in splat order, so a stable sort by key orders ties by splat index
    std::stable_sort(expected.begin(), expected.end(), [](const Entry& a, const Entry& b) { return a.key < b.key; });

    // The sorted entries, position for position
    int keyMismatches = 0;
    const uint32_t compared = std::min<uint32_t>(entries, uint32_t(expected.size()));
    for (uint32_t e = 0; e < compared; ++e) {
        const uint32_t key = entryKeys[e];
        const uint32_t splat = entrySplats[e];
        if (key != expected[e].key || splat != expected[e].splat) {
            ++keyMismatches;
            fail("entry " + std::to_string(e) + ": GPU key " + std::to_string(key) + " (tile " +
                 std::to_string(keyTile(key, depthBits)) + ") splat " + std::to_string(splat) + ", CPU key " +
                 std::to_string(expected[e].key) + " (tile " + std::to_string(keyTile(expected[e].key, depthBits)) +
                 ") splat " + std::to_string(expected[e].splat));
        }
    }

    // Each tile's range: the CPU's count of entries, sorted by key, without a splat twice
    std::vector<glm::uvec2> expectedRanges(tileCount, glm::uvec2(0u));
    for (uint32_t e = 0; e < expected.size(); ++e) {
        const uint32_t tile = keyTile(expected[e].key, depthBits);
        if (expectedRanges[tile] == glm::uvec2(0u)) expectedRanges[tile].x = e;
        expectedRanges[tile].y = e + 1;
    }
    uint32_t occupied = 0;
    uint32_t largest = 0;
    for (uint32_t t = 0; t < tileCount; ++t) {
        const glm::uvec2 r = ranges[t];
        const glm::uvec2 want = expectedRanges[t];
        if (r != want) {
            fail("tile " + std::to_string(t) + ": GPU range [" + std::to_string(r.x) + ", " + std::to_string(r.y) +
                 "), CPU [" + std::to_string(want.x) + ", " + std::to_string(want.y) + ")");
            continue;
        }
        if (r.y > r.x) ++occupied;
        largest = std::max(largest, r.y - r.x);
        std::set<uint32_t> seen;
        for (uint32_t e = r.x; e < r.y && e < entries; ++e) {
            if (keyTile(entryKeys[e], depthBits) != t) fail("tile " + std::to_string(t) + "'s entry " + std::to_string(e) + " is of another tile");
            if (e > r.x && entryKeys[e] < entryKeys[e - 1]) fail("tile " + std::to_string(t) + " isn't sorted at entry " + std::to_string(e));
            if (!seen.insert(entrySplats[e]).second) {
                fail("tile " + std::to_string(t) + " holds splat " + std::to_string(entrySplats[e]) + " twice");
            }
        }
    }

    std::cout << "tile_test: " << (overflowed ? "overflowing frame, " : "") << "camera " << cameraIndex << " (" << cam.size.x << "x" << cam.size.y << ", " << tileCount
              << " tiles, " << depthBits << " depth bits): " << splats.size() << " splats, " << culled
              << " culled, " << total << " entries in " << occupied << " tiles (at most " << largest
              << " in one), capacity " << capacity << ", " << entries << " held" << std::endl;
    if (fail.count > 0) {
        std::cout << "tile_test: FAILED, " << fail.count << " mismatches (" << keyMismatches << " entries)"
                  << std::endl;
        std::exit(1);
    }
    return overflowed ? checkedOverflowing : checkedComplete;
}

}  // namespace

Result<tile_test::MainInputs> tile_test::mainInputs(VulkanContext&) {
    return MainInputs{
        .check = check,
        .fail = [] {
            std::cout.flush();
            std::exit(1);
        },
    };
}
