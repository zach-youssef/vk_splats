// render_test's checks, which render_test.cpp gives the native program and
// web-check/render_test_check.cpp runs on a web run's readbacks: one frame's rasterizer outputs
// (lib/render_graph.zgal's step 5) against
//   - a CPU blend, in double, of the GPU's projections over the GPU's sorted tile entries and ranges
//     (tile_test checks those exactly): every pixel's colour (C + T bg) and final transmittance T
//     within kCpuTolerance, and its last entry exactly. This isolates the rasterizer: the only
//     differences left are float against double in the Gaussian and the blend.
//   - with --oracle (an S1 scene), the closed-form oracle (tests/oracle/closed_form.h), which
//     recomputes everything from the splat's parameters, the camera's rotation about z
//     (world_to_camera_about_z) and its focal length and principal point: colour and T within
//     kOracleTolerance (1e-3, the plan's bound).
//     The oracle has no tile cutoff, so a splat blends only into the tiles of its rectangle
//     (tileRect from the oracle's mean and radius_px), which must be the GPU's.
// A pixel where a threshold decision (alpha against 1/255, T against 1e-4, a nonzero Gaussian
// exponent against 0) sits within rounding of its bound may go either way on the GPU, and either
// is right: it's borderline, and compared with a tolerance widened by what one flip can change
// (flipTolerance) and without its last entry. At most kBorderlineShare of the pixels may be.
// Every pixel's outputs must be finite. check() prints a hash of the outputs, so test.sh can
// compare runs; sameColours and sameEntries compare frames bit for bit.
//
// Before including it, alias the namespace with SplatGraph's structs and functions (a .zgal file's
// that imports lib/render_graph.zgal): namespace render_types = render_test;
#pragma once

#include "../../tests/oracle/zgal_scene.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace render_check {

using namespace render_types;

constexpr int kShown = 8;  // mismatches printed before the rest are only counted
constexpr double kCpuTolerance = 1e-4;
constexpr double kOracleTolerance = 1e-3;
// How close to a threshold a value is borderline: relative for alpha and T (float's exp and the
// projection's rounding against double), absolute for the exponent
constexpr double kCpuMargin = 1e-4;
constexpr double kOracleMargin = 1e-3;
constexpr double kPowerMargin = 1e-5;
// The most pixels that may be borderline, as a share of the image (observed: at most 3 pixels)
constexpr double kBorderlineShare = 0.01;

struct Failures {
    int count = 0;
    void operator()(const std::string& what) {
        if (count < kShown) std::cout << "render_test: MISMATCH " << what << std::endl;
        ++count;
    }
};

[[noreturn]] inline void usage(const std::string& what) {
    std::cout << "render_test: " << what << std::endl;
    std::exit(1);
}

inline std::string pixelName(uint32_t x, uint32_t y) {
    return "pixel (" + std::to_string(x) + ", " + std::to_string(y) + ")";
}

using oracle::Borderline;
using oracle::near;

struct Blend {
    oracle::Rgb colour{0.0, 0.0, 0.0};  // C + T bg
    double t = 1.0;
    uint32_t last = 0;
    Borderline borderline;
    bool stopped = false;  // the transmittance stop ended it
};

// The kernel's blend in double: the same order, thresholds and last entry
inline Blend cpuBlend(const zgal::Data<Projected>& projected, const zgal::Data<uint32_t>& entrySplats, glm::uvec2 range,
               double px, double py, const oracle::Rgb& bg) {
    Blend b;
    b.last = range.x;
    b.borderline.see(bg);
    oracle::Rgb c{0.0, 0.0, 0.0};
    for (uint32_t e = range.x; e < range.y; ++e) {
        const Projected& p = projected[entrySplats[e]];
        b.borderline.see({p.colour.x, p.colour.y, p.colour.z});
        const double dx = double(p.mean.x) - px, dy = double(p.mean.y) - py;
        const double power = -0.5 * (p.conic.x * dx * dx + p.conic.z * dy * dy) - p.conic.y * dx * dy;
        // The conic is positive definite, so the exponent is negative but for rounding, and exactly
        // 0 (which both sides blend) only at the mean
        if (power != 0.0 && std::abs(power) <= kPowerMargin) {
            b.borderline.add(std::min(oracle::kAlphaMax, double(p.opacity)) * b.t);
        }
        if (power > 0.0) continue;
        const double alpha = std::min(oracle::kAlphaMax, double(p.opacity) * std::exp(power));
        if (near(alpha, oracle::kAlphaMin, kCpuMargin)) b.borderline.add(alpha * b.t);
        if (alpha < oracle::kAlphaMin) continue;
        const double next = b.t * (1.0 - alpha);
        if (near(next, oracle::kTransmittanceMin, kCpuMargin)) b.borderline.add(b.t);
        if (next < oracle::kTransmittanceMin) {
            b.stopped = true;
            break;
        }
        for (int ch = 0; ch < 3; ++ch) c[ch] += double(p.colour[ch]) * alpha * b.t;
        b.t = next;
        b.last = e + 1;
    }
    for (int ch = 0; ch < 3; ++ch) b.colour[ch] = c[ch] + b.t * bg[ch];
    return b;
}

// The oracle's splat for an S1 splat seen by a camera rotated by phi about z
inline oracle::Splat oracleSplat(const Splat& s, double phi) {
    const std::optional<oracle::Splat> o = oracle::sceneSplat(s, phi);
    if (!o) usage("--oracle needs splats rotated about z only (S1)");
    return *o;
}

inline bool check(const zgal::Data<Splat>& splats, const zgal::Data<Camera>& cameras, const zgal::Data<uint32_t>& counts,
           const zgal::Data<Projected>& projected, const zgal::Data<uint32_t>& entrySplats,
           const zgal::Data<uint32_t>& ranges, const zgal::Data<glm::vec4>& colours,
           const zgal::Data<uint32_t>& lastEntries, uint32_t cameraIndex, glm::vec3 background, bool useOracle) {
    if (counts.size() != countSize || entrySplats.size() != counts[countCapacity]) return false;
    if (cameraIndex >= cameras.size()) {
        usage("--camera " + std::to_string(cameraIndex) + " is past the file's " + std::to_string(cameras.size()) +
              " cameras");
    }
    if (projected.size() != splats.size()) {
        usage("--splat-count " + std::to_string(projected.size()) + " isn't the file's " +
              std::to_string(splats.size()) + " splats");
    }
    const Camera& cam = cameras[cameraIndex];
    const uint32_t width = cam.size.x, height = cam.size.y;
    const glm::uvec2 grid = tileGrid(cam.size);
    if (ranges.size() != 2 * grid.x * grid.y || colours.size() != width * height || lastEntries.size() != width * height) {
        usage("--width and --height aren't camera " + std::to_string(cameraIndex) + "'s size " +
              std::to_string(width) + "x" + std::to_string(height));
    }
    const uint32_t entries = counts[countEntries];
    if (counts[countTotal] > counts[countCapacity]) usage("the frame overflowed its capacity; raise --capacity");
    const oracle::Rgb bg{background.x, background.y, background.z};

    Failures fail;
    // The tile ranges themselves: the CPU blend below reads the GPU's ranges, so a lost start or end
    // write (an inverted or empty range) would blend the same wrong entries on both sides. Every
    // sorted entry is of exactly one tile, so the non-empty ranges must partition [0, entries).
    uint32_t largestTile = 0;
    std::vector<glm::uvec2> occupied;
    for (uint32_t t = 0; t < grid.x * grid.y; ++t) {
        const glm::uvec2 range(ranges[2 * t], ranges[2 * t + 1]);
        if (range.x > range.y) {
            fail("tile " + std::to_string(t) + "'s range [" + std::to_string(range.x) + ", " +
                 std::to_string(range.y) + ") is inverted");
        } else if (range.x < range.y) {
            occupied.push_back(range);
            largestTile = std::max(largestTile, range.y - range.x);
        }
    }
    std::sort(occupied.begin(), occupied.end(), [](glm::uvec2 a, glm::uvec2 b) { return a.x < b.x; });
    uint32_t covered = 0;
    for (const glm::uvec2& range : occupied) {
        if (range.x != covered) {
            fail("the tile ranges " + std::string(range.x < covered ? "overlap" : "leave a gap") + " at entry " +
                 std::to_string(std::min(range.x, covered)));
        }
        covered = std::max(covered, range.y);
    }
    if (covered != entries) {
        fail("the tile ranges cover up to entry " + std::to_string(covered) + ", not the " +
             std::to_string(entries) + " entries");
    }

    // The rasterizer against the CPU's blend of the same entries
    double colourError = 0.0, tError = 0.0;
    uint32_t borderline = 0, blendedPixels = 0, stoppedPixels = 0;
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const uint32_t index = y * width + x;
            const glm::vec4 gpu = colours[index];
            if (!std::isfinite(gpu.x) || !std::isfinite(gpu.y) || !std::isfinite(gpu.z) || !std::isfinite(gpu.w)) {
                fail(pixelName(x, y) + " isn't finite");
                continue;
            }
            const uint32_t tile = (y / tileSize) * grid.x + x / tileSize;
            const glm::uvec2 range(ranges[2 * tile], ranges[2 * tile + 1]);
            if (range.y > entries) {
                fail(pixelName(x, y) + "'s tile range ends past the " + std::to_string(entries) + " entries");
                continue;
            }
            const Blend cpu = cpuBlend(projected, entrySplats, range, x + 0.5, y + 0.5, bg);
            if (cpu.last != range.x) ++blendedPixels;
            if (cpu.stopped) ++stoppedPixels;
            double error = 0.0;
            for (int ch = 0; ch < 3; ++ch) error = std::max(error, std::abs(double(gpu[ch]) - cpu.colour[ch]));
            const double te = std::abs(double(gpu.w) - cpu.t);
            const Borderline& border = cpu.borderline;
            if (border.any) {
                ++borderline;
            } else {
                colourError = std::max(colourError, error);
                tError = std::max(tError, te);
            }
            // A flipped decision moves the last entry too, but never out of the tile's range
            const bool lastOk = border.any ? lastEntries[index] >= range.x && lastEntries[index] <= range.y
                                           : lastEntries[index] == cpu.last;
            if (error > border.colourTolerance(kCpuTolerance) || te > border.tTolerance(kCpuTolerance) || !lastOk) {
                fail(pixelName(x, y) + ": GPU (" + std::to_string(gpu.x) + ", " + std::to_string(gpu.y) + ", " +
                     std::to_string(gpu.z) + ") T " + std::to_string(gpu.w) + " last " +
                     std::to_string(lastEntries[index]) + ", CPU (" + std::to_string(cpu.colour[0]) + ", " +
                     std::to_string(cpu.colour[1]) + ", " + std::to_string(cpu.colour[2]) + ") T " +
                     std::to_string(cpu.t) + " last " + std::to_string(cpu.last) +
                     (border.any ? " (borderline)" : ""));
            }
        }
    }
    uint64_t hash = oracle::fnv1a(oracle::kFnvBasis, colours.elements().data(), colours.size() * sizeof(glm::vec4));
    hash = oracle::fnv1a(hash, lastEntries.elements().data(), lastEntries.size() * sizeof(uint32_t));
    char hashText[17];
    std::snprintf(hashText, sizeof hashText, "%016llx", static_cast<unsigned long long>(hash));
    std::cout << "render_test: camera " << cameraIndex << " (" << width << "x" << height << "): " << splats.size()
              << " splats, " << entries << " entries (at most " << largestTile << " in a tile), " << blendedPixels << " pixels blended something, "
              << stoppedPixels << " stopped at T < 1e-4; against the "
              << "CPU blend: colour within " << colourError << ", T within " << tError << " (" << borderline
              << " borderline pixels, within what a flip changes)" << std::endl;

    if (useOracle) {
        const std::optional<oracle::SceneCamera> scene = oracle::sceneCamera(cam);
        if (!scene) usage("--oracle needs a camera at the origin rotated about z with fx = fy (S1)");
        const double phi = scene->phi;
        const oracle::Camera& ocam = scene->camera;
        std::vector<oracle::Splat> os;
        std::vector<glm::uvec4> rects;
        for (uint32_t i = 0; i < splats.size(); ++i) {
            const oracle::Splat s = oracleSplat(splats[i], phi);
            const oracle::Vec2 mu = oracle::mean2d(ocam, s);
            const int radius = oracle::radius_px(oracle::cov2d(ocam, s));
            const glm::uvec4 rect = tileRect(glm::vec2(mu.x, mu.y), uint32_t(radius), grid);
            if (rect != projected[i].tiles) {
                fail("splat " + std::to_string(i) + ": the oracle's tiles (" + std::to_string(rect.x) + ", " +
                     std::to_string(rect.y) + ", " + std::to_string(rect.z) + ", " + std::to_string(rect.w) +
                     ") aren't the GPU's");
            }
            os.push_back(s);
            rects.push_back(rect);
        }
        double oracleColour = 0.0, oracleT = 0.0;
        uint32_t oracleBorderline = 0;
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                const uint32_t tx = x / tileSize, ty = y / tileSize;
                const oracle::Vec2 p = oracle::pixel_centre(int(x), int(y));
                std::vector<oracle::Fragment> fragments;
                for (size_t i = 0; i < os.size(); ++i) {
                    const glm::uvec4 r = rects[i];
                    if (tx < r.x || tx >= r.z || ty < r.y || ty >= r.w) continue;
                    fragments.push_back({os[i].z, oracle::raw_alpha(ocam, os[i], p), oracle::colour(os[i])});
                }
                const Borderline border = oracle::borderline(fragments, bg, kOracleMargin);
                const oracle::PixelValue want = oracle::composite(fragments, bg);
                const glm::vec4 gpu = colours[y * width + x];
                double error = 0.0;
                for (int ch = 0; ch < 3; ++ch) error = std::max(error, std::abs(double(gpu[ch]) - want.colour[ch]));
                const double te = std::abs(double(gpu.w) - want.transmittance);
                if (border.any) {
                    ++oracleBorderline;
                } else {
                    oracleColour = std::max(oracleColour, error);
                    oracleT = std::max(oracleT, te);
                }
                if (error > border.colourTolerance(kOracleTolerance) || te > border.tTolerance(kOracleTolerance)) {
                    fail(pixelName(x, y) + " against the oracle: GPU (" + std::to_string(gpu.x) + ", " +
                         std::to_string(gpu.y) + ", " + std::to_string(gpu.z) + ") T " + std::to_string(gpu.w) +
                         ", oracle (" + std::to_string(want.colour[0]) + ", " + std::to_string(want.colour[1]) + ", " +
                         std::to_string(want.colour[2]) + ") T " + std::to_string(want.transmittance) +
                         (border.any ? " (borderline)" : ""));
                }
            }
        }
        std::cout << "render_test: against the closed-form oracle (camera phi " << phi << "): colour within "
                  << oracleColour << ", T within " << oracleT << " (" << oracleBorderline
                  << " borderline pixels, within what a flip changes)" << std::endl;
        if (oracleBorderline > kBorderlineShare * width * height) {
            fail("over 1% of the pixels are borderline for the oracle");
        }
    }
    if (borderline > kBorderlineShare * width * height) fail("over 1% of the pixels are borderline for the CPU blend");

    if (fail.count > 0) {
        std::cout << "render_test: FAILED, " << fail.count << " mismatches" << std::endl;
        std::exit(1);
    }
    std::cout << "render_test: hash " << hashText << std::endl;
    return true;
}

}  // namespace render_check
