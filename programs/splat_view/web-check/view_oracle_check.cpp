// view_oracle_check <splats.bin> <cameras.bin> <camera> <accum file>: splat_view's accum (the
// viewer graph's rgba16f target, lib/view_graph.zgal) for an S1 scene, written by a run with
// --inspect and SPLAT_VIEW_ACCUM=<accum file> (native or web), against the closed-form oracle
// (tests/oracle/closed_form.h) on the scene's splats and camera <camera>: every pixel's rgb against
// the oracle's sum of c alpha T (composite() over a black background: accum has none) and its a
// against 1 - T, within kTolerance. The accum file holds width x height pixels, row by row, each
// two uints: r | g << 16 and b | a << 16, as halves.
//
// The fixture must be one the viewer draws exactly as the oracle does, which it checks:
//   - an S1 scene (a camera at the origin rotated about z with fx = fy, splats rotated about z with
//     DC colour only);
//   - every splat's opacity below kOpacityMax (0.35): alpha at 3 sigma (o e^-4.5) is then below
//     1/255, so the viewer's 3-sigma quad leaves out no pixel the oracle (which has no cutoff)
//     blends, and alpha never reaches the 0.99 clamp;
//   - at most kMaxBlends splats blended at any pixel: T stays above 0.65^3, far from Rasterize's
//     1e-4 stop, which the viewer doesn't have.
// kTolerance (2e-3) is the halves' rounding, (2 blends + 1) 2^-12 for three blends (each blend
// rounds the fragment's output and the result; values are below 1, where a half's step is at most
// 2^-11), plus the GPU's exp against double's. A pixel with an alpha within kMargin (relative) of
// 1/255 may blend or not on the GPU: it's borderline, compared with the tolerance widened by what
// the flip changes (zgal_scene.h's Borderline). At most 1% of the pixels may be.
// Exits 1 on a mismatch or a fixture outside these bounds.
#include "view_oracle_check.zgal.h"

#include "oracle/zgal_scene.h"

#include <glm/gtc/packing.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

using namespace view_oracle_check;

namespace {

constexpr double kTolerance = 2e-3;
constexpr double kMargin = 1e-3;
constexpr double kOpacityMax = 0.35;
constexpr size_t kMaxBlends = 3;
constexpr double kBorderlineShare = 0.01;
constexpr int kShown = 8;  // mismatches printed before the rest are only counted

[[noreturn]] void fail(const std::string& what) {
    std::cout << "view_oracle_check: " << what << std::endl;
    std::exit(1);
}

template <typename T>
std::vector<T> read(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail(std::string("can't read ") + path);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (bytes.size() % sizeof(T) != 0) {
        fail(std::string(path) + " has " + std::to_string(bytes.size()) + " bytes, not a whole number of " +
             std::to_string(sizeof(T)) + "-byte elements");
    }
    std::vector<T> elements(bytes.size() / sizeof(T));
    std::memcpy(elements.data(), bytes.data(), bytes.size());
    return elements;
}

uint32_t index(const char* text) {
    char* end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value > UINT32_MAX) fail(std::string("not a camera index: ") + text);
    return uint32_t(value);
}

std::string rgbText(double r, double g, double b) {
    return "(" + std::to_string(r) + ", " + std::to_string(g) + ", " + std::to_string(b) + ")";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) fail("usage: view_oracle_check <splats.bin> <cameras.bin> <camera> <accum file>");
    const std::vector<Splat> splats = read<Splat>(argv[1]);
    const std::vector<Camera> cameras = read<Camera>(argv[2]);
    const uint32_t cameraIndex = index(argv[3]);
    const std::vector<glm::uvec2> accum = read<glm::uvec2>(argv[4]);

    if (cameraIndex >= cameras.size()) {
        fail("camera " + std::to_string(cameraIndex) + " is past the file's " + std::to_string(cameras.size()));
    }
    const std::optional<oracle::SceneCamera> scene = oracle::sceneCamera(cameras[cameraIndex]);
    if (!scene) fail("the camera isn't S1's: at the origin, rotated about z, with fx = fy");
    const oracle::Camera& cam = scene->camera;
    const uint32_t width = uint32_t(cam.width), height = uint32_t(cam.height);
    if (accum.size() != size_t(width) * height) {
        fail("the accum file holds " + std::to_string(accum.size()) + " pixels, not camera " +
             std::to_string(cameraIndex) + "'s " + std::to_string(width) + "x" + std::to_string(height));
    }
    if (splats.empty()) fail("the scene has no splats");

    std::vector<oracle::Splat> os;
    for (size_t i = 0; i < splats.size(); ++i) {
        const Splat& s = splats[i];
        const std::string name = "splat " + std::to_string(i);
        const std::optional<oracle::Splat> o = oracle::sceneSplat(s, scene->phi);
        if (!o) fail(name + " isn't rotated about z only (S1)");
        for (int k = 3; k < 48; ++k) {
            if (s.sh[k] != 0.0f) fail(name + " has SH above DC, which the oracle hasn't");
        }
        if (!(oracle::sigmoid(s.opacity) < kOpacityMax)) {
            fail(name + "'s opacity " + std::to_string(oracle::sigmoid(s.opacity)) +
                 " isn't below 0.35: alpha at 3 sigma would reach 1/255, past the viewer's quad");
        }
        os.push_back(*o);
    }

    const oracle::Rgb black{0.0, 0.0, 0.0};
    int failures = 0;
    double colourError = 0.0, tError = 0.0;
    uint32_t borderline = 0, blended = 0, covered = 0;
    size_t mostBlends = 0;
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const glm::uvec2 packed = accum[size_t(y) * width + x];
            const glm::vec2 rg = glm::unpackHalf2x16(packed.x), ba = glm::unpackHalf2x16(packed.y);
            const glm::vec4 gpu(rg.x, rg.y, ba.x, ba.y);
            const std::string pixel = "pixel (" + std::to_string(x) + ", " + std::to_string(y) + ")";
            if (!std::isfinite(gpu.x) || !std::isfinite(gpu.y) || !std::isfinite(gpu.z) || !std::isfinite(gpu.w)) {
                if (failures++ < kShown) std::cout << "view_oracle_check: MISMATCH " << pixel << " isn't finite" << std::endl;
                continue;
            }
            if (gpu.w > 0.0f) ++covered;

            const oracle::Vec2 p = oracle::pixel_centre(int(x), int(y));
            std::vector<oracle::Fragment> fragments;
            size_t blends = 0;
            for (const oracle::Splat& s : os) {
                const double alpha = oracle::raw_alpha(cam, s, p);
                fragments.push_back({s.z, alpha, oracle::colour(s)});
                if (alpha >= oracle::kAlphaMin) ++blends;
            }
            if (blends > kMaxBlends) {
                fail(pixel + " blends " + std::to_string(blends) + " splats: the tolerance holds for at most 3");
            }
            mostBlends = std::max(mostBlends, blends);
            if (blends > 0) ++blended;

            const oracle::PixelValue want = oracle::composite(fragments, black);
            const oracle::Borderline border = oracle::borderline(fragments, black, kMargin);
            double error = 0.0;
            for (int ch = 0; ch < 3; ++ch) error = std::max(error, std::abs(double(gpu[ch]) - want.colour[ch]));
            const double te = std::abs((1.0 - double(gpu.w)) - want.transmittance);
            if (border.any) {
                ++borderline;
            } else {
                colourError = std::max(colourError, error);
                tError = std::max(tError, te);
            }
            if (error > border.colourTolerance(kTolerance) || te > border.tTolerance(kTolerance)) {
                if (failures++ < kShown) {
                    std::cout << "view_oracle_check: MISMATCH " << pixel << ": accum " << rgbText(gpu.x, gpu.y, gpu.z)
                              << " a " << gpu.w << ", oracle "
                              << rgbText(want.colour[0], want.colour[1], want.colour[2]) << " a "
                              << 1.0 - want.transmittance << (border.any ? " (borderline)" : "") << std::endl;
                }
            }
        }
    }
    if (blended == 0) fail("the oracle blends no pixel: the fixture shows nothing");
    if (borderline > kBorderlineShare * width * height) fail("over 1% of the pixels are borderline");
    std::cout << "view_oracle_check: camera " << cameraIndex << " (" << width << "x" << height << ", phi "
              << scene->phi << "): " << splats.size() << " splats, " << blended << " pixels blended (up to "
              << mostBlends << " splats), " << covered << " covered in accum; colour within " << colourError
              << ", 1 - T within " << tError << " (" << borderline << " borderline pixels)" << std::endl;
    if (failures > 0) fail("FAILED, " + std::to_string(failures) + " mismatches");
    std::cout << "view_oracle_check: passed" << std::endl;
    return 0;
}
