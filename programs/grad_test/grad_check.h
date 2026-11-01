// grad_test's checks, which grad_test.cpp gives the native program and web-check/grad_test_check.cpp
// runs on a web run's readbacks: one frame's gradients with respect to the splats' projections
// (lib/train_graph.zgal's splatGradients) and its loss
//   - every gradient and the loss finite; a hash of them printed, so test.sh can compare runs;
//   - with --oracle (S1's "gradient" variant: one splat blended into every pixel, a constant
//     target), against the closed-form oracle (tests/oracle/closed_form.h's projected_gradients,
//     L2 or L1) at the oracle's projection, with the target rounded to 8 bits as the targets file
//     has it: the loss and each gradient (the mean's x and y, the conic's a, b and c, the opacity,
//     the colour's r, g and b) within kOracleTolerance (1e-3, the plan's bound) relative to the
//     oracle's, or to 1e-2 of the largest of its kind (mean, conic, colour) for one near 0.
// The oracle's conic gradient counts both off-diagonal entries in b, as the kernel's does.
//
// Before including it, alias the namespace with TrainGraph's structs (a .zgal file's that imports
// lib/train_graph.zgal): namespace render_types = grad_test;
#pragma once

#include "../../tests/oracle/closed_form.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

namespace grad_check {

using namespace render_types;

constexpr double kOracleTolerance = 1e-3;
// A gradient's error is relative to the larger of its own size and this share of its kind's largest
constexpr double kKindFloor = 1e-2;
constexpr int kShown = 8;  // mismatches printed before the rest are only counted

[[noreturn]] inline void usage(const std::string& what) {
    std::cout << "grad_test: " << what << std::endl;
    std::exit(1);
}

// The oracle's splat for an S1 splat seen by a camera rotated by phi about z (as render_check.h's)
inline oracle::Splat oracleSplat(const Splat& s, double phi) {
    if (s.rotation.y != 0.0f || s.rotation.z != 0.0f) usage("--oracle needs splats rotated about z only (S1)");
    oracle::Splat w{};
    w.x = s.position.x;
    w.y = s.position.y;
    w.z = s.position.z;
    w.log_scale = {s.logScale.x, s.logScale.y, s.logScale.z};
    // (w, x, y, z) in (.x, .y, .z, .w): quat_about_z(theta) = (cos(theta/2), 0, 0, sin(theta/2))
    w.theta = 2.0 * std::atan2(double(s.rotation.w), double(s.rotation.x));
    w.opacity_logit = s.opacity;
    w.dc = {s.sh[0], s.sh[1], s.sh[2]};
    return oracle::world_to_camera_about_z(w, phi);
}

inline uint64_t fnv1a(uint64_t h, const void* bytes, size_t size) {
    const auto* p = static_cast<const unsigned char*>(bytes);
    for (size_t i = 0; i < size; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

// The quantized value a target channel has in the RGBA8 targets file (test.sh writes round(255 t))
inline double quantized(double t) { return std::round(std::clamp(t, 0.0, 1.0) * 255.0) / 255.0; }

// Checks a frame's readbacks; false when they aren't of one complete frame yet. Exits the program
// on a mismatch.
inline bool check(const zgal::Data<Splat>& splats, const zgal::Data<Camera>& cameras, const zgal::Data<Projected>& projected,
                  const zgal::Data<ProjectedGradient>& gradients, const zgal::Data<float>& losses, uint32_t cameraIndex,
                  glm::vec3 background, glm::vec3 target, bool l1, bool useOracle) {
    if (gradients.size() != splats.size() || projected.size() != splats.size() || losses.size() != 3) return false;
    if (cameraIndex >= cameras.size()) usage("--camera is past the cameras");
    int failures = 0;
    auto fail = [&](const std::string& what) {
        if (failures < kShown) std::cout << "grad_test: MISMATCH " << what << std::endl;
        ++failures;
    };

    uint64_t hash = 0xcbf29ce484222325ull;
    for (uint32_t i = 0; i < gradients.size(); ++i) {
        const ProjectedGradient& g = gradients[i];
        const std::array<float, 9> values = {g.mean.x,  g.mean.y,   g.conic.x,  g.conic.y, g.conic.z,
                                             g.opacity, g.colour.x, g.colour.y, g.colour.z};
        for (float v : values)
            if (!std::isfinite(v)) fail("splat " + std::to_string(i) + "'s gradients aren't finite");
        hash = fnv1a(hash, values.data(), sizeof values);
    }
    const float loss = losses[0];
    if (!std::isfinite(loss)) fail("the loss isn't finite");
    hash = fnv1a(hash, &loss, sizeof loss);
    char hashText[17];
    std::snprintf(hashText, sizeof hashText, "%016llx", static_cast<unsigned long long>(hash));
    std::cout << "grad_test: camera " << cameraIndex << ": " << splats.size() << " splats, " << (l1 ? "L1" : "L2")
              << " loss " << loss << std::endl;

    if (useOracle) {
        if (splats.size() != 1) usage("--oracle needs one splat (S1's gradient variant)");
        const Camera& cam = cameras[cameraIndex];
        const glm::mat4& m = cam.worldToCamera;
        if (cam.intrinsics.x != cam.intrinsics.y || m[3] != glm::vec4(0.0f, 0.0f, 0.0f, 1.0f) || m[2][2] != 1.0f) {
            usage("--oracle needs a camera at the origin rotated about z with fx = fy (S1)");
        }
        const double phi = std::atan2(double(m[0][1]), double(m[0][0]));
        const uint32_t width = cam.size.x, height = cam.size.y;
        const oracle::Camera ocam{cam.intrinsics.x, int(width), int(height), cam.intrinsics.z - 0.5 * width,
                                  cam.intrinsics.w - 0.5 * height};
        const oracle::Projected pr = oracle::project(ocam, oracleSplat(splats[0], phi));
        const oracle::Rgb t{quantized(target.x), quantized(target.y), quantized(target.z)};
        const oracle::Rgb bg{background.x, background.y, background.z};
        const oracle::L2Gradients2d o =
            oracle::projected_gradients(ocam, pr, t, bg, l1 ? oracle::Loss::kL1 : oracle::Loss::kL2);

        const ProjectedGradient& g = gradients[0];
        struct Value {
            const char* name;
            double gpu, oracle;
            int kind;  // 0 mean, 1 conic, 2 opacity, 3 colour, 4 loss
        };
        const std::array<Value, 10> values = {{
            {"dL/dmean.x", g.mean.x, o.mean.x, 0},
            {"dL/dmean.y", g.mean.y, o.mean.y, 0},
            {"dL/dconic.a", g.conic.x, o.conic.a, 1},
            {"dL/dconic.b", g.conic.y, o.conic.b, 1},
            {"dL/dconic.c", g.conic.z, o.conic.c, 1},
            {"dL/dopacity", g.opacity, o.opacity, 2},
            {"dL/dcolour.r", g.colour.x, o.colour[0], 3},
            {"dL/dcolour.g", g.colour.y, o.colour[1], 3},
            {"dL/dcolour.b", g.colour.z, o.colour[2], 3},
            {"the loss", loss, o.loss, 4},
        }};
        std::array<double, 5> largest{};
        for (const Value& v : values) largest[v.kind] = std::max(largest[v.kind], std::abs(v.oracle));
        double worst = 0.0;
        for (const Value& v : values) {
            const double error = std::abs(v.gpu - v.oracle) / std::max(std::abs(v.oracle), kKindFloor * largest[v.kind]);
            worst = std::max(worst, error);
            if (!(error <= kOracleTolerance)) {
                fail(std::string(v.name) + ": " + std::to_string(v.gpu) + ", the oracle's " + std::to_string(v.oracle) +
                     " (relative error " + std::to_string(error) + ")");
            }
        }
        std::cout << "grad_test: against the closed-form oracle: within " << worst << " relative" << std::endl;
    }

    if (failures > 0) {
        std::cout << "grad_test: FAILED, " << failures << " mismatches" << std::endl;
        std::exit(1);
    }
    std::cout << "grad_test: hash " << hashText << std::endl;
    return true;
}

}  // namespace grad_check
