// grad_test's checks, which grad_test.cpp gives the native program and web-check/grad_test_check.cpp
// runs on a web run's readbacks: one frame's gradients with respect to the splats' projections
// (lib/train_graph.zgal's splatGradients) and parameters (parameterGradients), the densification's
// statistics (densifyStats) and the loss
//   - every gradient and the loss finite; a hash of them printed, so test.sh can compare runs;
//   - each splat's parameter gradients the CPU's preprocessBackward (the same ZGAL, compiled to C++)
//     of the GPU's projected gradients, within kKernelTolerance (1e-3) relative to the CPU's or to
//     1e-2 of the largest of its group (position, opacity, log-scale, rotation, SH): the kernel's
//     plumbing (every field written where it belongs, culled splats' 0);
//   - the densification's statistics: every visible splat seen by the same number n >= 1 of
//     frames (they're all the same frame), every culled one by none, and its norm sum n times
//     densifyNorm of its 2D mean's gradient, within 1e-4;
//   - with --oracle (S1's "gradient" variant: one splat blended into every pixel, a constant
//     target), against the closed-form oracle (tests/oracle/closed_form.h's projected_gradients,
//     L2 or L1) at the oracle's projection, with the target rounded to 8 bits as the targets file
//     has it: the loss and each gradient (the mean's x and y, the conic's a, b and c, the opacity,
//     the colour's r, g and b) within kOracleTolerance (1e-3, the plan's bound) relative to the
//     oracle's, or to 1e-2 of the largest of its kind (mean, conic, colour) for one near 0; and
//     the parameters' the oracle has (single_splat_gradients: the world position's x and y, the
//     log-scale, the opacity logit and the SH DC) likewise.
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
#include <vector>

namespace grad_check {

using namespace render_types;

constexpr double kOracleTolerance = 1e-3;
constexpr double kKernelTolerance = 1e-3;
constexpr double kDensifyTolerance = 1e-4;
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
// A splat's parameter gradients as floats, without the padding (position, opacity, log-scale,
// rotation, then the 48 SH coefficients), and the group each is in: 0 position, 1 opacity,
// 2 log-scale, 3 rotation, 4 SH
inline std::array<float, 59> parameterValues(const Splat& g) {
    std::array<float, 59> v{};
    const float head[11] = {g.position.x, g.position.y, g.position.z, g.opacity,    g.logScale.x, g.logScale.y,
                            g.logScale.z, g.rotation.x, g.rotation.y, g.rotation.z, g.rotation.w};
    std::copy(std::begin(head), std::end(head), v.begin());
    std::copy(std::begin(g.sh), std::end(g.sh), v.begin() + 11);
    return v;
}
inline int parameterGroup(int i) { return i < 3 ? 0 : i == 3 ? 1 : i < 7 ? 2 : i < 11 ? 3 : 4; }

// The CPU's parameter gradients for a splat from its projected gradients (TrainGraph's
// BackwardPreprocess, culled splats' 0)
inline Splat cpuParameterGradients(const Splat& s, const Camera& cam, uint32_t shDegree, const Projected& p,
                                   const ProjectedGradient& g) {
    Splat out{};
    if (p.tileCount == 0) return out;
    const SplatGradient r = preprocessBackward(s, cam, shDegree, g);
    out.position = r.position;
    out.opacity = r.opacity;
    out.logScale = r.logScale;
    out.rotation = r.rotation;
    for (uint32_t k = 0; k < 16; ++k) {
        const glm::vec3 c = shGradient(r, k, shDegree);
        out.sh[3 * k] = c.x;
        out.sh[3 * k + 1] = c.y;
        out.sh[3 * k + 2] = c.z;
    }
    return out;
}

inline bool check(const zgal::Data<Splat>& splats, const zgal::Data<Camera>& cameras, const zgal::Data<Projected>& projected,
                  const zgal::Data<ProjectedGradient>& gradients, const zgal::Data<Splat>& parameterGradients,
                  const zgal::Data<DensifyStat>& densify, const zgal::Data<float>& losses, uint32_t cameraIndex,
                  uint32_t shDegree, glm::vec3 background, glm::vec3 target, bool l1, bool useOracle) {
    if (gradients.size() != splats.size() || projected.size() != splats.size() ||
        parameterGradients.size() != splats.size() || densify.size() != splats.size() || losses.size() != 3) {
        return false;
    }
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
        const std::array<float, 59> parameters = parameterValues(parameterGradients[i]);
        for (float v : parameters)
            if (!std::isfinite(v)) fail("splat " + std::to_string(i) + "'s parameter gradients aren't finite");
        hash = fnv1a(hash, parameters.data(), sizeof parameters);
    }
    const float loss = losses[0];
    if (!std::isfinite(loss)) fail("the loss isn't finite");
    hash = fnv1a(hash, &loss, sizeof loss);
    char hashText[17];
    std::snprintf(hashText, sizeof hashText, "%016llx", static_cast<unsigned long long>(hash));
    std::cout << "grad_test: camera " << cameraIndex << ": " << splats.size() << " splats, " << (l1 ? "L1" : "L2")
              << " loss " << loss << std::endl;

    // The kernel against the CPU's preprocessBackward, group by group
    {
        const Camera& cam = cameras[cameraIndex];
        std::vector<std::array<float, 59>> cpu(splats.size());
        std::array<double, 5> largest{};
        for (uint32_t i = 0; i < splats.size(); ++i) {
            cpu[i] = parameterValues(cpuParameterGradients(splats[i], cam, shDegree, projected[i], gradients[i]));
            for (int j = 0; j < 59; ++j) largest[parameterGroup(j)] = std::max(largest[parameterGroup(j)], double(std::abs(cpu[i][j])));
        }
        double worst = 0.0;
        for (uint32_t i = 0; i < splats.size(); ++i) {
            const std::array<float, 59> gpu = parameterValues(parameterGradients[i]);
            for (int j = 0; j < 59; ++j) {
                const double scale = std::max(double(std::abs(cpu[i][j])), kKindFloor * largest[parameterGroup(j)]);
                const double error = scale > 0.0 ? std::abs(double(gpu[j]) - cpu[i][j]) / scale : std::abs(gpu[j]);
                worst = std::max(worst, error);
                if (!(error <= kKernelTolerance)) {
                    fail("splat " + std::to_string(i) + "'s parameter gradient " + std::to_string(j) + ": " +
                         std::to_string(gpu[j]) + ", the CPU's " + std::to_string(cpu[i][j]));
                }
            }
        }
        std::cout << "grad_test: the parameter gradients against the CPU's: within " << worst << " relative" << std::endl;

        // The densification's statistics
        uint32_t seen = 0;
        double densifyWorst = 0.0;
        for (uint32_t i = 0; i < splats.size(); ++i) {
            const DensifyStat& d = densify[i];
            if (projected[i].tileCount == 0) {
                if (d.visible != 0 || d.gradientNorm != 0.0f) fail("culled splat " + std::to_string(i) + " has densification statistics");
                continue;
            }
            if (seen == 0) seen = d.visible;
            if (d.visible == 0 || d.visible != seen) {
                fail("splat " + std::to_string(i) + " was seen by " + std::to_string(d.visible) + " frames, not " +
                     std::to_string(seen));
                continue;
            }
            const double expected = double(d.visible) * densifyNorm(gradients[i].mean, cam.size);
            const double error = std::abs(d.gradientNorm - expected) / std::max(expected, 1e-30);
            densifyWorst = std::max(densifyWorst, expected > 0.0 ? error : double(d.gradientNorm));
            if (!(expected > 0.0 ? error <= kDensifyTolerance : d.gradientNorm == 0.0f)) {
                fail("splat " + std::to_string(i) + "'s densification norm " + std::to_string(d.gradientNorm) + ", not " +
                     std::to_string(expected));
            }
        }
        std::cout << "grad_test: the densification's statistics: " << seen << " frames, within " << densifyWorst
                  << " relative" << std::endl;
    }

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
        const oracle::Splat view = oracleSplat(splats[0], phi);
        const oracle::Rgb t{quantized(target.x), quantized(target.y), quantized(target.z)};
        const oracle::Rgb bg{background.x, background.y, background.z};
        const oracle::L2Gradients full =
            oracle::single_splat_gradients(ocam, view, t, bg, l1 ? oracle::Loss::kL1 : oracle::Loss::kL2);
        const oracle::L2Gradients2d& o = full.screen;
        // The camera-space x and y gradients to the world's: x_c = co x - si y, y_c = si x + co y
        const double co = std::cos(phi), si = std::sin(phi);
        const Splat& pg = parameterGradients[0];

        const ProjectedGradient& g = gradients[0];
        struct Value {
            const char* name;
            double gpu, oracle;
            int kind;  // 0 mean, 1 conic, 2 opacity, 3 colour, 4 loss, 5 position, 6 log-scale, 7 opacity
                       // logit, 8 SH DC
        };
        const std::array<Value, 19> values = {{
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
            {"dL/dposition.x", pg.position.x, co * full.x + si * full.y, 5},
            {"dL/dposition.y", pg.position.y, -si * full.x + co * full.y, 5},
            {"dL/dlogScale.x", pg.logScale.x, full.log_scale[0], 6},
            {"dL/dlogScale.y", pg.logScale.y, full.log_scale[1], 6},
            {"dL/dlogScale.z", pg.logScale.z, full.log_scale[2], 6},
            {"dL/d(opacity logit)", pg.opacity, full.opacity_logit, 7},
            {"dL/dsh[0] (red DC)", pg.sh[0], full.dc[0], 8},
            {"dL/dsh[1] (green DC)", pg.sh[1], full.dc[1], 8},
            {"dL/dsh[2] (blue DC)", pg.sh[2], full.dc[2], 8},
        }};
        std::array<double, 9> largest{};
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
