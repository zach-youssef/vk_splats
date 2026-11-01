// The backward preprocess's math (lib/splat_backward.zgal), through the C++ that ZGAL generates for
// its functions (splat_backward_test.zgal.h, from splat_backward_test.zgal):
//   - against the closed-form oracle (tests/oracle/closed_form.h's single_splat_l2): the gradients
//     of the log-scales, the opacity logit, the SH DC coefficients and the in-plane position, from
//     the oracle's own projected gradients, for splats rotated about z seen by cameras rotated about
//     z;
//   - against central differences of the forward pass (projectSplat) for everything the oracle
//     doesn't cover: general rotations and cameras, the quaternion, depth, SH degrees 1-3 and the
//     view direction, the colour's clamp and the Jacobian's frustum clamp. The loss is a random
//     linear function of the projection (its mean, conic, opacity and colour), so its gradient with
//     respect to the projection is the function's weights.
// The generated C++ computes in float: the oracle comparison holds to 1e-3 relative (to the larger
// of the value and 1e-2 of its kind's largest), and the differences, which lose more to float
// rounding, to 1e-2 (extrapolated from two steps, so the step can be large enough to keep the
// rounding small without truncation error: the worst error is below 5e-3 with clang and GCC).
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include "oracle/closed_form.h"
#include "splat_backward_test.zgal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace sb = splat_backward_test;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr float kNear = 0.2f;
constexpr uint32_t kNoSh = 48;  // shColourPerturbed's "no coefficient"

// A camera at the origin rotated by phi about its z axis (R = Rz(phi), t = 0): X_cam = R X
sb::Camera about_z_camera(const oracle::Camera& c, double phi) {
    sb::Camera cam{};
    const float co = static_cast<float>(std::cos(phi)), si = static_cast<float>(std::sin(phi));
    cam.worldToCamera = glm::mat4(1.0f);
    cam.worldToCamera[0] = glm::vec4(co, si, 0.0f, 0.0f);   // column 0
    cam.worldToCamera[1] = glm::vec4(-si, co, 0.0f, 0.0f);  // column 1
    cam.intrinsics = glm::vec4(c.f, c.f, c.cx(), c.cy());
    cam.center = glm::vec3(0.0f);
    cam.size = glm::uvec2(c.width, c.height);
    cam.imageBatch = 0xFFFFFFFFu;
    return cam;
}

double group_scale(const std::vector<double>& expected) {
    double m = 0.0;
    for (double e : expected) m = std::max(m, std::abs(e));
    return m;
}

// Each of a group's values to tol relative to the larger of its own size and 1e-2 of the group's
// largest
void check_group(const std::string& name, const std::vector<double>& actual, const std::vector<double>& expected,
                 double tol) {
    const double scale = group_scale(expected);
    for (size_t i = 0; i < actual.size(); ++i) {
        const double error = std::abs(actual[i] - expected[i]) / std::max({std::abs(expected[i]), 1e-2 * scale, 1e-30});
        INFO(name << "[" << i << "] = " << actual[i] << ", expected " << expected[i] << " (relative error " << error
                  << ")");
        CHECK(error <= tol);
    }
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Against the closed-form oracle
// ---------------------------------------------------------------------------------------------
TEST_CASE("preprocessBackward matches the closed-form oracle: log-scales, opacity, DC, in-plane position") {
    const oracle::Camera ocam{100.0, 65, 65};
    struct Case {
        const char* name;
        oracle::Splat world;  // in world space; the camera is rotated by phi
        double phi;
    };
    auto splat = [](double x, double y, double z, std::array<double, 3> scale, double theta, double opacity,
                    oracle::Rgb dc) {
        return oracle::Splat{x, y, z, {std::log(scale[0]), std::log(scale[1]), std::log(scale[2])}, theta, opacity, dc};
    };
    const std::vector<Case> cases = {
        {"isotropic on axis", splat(0, 0, 5, {0.2, 0.2, 0.2}, 0.3, 0.4, {1.0, -0.5, 0.25}), 0.0},
        {"anisotropic axis-aligned", splat(0, 0, 4, {0.3, 0.1, 0.2}, 0.0, 0.4, {1.0, -0.5, 0.25}), 0.0},
        {"anisotropic rotated 0.7", splat(0.05, -0.02, 3, {0.25, 0.08, 0.5}, 0.7, 0.4, {1.0, -0.5, 0.25}), 0.0},
        {"off axis isotropic", splat(0.4, -0.3, 4, {0.15, 0.15, 0.15}, 0.0, 0.4, {1.0, -0.5, 0.25}), 0.0},
        {"off axis anisotropic, a clamped colour", splat(-0.5, 0.35, 3.5, {0.3, 0.05, 0.4}, -1.1, -1.5, {-3.0, 0.0, 2.0}),
         0.0},
        {"off axis, camera rotated 0.6", splat(0.3, 0.2, 4, {0.2, 0.07, 0.3}, 0.4, -0.5, {0.5, 1.5, -0.7}), 0.6},
        {"off axis, camera rotated -1.3", splat(-0.2, 0.4, 3, {0.12, 0.2, 0.25}, 1.2, 0.1, {-0.4, 0.3, 0.9}), -1.3},
    };
    const oracle::Rgb target{0.2, 0.7, 0.4}, bg{0.1, 0.3, 0.9};
    for (const Case& c : cases) {
        INFO(c.name);
        const sb::Camera cam = about_z_camera(ocam, c.phi);
        const oracle::Splat view = oracle::world_to_camera_about_z(c.world, c.phi);
        const oracle::L2Gradients o = oracle::single_splat_l2(ocam, view, target, bg);

        sb::Splat s{};
        s.position = glm::vec3(c.world.x, c.world.y, c.world.z);
        s.opacity = static_cast<float>(c.world.opacity_logit);
        s.logScale = glm::vec3(c.world.log_scale[0], c.world.log_scale[1], c.world.log_scale[2]);
        const oracle::Quat q = oracle::quat_about_z(c.world.theta);
        // Not unit length: the gradient is the normalized quaternion's, whatever the length
        s.rotation = 1.7f * glm::vec4(q.w, q.x, q.y, q.z);
        s.sh.fill(0.0f);
        for (int ch = 0; ch < 3; ++ch) s.sh[ch] = static_cast<float>(c.world.dc[ch]);
        REQUIRE(sb::projectSplat(s, cam, kNear, 0).tileCount > 0);

        const oracle::L2Gradients2d& sc = o.screen;
        sb::ProjectedGradient g{};
        g.mean = glm::vec2(sc.mean.x, sc.mean.y);
        g.conic = glm::vec3(sc.conic.a, sc.conic.b, sc.conic.c);
        g.opacity = static_cast<float>(sc.opacity);
        g.colour = glm::vec3(sc.colour[0], sc.colour[1], sc.colour[2]);
        for (uint32_t degree : {0u, 3u}) {
            INFO("SH degree " << degree);
            const sb::SplatGradient r = sb::preprocessBackward(s, cam, degree, g);
            check_group("dL/dlogScale", {r.logScale.x, r.logScale.y, r.logScale.z},
                        {o.log_scale[0], o.log_scale[1], o.log_scale[2]}, 1e-3);
            check_group("dL/dopacity", {r.opacity}, {o.opacity_logit}, 1e-3);
            const glm::vec3 dc = sb::shGradient(r, 0, degree);
            check_group("dL/dDC", {dc.x, dc.y, dc.z}, {o.dc[0], o.dc[1], o.dc[2]}, 1e-3);
            // The oracle's x and y are the camera's; the world's are R^T of them
            const double co = std::cos(c.phi), si = std::sin(c.phi);
            check_group("dL/dposition.xy", {r.position.x, r.position.y},
                        {co * o.x + si * o.y, -si * o.x + co * o.y}, 1e-3);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Against central differences of projectSplat
// ---------------------------------------------------------------------------------------------
namespace {

// The loss: a linear function of the projection, sum of weights times its mean, conic, opacity
// and colour (whose gradient with respect to the projection is the weights)
double linear_loss(const sb::Projected& p, const sb::ProjectedGradient& w) {
    return double(w.mean.x) * p.mean.x + double(w.mean.y) * p.mean.y + double(w.conic.x) * p.conic.x +
           double(w.conic.y) * p.conic.y + double(w.conic.z) * p.conic.z + double(w.opacity) * p.opacity +
           double(w.colour.x) * p.colour.x + double(w.colour.y) * p.colour.y + double(w.colour.z) * p.colour.z;
}

// The splat's float at offset f of the 60 (data-format.md's layout: position 0-2, opacity 3,
// logScale 4-6, rotation 8-11, sh 12-59; 7 is padding)
float& field(sb::Splat& s, int f) {
    if (f < 3) return s.position[f];
    if (f == 3) return s.opacity;
    if (f < 7) return s.logScale[f - 4];
    if (f < 12) return s.rotation[f - 8];
    return s.sh[f - 12];
}

double analytic(const sb::SplatGradient& g, int f, uint32_t degree) {
    if (f < 3) return g.position[f];
    if (f == 3) return g.opacity;
    if (f < 7) return g.logScale[f - 4];
    if (f < 12) return g.rotation[f - 8];
    const int i = f - 12;
    return sb::shGradient(g, i / 3, degree)[i % 3];
}

// A random rotation matrix's quaternion: (w, x, y, z) of a random unit quaternion, scaled
glm::vec4 random_quat(std::mt19937& rng, float length) {
    std::normal_distribution<float> n(0.0f, 1.0f);
    return length * glm::normalize(glm::vec4(n(rng), n(rng), n(rng), n(rng)));
}

glm::mat3 quat_matrix(glm::vec4 q) {
    q = glm::normalize(q);
    const float w = q.x, x = q.y, y = q.z, z = q.w;
    return glm::mat3(glm::vec3(1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)),
                     glm::vec3(2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)),
                     glm::vec3(2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)));
}

// A camera with a random rotation and translation, fx != fy, an off-centre principal point
sb::Camera random_camera(std::mt19937& rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    sb::Camera cam{};
    const glm::mat3 r = quat_matrix(random_quat(rng, 1.0f));
    const glm::vec3 t(u(rng), u(rng), u(rng));
    cam.worldToCamera = glm::mat4(r);
    cam.worldToCamera[3] = glm::vec4(t, 1.0f);
    cam.intrinsics = glm::vec4(110.0f, 90.0f, 52.0f, 38.0f);
    cam.center = -(glm::transpose(r) * t);
    cam.size = glm::uvec2(96, 72);
    cam.imageBatch = 0xFFFFFFFFu;
    return cam;
}

// A world point that camera sees at pixel (px, py) and depth z
glm::vec3 world_at(const sb::Camera& cam, float px, float py, float z) {
    const glm::vec4 in = cam.intrinsics;
    const glm::vec3 t((px - in.z) / in.x * z, (py - in.w) / in.y * z, z);
    const glm::mat3 r(cam.worldToCamera);
    return glm::transpose(r) * (t - glm::vec3(cam.worldToCamera[3]));
}

struct FdStats {
    int checked = 0;
};

// Checks every parameter's gradient of linear_loss(projectSplat(s)) against central differences,
// extrapolated (Richardson) from steps h and 2h
void check_against_differences(const sb::Splat& s, const sb::Camera& cam, uint32_t degree,
                               const sb::ProjectedGradient& w, FdStats& stats) {
    REQUIRE(sb::projectSplat(s, cam, kNear, degree).tileCount > 0);
    const sb::SplatGradient g = sb::preprocessBackward(s, cam, degree, w);
    struct Group {
        const char* name;
        int first, last;  // [first, last)
    };
    const Group groups[] = {{"position", 0, 3}, {"opacity", 3, 4}, {"logScale", 4, 7},
                            {"rotation", 8, 12}, {"sh", 12, 60}};
    for (const Group& group : groups) {
        std::vector<double> actual, expected;
        for (int f = group.first; f < group.last; ++f) {
            const float v = field(const_cast<sb::Splat&>(s), f);
            // Relative steps; positions by a share of the depth, so the mean moves about 0.1 px, and
            // the quaternion's components by a share of its length (the rotation depends only on
            // q / |q|, and a step relative to a small component left the float forward pass's
            // rounding at the check's floor, 1e-4 of the group's largest gradient)
            const float t_z = (glm::mat3(cam.worldToCamera) * s.position + glm::vec3(cam.worldToCamera[3])).z;
            const float h = f < 3                ? 1e-3f * t_z
                            : f >= 12            ? 1e-2f
                            : f >= 8             ? 1e-2f * glm::length(s.rotation)
                                                 : std::max(3e-3f * std::abs(v), 3e-3f);
            // The central difference over [v - k h, v + k h], divided by the step float rounding
            // gives
            auto central = [&](float k, sb::Projected& pp) {
                sb::Splat plus = s, minus = s;
                field(plus, f) = v + k * h;
                field(minus, f) = v - k * h;
                pp = sb::projectSplat(plus, cam, kNear, degree);
                const sb::Projected pm = sb::projectSplat(minus, cam, kNear, degree);
                REQUIRE(pp.tileCount > 0);
                REQUIRE(pm.tileCount > 0);
                return (linear_loss(pp, w) - linear_loss(pm, w)) / (double(field(plus, f)) - double(field(minus, f)));
            };
            // Richardson's extrapolation of the steps h and 2h, (4 D(h) - D(2h)) / 3, cancels the
            // h^2 error: at 1e-2 |q| the quaternion's plain central difference was off by up to
            // 0.9 % (truncation), and at smaller steps by more (rounding)
            sb::Projected pp, pp2;
            const double d1 = central(1.0f, pp), d2 = central(2.0f, pp2);
            expected.push_back((4.0 * d1 - d2) / 3.0);
            actual.push_back(analytic(g, f, degree));
            // shColourPerturbed adds the step to the same coefficient
            if (f >= 12) {
                const sb::Projected viaIndex = sb::projectSplatPerturbed(s, cam, kNear, degree, f - 12, h);
                CHECK(viaIndex.colour == pp.colour);
            }
        }
        check_group(group.name, actual, expected, 1e-2);
        ++stats.checked;
    }
}

// Random weights for a splat's projection p, each term of the loss about as large as the others:
// the conic's entries, ~1 / covariance, weighted by the covariance's scale, and the mean's by
// 1 / the radius, so a shift by the radius matters as much as the opacity
sb::ProjectedGradient random_weights(std::mt19937& rng, const sb::Projected& p) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    sb::ProjectedGradient w{};
    w.mean = glm::vec2(u(rng), u(rng)) / float(std::max(p.radius, 1u));
    w.conic = glm::vec3(u(rng), u(rng), u(rng)) / std::max(std::abs(p.conic.x), std::abs(p.conic.z));
    w.opacity = u(rng);
    w.colour = glm::vec3(u(rng), u(rng), u(rng));
    return w;
}

sb::Splat random_splat(std::mt19937& rng, const sb::Camera& cam, float px, float py) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    sb::Splat s{};
    s.position = world_at(cam, px, py, 3.0f + u(rng));
    s.opacity = u(rng);
    s.logScale = glm::vec3(-2.0f + 0.5f * u(rng), -2.0f + 0.5f * u(rng), -2.0f + 0.5f * u(rng));
    s.rotation = random_quat(rng, 0.5f + std::abs(u(rng)));
    for (float& c : s.sh) c = 0.5f * u(rng);
    return s;
}

}  // namespace

TEST_CASE("preprocessBackward is projectSplat's chain rule: any rotation, camera and SH degree") {
    std::mt19937 rng(17);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    FdStats stats;
    for (int i = 0; i < 24; ++i) {
        const sb::Camera cam = random_camera(rng);
        const uint32_t degree = static_cast<uint32_t>(i % 4);
        // Away from the colour's clamp, which a step could cross (the next case tests it)
        sb::Splat s;
        do s = random_splat(rng, cam, 10.0f + 76.0f * u(rng), 10.0f + 52.0f * u(rng));
        while (glm::any(glm::lessThan(sb::projectSplat(s, cam, kNear, degree).colour, glm::vec3(0.1f))));
        INFO("case " << i << ", SH degree " << degree);
        check_against_differences(s, cam, degree, random_weights(rng, sb::projectSplat(s, cam, kNear, degree)), stats);
    }
    CHECK(stats.checked == 24 * 5);
}

TEST_CASE("preprocessBackward: the colour's clamp at 0 passes no gradient") {
    std::mt19937 rng(5);
    const sb::Camera cam = random_camera(rng);
    sb::Splat s = random_splat(rng, cam, 40.0f, 30.0f);
    s.sh[0] = -5.0f;  // red's DC: far below the clamp whatever the other bands add
    const sb::Projected p = sb::projectSplat(s, cam, kNear, 3);
    REQUIRE(p.colour.x == 0.0f);
    REQUIRE(p.colour.y > 0.0f);
    FdStats stats;
    const sb::ProjectedGradient w = random_weights(rng, p);
    check_against_differences(s, cam, 3, w, stats);
    const sb::SplatGradient g = sb::preprocessBackward(s, cam, 3, w);
    CHECK(g.colour.x == 0.0f);
    for (uint32_t k = 0; k < 16; ++k) CHECK(sb::shGradient(g, k, 3).x == 0.0f);
}

TEST_CASE("preprocessBackward: where the frustum guard clamps x/z, x passes nothing through J") {
    // A large splat whose mean is 1.3 image widths right of the image: x/z is clamped in J, and
    // its square still reaches the image
    std::mt19937 rng(9);
    const sb::Camera cam = random_camera(rng);
    for (float px : {96.0f * 1.3f, -96.0f * 0.35f}) {
        INFO("mean at x = " << px << " px");
        sb::Splat s = random_splat(rng, cam, px, 36.0f);
        s.logScale = glm::vec3(0.0f, -0.3f, -0.5f);
        const sb::Projected p = sb::projectSplat(s, cam, kNear, 2);
        REQUIRE(p.tileCount > 0);
        FdStats stats;
        check_against_differences(s, cam, 2, random_weights(rng, p), stats);
    }
}

TEST_CASE("shGradient is 0 above the degree, and shCoefficientCount is (degree + 1)^2, at most 16") {
    CHECK(sb::shCoefficientCount(0) == 1);
    CHECK(sb::shCoefficientCount(1) == 4);
    CHECK(sb::shCoefficientCount(2) == 9);
    CHECK(sb::shCoefficientCount(3) == 16);
    CHECK(sb::shCoefficientCount(7) == 16);
    sb::SplatGradient g{};
    g.colour = glm::vec3(1.0f, 2.0f, 3.0f);
    g.direction = glm::normalize(glm::vec3(0.3f, -0.4f, 0.8f));
    for (uint32_t degree = 0; degree <= 3; ++degree)
        for (uint32_t k = 0; k < 16; ++k) {
            const glm::vec3 v = sb::shGradient(g, k, degree);
            if (k >= (degree + 1) * (degree + 1)) CHECK(v == glm::vec3(0.0f));
            else CHECK(v == sb::shBasis(g.direction, k) * g.colour);
        }
}

TEST_CASE("densifyNorm is the 2D-mean gradient's norm per NDC unit") {
    CHECK(sb::densifyNorm(glm::vec2(1.0f, 0.0f), glm::uvec2(100, 50)) == doctest::Approx(50.0));
    CHECK(sb::densifyNorm(glm::vec2(0.0f, 2.0f), glm::uvec2(100, 50)) == doctest::Approx(50.0));
    CHECK(sb::densifyNorm(glm::vec2(3.0f / 50.0f, 4.0f / 25.0f), glm::uvec2(100, 50)) == doctest::Approx(std::sqrt(9.0 + 16.0)));
}
