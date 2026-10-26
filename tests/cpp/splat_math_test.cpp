// The splat math module (lib/splat_math.zgal), through the C++ that ZGAL generates for its functions
// (splat_math_test.zgal.h, from splat_math_test.zgal): the projection, conic, radius and colour
// against the closed-form oracles (tests/oracle/closed_form.h), and what the oracles don't cover
// (general rotations, the frustum guard, SH bands 1-3, tiles, sort keys) against independent
// derivations in double here.
//
// Tolerance: the generated C++ computes in float, the oracles in double, so values agree to
// kTol = 1e-5 relative to max(1, |expected|) (absolute below 1). A symmetric 2x2 matrix (a
// covariance or conic) is held to 1e-5 relative to its larger diagonal entry, so a small conic is
// held to its own scale. Integers (radii, tiles, keys) agree exactly.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include "oracle/closed_form.h"
#include "splat_math_test.zgal.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace sm = splat_math_test;

namespace {

constexpr double kTol = 1e-5;
constexpr double kPi = 3.14159265358979323846;

bool close(double actual, double expected, double tol = kTol) {
    return std::abs(actual - expected) <= tol * std::max(1.0, std::abs(expected));
}

#define CHECK_CLOSE(actual, expected)                                                            \
    do {                                                                                         \
        const double a_ = (actual), e_ = (expected);                                            \
        INFO(#actual " = " << a_ << ", expected " << e_);                                        \
        CHECK(close(a_, e_));                                                                    \
    } while (0)

// A symmetric 2x2 (a, b, c) to 1e-5 of max(|a|, |c|), which bounds |b| for a positive-definite one
#define CHECK_SYM2(actual, expected)                                                             \
    do {                                                                                         \
        const glm::vec3 a_ = (actual);                                                           \
        const oracle::Sym2 e_ = (expected);                                                      \
        const double s_ = std::max(std::abs(e_.a), std::abs(e_.c));                              \
        INFO(#actual " = (" << a_.x << ", " << a_.y << ", " << a_.z << "), expected (" << e_.a   \
                            << ", " << e_.b << ", " << e_.c << ")");                             \
        CHECK(std::abs(a_.x - e_.a) <= kTol * s_);                                               \
        CHECK(std::abs(a_.y - e_.b) <= kTol * s_);                                               \
        CHECK(std::abs(a_.z - e_.c) <= kTol * s_);                                               \
    } while (0)

using Mat3d = std::array<std::array<double, 3>, 3>;  // [row][column]

// The canonical camera: R = I, t = 0, f = fx = fy, principal point at the image centre.
sm::Camera canonical_camera(const oracle::Camera& c) {
    sm::Camera cam{};
    cam.worldToCamera = glm::mat4(1.0f);
    cam.intrinsics = glm::vec4(c.f, c.f, c.cx(), c.cy());
    cam.center = glm::vec3(0.0f);
    cam.size = glm::uvec2(c.width, c.height);
    cam.imageBatch = 0xFFFFFFFFu;
    return cam;
}

// A camera rotated by phi about its z axis (R = Rz(phi)) with translation t: X_cam = R X + t.
sm::Camera rotated_camera(const oracle::Camera& c, double phi, glm::vec3 t) {
    sm::Camera cam = canonical_camera(c);
    const float co = static_cast<float>(std::cos(phi)), si = static_cast<float>(std::sin(phi));
    glm::mat4 m(1.0f);
    m[0] = glm::vec4(co, si, 0.0f, 0.0f);   // column 0
    m[1] = glm::vec4(-si, co, 0.0f, 0.0f);  // column 1
    m[3] = glm::vec4(t, 1.0f);
    cam.worldToCamera = m;
    cam.center = -(glm::transpose(glm::mat3(m)) * t);
    return cam;
}

glm::vec4 quat_vec4(oracle::Quat q) {
    return glm::vec4(static_cast<float>(q.w), static_cast<float>(q.x), static_cast<float>(q.y),
                     static_cast<float>(q.z));
}

// The kernel's Splat for an oracle splat at world position p (only DC colour).
sm::Splat kernel_splat(const oracle::Splat& s, glm::vec3 p) {
    sm::Splat k{};
    k.position = p;
    k.opacity = static_cast<float>(s.opacity_logit);
    k.logScale = glm::vec3(s.log_scale[0], s.log_scale[1], s.log_scale[2]);
    k.rotation = quat_vec4(oracle::quat_about_z(s.theta));
    k.sh.fill(0.0f);
    for (int ch = 0; ch < 3; ++ch) k.sh[ch] = static_cast<float>(s.dc[ch]);
    return k;
}

oracle::Splat make_splat(double x, double y, double z, std::array<double, 3> scale, double theta,
                         double opacity = 0.4, oracle::Rgb dc = {1.0, -0.5, 0.25}) {
    return {x, y, z, {std::log(scale[0]), std::log(scale[1]), std::log(scale[2])}, theta, opacity, dc};
}

// The oracle setups: on the axis (isotropic, anisotropic, rotated by 90 degrees, rotated by an
// arbitrary angle) and off the axis, where the Jacobian's third column and sz matter.
struct Case {
    const char* name;
    oracle::Splat splat;
};
const std::vector<Case>& cases() {
    static const std::vector<Case> all = {
        {"isotropic on axis", make_splat(0, 0, 5, {0.2, 0.2, 0.2}, 0.3)},
        {"anisotropic axis-aligned", make_splat(0, 0, 4, {0.3, 0.1, 0.2}, 0.0)},
        {"anisotropic rotated 90", make_splat(0, 0, 4, {0.3, 0.1, 0.2}, kPi / 2)},
        {"anisotropic rotated 0.7", make_splat(0, 0, 3, {0.25, 0.08, 0.5}, 0.7)},
        {"off axis isotropic", make_splat(0.4, -0.3, 4, {0.15, 0.15, 0.15}, 0.0)},
        {"off axis anisotropic", make_splat(-0.5, 0.35, 3.5, {0.3, 0.05, 0.4}, -1.1, -1.5, {-3.0, 0.0, 2.0})},
    };
    return all;
}

const oracle::Camera kCam{100.0, 65, 65};

// The radius is an integer from ceil: comparing it exactly is meaningful only when the continuous
// value isn't within float noise of an integer.
void require_radius_margin(const oracle::Sym2& cov) {
    const double v = 3.0 * std::sqrt(oracle::radius_eigenvalue(cov));
    INFO("3 sqrt(lambda) = " << v);
    REQUIRE(std::abs(v - std::round(v)) > 1e-3);
}

// R of an axis-angle rotation (Rodrigues), independent of the quaternion formula.
Mat3d rodrigues(std::array<double, 3> axis, double angle) {
    const double n = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
    const double x = axis[0] / n, y = axis[1] / n, z = axis[2] / n;
    const double c = std::cos(angle), s = std::sin(angle), C = 1 - c;
    return {{{c + x * x * C, x * y * C - z * s, x * z * C + y * s},
             {y * x * C + z * s, c + y * y * C, y * z * C - x * s},
             {z * x * C - y * s, z * y * C + x * s, c + z * z * C}}};
}

// The quaternion of the same rotation, scaled by k (the module normalizes it).
glm::vec4 axis_angle_quat(std::array<double, 3> axis, double angle, double k) {
    const double n = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
    const double s = std::sin(angle / 2) / n;
    return glm::vec4(k * std::cos(angle / 2), k * s * axis[0], k * s * axis[1], k * s * axis[2]);
}

glm::mat3 to_glm(const Mat3d& m) {
    glm::mat3 g;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) g[c][r] = static_cast<float>(m[r][c]);
    return g;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Against the closed-form oracles
// ---------------------------------------------------------------------------------------------
TEST_CASE("quatToMatrix gives the rotation about z the oracles assume") {
    for (double theta : {0.0, 0.3, kPi / 2, -1.1, 2.5}) {
        const glm::mat3 r = sm::quatToMatrix(quat_vec4(oracle::quat_about_z(theta)));
        const double co = std::cos(theta), si = std::sin(theta);
        const Mat3d expected = {{{co, -si, 0}, {si, co, 0}, {0, 0, 1}}};
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col) CHECK_CLOSE(r[col][row], expected[row][col]);
    }
}

TEST_CASE("covariance3d matches the oracle's covariance about z") {
    for (const Case& c : cases()) {
        INFO(c.name);
        const oracle::Cov3dAboutZ v = oracle::cov3d(c.splat);
        const sm::Splat k = kernel_splat(c.splat, glm::vec3(0.0f));
        const glm::mat3 m = sm::covariance3d(k.logScale, k.rotation);
        CHECK_CLOSE(m[0][0], v.v00);
        CHECK_CLOSE(m[1][0], v.v01);
        CHECK_CLOSE(m[0][1], v.v01);
        CHECK_CLOSE(m[1][1], v.v11);
        CHECK_CLOSE(m[2][2], v.v22);
        CHECK_CLOSE(m[2][0], 0.0);
        CHECK_CLOSE(m[2][1], 0.0);
    }
}

TEST_CASE("projectSplat matches the oracles: mean, covariance, conic, radius, opacity, colour") {
    const sm::Camera cam = canonical_camera(kCam);
    for (const Case& c : cases()) {
        INFO(c.name);
        const oracle::Splat& s = c.splat;
        const sm::Splat k = kernel_splat(s, glm::vec3(s.x, s.y, s.z));
        const oracle::Vec2 mu = oracle::mean2d(kCam, s);
        const oracle::Sym2 cov = oracle::cov2d(kCam, s);
        const oracle::Sym2 conic = oracle::inverse(cov);
        require_radius_margin(cov);

        // The stages
        const glm::vec3 t = sm::toCamera(cam.worldToCamera, k.position);
        const glm::vec3 kcov = sm::covariance2d(t, sm::covariance3d(k.logScale, k.rotation),
                                                glm::mat3(cam.worldToCamera), cam.intrinsics, glm::vec2(cam.size));
        CHECK_SYM2(kcov, cov);
        CHECK_CLOSE(sm::det2(kcov), oracle::det(cov));
        CHECK(sm::radiusPx(kcov) == static_cast<uint32_t>(oracle::radius_px(cov)));

        // The whole preprocess
        const sm::Projected p = sm::projectSplat(k, cam, 0.2f, 0);
        REQUIRE(p.tileCount > 0);
        CHECK_CLOSE(p.mean.x, mu.x);
        CHECK_CLOSE(p.mean.y, mu.y);
        CHECK_CLOSE(p.depth, s.z);
        CHECK_SYM2(p.conic, conic);
        CHECK(p.radius == static_cast<uint32_t>(oracle::radius_px(cov)));
        CHECK_CLOSE(p.opacity, oracle::sigmoid(s.opacity_logit));
        const oracle::Rgb col = oracle::colour(s);
        CHECK_CLOSE(p.colour.x, col[0]);
        CHECK_CLOSE(p.colour.y, col[1]);
        CHECK_CLOSE(p.colour.z, col[2]);
        const glm::uvec4 rect = sm::tileRect(p.mean, p.radius, sm::tileGrid(cam.size));
        CHECK(p.tiles == rect);
        CHECK(p.tileCount == sm::tileCountOf(rect));
    }
}

TEST_CASE("a rotated, translated camera uses W, not W^T (world_to_camera_about_z)") {
    const glm::vec3 t(0.1f, -0.2f, 0.5f);
    for (double phi : {0.6, -1.3, kPi / 2}) {
        const sm::Camera cam = rotated_camera(kCam, phi, t);
        for (const Case& c : cases()) {
            INFO(c.name << ", phi " << phi);
            // The world splat whose camera-space view is c.splat: undo t, then Rz(-phi)
            const double cx = c.splat.x - t.x, cy = c.splat.y - t.y, cz = c.splat.z - t.z;
            oracle::Splat world = c.splat;
            world.x = std::cos(phi) * cx + std::sin(phi) * cy;
            world.y = -std::sin(phi) * cx + std::cos(phi) * cy;
            world.z = cz;
            world.theta = c.splat.theta - phi;
            oracle::Splat view = oracle::world_to_camera_about_z(world, phi);
            view.x += t.x;
            view.y += t.y;
            view.z += t.z;
            REQUIRE(close(view.x, c.splat.x, 1e-12));
            REQUIRE(close(view.theta, c.splat.theta, 1e-12));

            const sm::Projected p = sm::projectSplat(kernel_splat(world, glm::vec3(world.x, world.y, world.z)), cam, 0.2f, 0);
            const oracle::Vec2 mu = oracle::mean2d(kCam, view);
            const oracle::Sym2 conic = oracle::inverse(oracle::cov2d(kCam, view));
            REQUIRE(p.tileCount > 0);
            CHECK_CLOSE(p.mean.x, mu.x);
            CHECK_CLOSE(p.mean.y, mu.y);
            CHECK_CLOSE(p.depth, view.z);
            CHECK_SYM2(p.conic, conic);
            CHECK(p.radius == static_cast<uint32_t>(oracle::radius_px(oracle::cov2d(kCam, view))));
        }
    }
}

TEST_CASE("DC colour: +0.5 and the clamp from below") {
    for (double dc : {-3.0, -1.7724, -1.0, 0.0, 0.5, 2.0}) {
        std::array<float, 48> sh{};
        sh[0] = sh[1] = sh[2] = static_cast<float>(dc);
        const glm::vec3 c = sm::shColour(sh, glm::vec3(0, 0, 1), 3);
        CHECK_CLOSE(c.x, oracle::sh_dc_to_colour(dc));
        CHECK_CLOSE(c.z, oracle::sh_dc_to_colour(dc));
    }
}

TEST_CASE("radiusPx and conicOf on the oracle's matrices, including the 0.1 floor") {
    for (oracle::Sym2 m : {oracle::Sym2{4.3, 0.0, 4.3}, oracle::Sym2{9.5, 0.0, 4.0}, oracle::Sym2{20.5, 3.1, 7.2},
                           oracle::Sym2{5.0, -2.0, 5.0}, oracle::Sym2{100.3, 0.05, 100.31}}) {
        require_radius_margin(m);
        const glm::vec3 v(m.a, m.b, m.c);
        CHECK(sm::radiusPx(v) == static_cast<uint32_t>(oracle::radius_px(m)));
        const oracle::Sym2 q = oracle::inverse(m);
        const glm::vec3 k = sm::conicOf(v);
        CHECK_SYM2(k, q);
    }
}

// ---------------------------------------------------------------------------------------------
// Beyond the oracles: general rotations and the frustum guard
// ---------------------------------------------------------------------------------------------
TEST_CASE("quatToMatrix matches Rodrigues' formula for any axis, and normalizes") {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (int i = 0; i < 50; ++i) {
        const std::array<double, 3> axis = {u(rng), u(rng), u(rng) + 0.01};
        const double angle = 3.0 * u(rng);
        const Mat3d expected = rodrigues(axis, angle);
        const glm::mat3 r = sm::quatToMatrix(axis_angle_quat(axis, angle, 0.2 + 3.0 * (u(rng) + 1.0)));
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col) CHECK_CLOSE(r[col][row], expected[row][col]);
    }
}

TEST_CASE("covariance2d matches J W Sigma W^T J^T with J from central differences, any rotation") {
    // Independent of the module's Jacobian: J is the pinhole projection's derivative, numerically.
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const double fx = 120.0, fy = 90.0, cx = 40.0, cy = 30.0, width = 96.0, height = 64.0;
    for (int i = 0; i < 50; ++i) {
        const Mat3d R = rodrigues({u(rng), u(rng), u(rng) + 0.01}, 3.0 * u(rng));
        const Mat3d W = rodrigues({u(rng), u(rng), u(rng) + 0.01}, 3.0 * u(rng));
        const std::array<double, 3> s = {std::exp(-2.0 + u(rng)), std::exp(-2.0 + u(rng)), std::exp(-2.0 + u(rng))};
        // A camera-space mean inside the image's view
        const double z = 3.0 + 2.0 * u(rng);
        const double t[3] = {(cx + 0.4 * width * u(rng) - cx) * z / fx, (cy + 0.4 * height * u(rng) - cy) * z / fy, z};
        double V[3][3] = {};  // W Sigma W^T in double, Sigma = R S^2 R^T
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                for (int m = 0; m < 3; ++m)
                    for (int l = 0; l < 3; ++l)
                        for (int n = 0; n < 3; ++n)
                            V[a][b] += W[a][m] * R[m][n] * s[n] * s[n] * R[l][n] * W[b][l];
        auto proj = [&](const double p[3], int k) { return k == 0 ? fx * p[0] / p[2] : fy * p[1] / p[2]; };
        double J[2][3];
        for (int k = 0; k < 2; ++k)
            for (int d = 0; d < 3; ++d) {
                const double h = 1e-4 * z;
                double pp[3] = {t[0], t[1], t[2]}, pm[3] = {t[0], t[1], t[2]};
                pp[d] += h;
                pm[d] -= h;
                J[k][d] = (proj(pp, k) - proj(pm, k)) / (2 * h);
            }
        double cov[2][2] = {};
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b)
                for (int m = 0; m < 3; ++m)
                    for (int l = 0; l < 3; ++l) cov[a][b] += J[a][m] * V[m][l] * J[b][l];

        glm::mat3 sigma(0.0f);
        const glm::mat3 r = to_glm(R);
        sigma = r * glm::mat3(glm::vec3(s[0] * s[0], 0, 0), glm::vec3(0, s[1] * s[1], 0), glm::vec3(0, 0, s[2] * s[2])) *
                glm::transpose(r);
        const glm::vec3 k = sm::covariance2d(glm::vec3(t[0], t[1], t[2]), sigma, to_glm(W), glm::vec4(fx, fy, cx, cy),
                                             glm::vec2(width, height));
        const double scale = std::max({1.0, cov[0][0], cov[1][1]});
        INFO("case " << i);
        CHECK(std::abs(k.x - (cov[0][0] + 0.3)) <= kTol * scale);
        CHECK(std::abs(k.y - cov[0][1]) <= kTol * scale);
        CHECK(std::abs(k.z - (cov[1][1] + 0.3)) <= kTol * scale);
    }
}

TEST_CASE("the frustum guard clamps x/z and y/z in the Jacobian at 1.3 half-widths") {
    // Centred principal point: the reference's limit, 1.3 tan(fov / 2) = 1.3 (W / 2) / fx
    const glm::vec4 in(100.0f, 100.0f, 32.0f, 32.0f);
    const glm::vec2 size(64.0f, 64.0f);
    const float lim = 1.3f * 32.0f / 100.0f;
    const glm::mat3 sigma = sm::covariance3d(glm::vec3(std::log(0.2f), std::log(0.1f), std::log(0.3f)),
                                             axis_angle_quat({1, 2, 3}, 0.8, 1.0));
    const glm::mat3 w(1.0f);
    const float z = 2.0f;
    // Beyond the guard on each side, the covariance is the guard's
    for (float sx : {-1.0f, 1.0f})
        for (float sy : {-1.0f, 1.0f}) {
            const glm::vec3 far_out = sm::covariance2d(glm::vec3(sx * 3.0f * lim * z, sy * 2.0f * lim * z, z), sigma, w, in, size);
            const glm::vec3 at_guard = sm::covariance2d(glm::vec3(sx * lim * z, sy * lim * z, z), sigma, w, in, size);
            for (int i = 0; i < 3; ++i) CHECK_CLOSE(far_out[i], at_guard[i]);
        }
    // Inside it, nothing is clamped
    const glm::vec3 inside = sm::covariance2d(glm::vec3(0.9f * lim * z, 0.0f, z), sigma, w, in, size);
    const glm::vec3 at_guard = sm::covariance2d(glm::vec3(lim * z, 0.0f, z), sigma, w, in, size);
    CHECK(std::abs(inside.x - at_guard.x) > 1e-3);
    // An off-centre principal point moves the guard with the image: [(-cx - 0.15 W) / fx, (1.15 W - cx) / fx]
    const glm::vec4 off(100.0f, 100.0f, 10.0f, 32.0f);
    const float lo = (-10.0f - 0.15f * 64.0f) / 100.0f, hi = (1.15f * 64.0f - 10.0f) / 100.0f;
    for (int i = 0; i < 3; ++i) {
        CHECK_CLOSE(sm::covariance2d(glm::vec3(2.0f * lo * z, 0, z), sigma, w, off, size)[i],
                    sm::covariance2d(glm::vec3(lo * z, 0, z), sigma, w, off, size)[i]);
        CHECK_CLOSE(sm::covariance2d(glm::vec3(2.0f * hi * z, 0, z), sigma, w, off, size)[i],
                    sm::covariance2d(glm::vec3(hi * z, 0, z), sigma, w, off, size)[i]);
    }
}

TEST_CASE("projectSplat culls behind the near plane and off the image") {
    const sm::Camera cam = canonical_camera(kCam);
    const oracle::Splat s = make_splat(0, 0, 4, {0.1, 0.1, 0.1}, 0.0);
    for (float z : {-1.0f, 0.0f, 0.1f, 0.2f}) {
        INFO("z = " << z);
        const sm::Projected p = sm::projectSplat(kernel_splat(s, glm::vec3(0, 0, z)), cam, 0.2f, 0);
        CHECK(p.tileCount == 0);
        CHECK(p.radius == 0);
    }
    CHECK(sm::projectSplat(kernel_splat(s, glm::vec3(0, 0, 0.21f)), cam, 0.2f, 0).tileCount > 0);
    // Mean 100 px right of the image with a radius of a few pixels
    const sm::Projected off = sm::projectSplat(kernel_splat(s, glm::vec3(4.0f * (65 + 100 - 32.5f) / 100.0f, 0, 4)), cam, 0.2f, 0);
    CHECK(off.tileCount == 0);
    // Mean on the right edge (x = 65 px), radius 9 px: [56, 74) meets tile columns 3 and 4 of 5
    const sm::Projected edge = sm::projectSplat(kernel_splat(s, glm::vec3(4.0f * (65 - 32.5f) / 100.0f, 0, 4)), cam, 0.2f, 0);
    CHECK(edge.tileCount > 0);
    CHECK(edge.radius == 9);
    CHECK(edge.tiles.x == 3);
    CHECK(edge.tiles.z == 5);  // clamped: a 65-px image has 5 tile columns
}

// Culled: every field of the Projected zero, not just tileCount
static void check_culled(const sm::Projected& p) {
    CHECK(p.tileCount == 0);
    CHECK(p.tiles == glm::uvec4(0));
    CHECK(p.radius == 0);
    CHECK(p.mean == glm::vec2(0.0f));
    CHECK(p.depth == 0.0f);
    CHECK(p.conic == glm::vec3(0.0f));
    CHECK(p.opacity == 0.0f);
    CHECK(p.colour == glm::vec3(0.0f));
}

TEST_CASE("projectSplat culls a splat whose 2D covariance is NaN") {
    // A zero quaternion normalizes to NaN, and an overflowing scale gives inf - inf; both make the
    // determinant NaN, which det <= 0 wouldn't catch (and ceil(NaN) to uint is undefined)
    const sm::Camera cam = canonical_camera(kCam);
    sm::Splat zeroQuat = kernel_splat(make_splat(0, 0, 4, {0.1, 0.1, 0.1}, 0.0), glm::vec3(0, 0, 4));
    zeroQuat.rotation = glm::vec4(0.0f);
    sm::Splat huge = kernel_splat(make_splat(0, 0, 4, {0.1, 0.1, 0.1}, 0.3), glm::vec3(0, 0, 4));
    huge.logScale = glm::vec3(100.0f, 100.0f, 100.0f);
    for (const sm::Splat& s : {zeroQuat, huge}) {
        const glm::vec3 cov = sm::covariance2d(sm::toCamera(cam.worldToCamera, s.position),
                                               sm::covariance3d(s.logScale, s.rotation),
                                               glm::mat3(cam.worldToCamera), cam.intrinsics, glm::vec2(cam.size));
        REQUIRE(std::isnan(sm::det2(cov)));
        // The opacity tells a culled splat from one that slipped through: sigmoid(0.4) isn't 0
        check_culled(sm::projectSplat(s, cam, 0.2f, 0));
    }
}

TEST_CASE("projectSplat: a huge finite scale is culled or gets a clamped radius, never an overflow") {
    // Canonical camera, z = 4, so a = c = (25 exp(l))^2 + 0.3. At logScale 18.6, det = a^2 ~ 8e37 is
    // finite but 3 sqrt(lambda) ~ 9e9 doesn't fit a uint: the radius clamps to radiusMax (2^20). At 20
    // and 30, det overflows to inf and the splat is culled.
    const sm::Camera cam = canonical_camera(kCam);
    for (float l : {18.6f, 20.0f, 30.0f}) {
        INFO("logScale " << l);
        sm::Splat s = kernel_splat(make_splat(0, 0, 4, {0.1, 0.1, 0.1}, 0.0), glm::vec3(0, 0, 4));
        s.logScale = glm::vec3(l);
        const glm::vec3 cov = sm::covariance2d(sm::toCamera(cam.worldToCamera, s.position),
                                               sm::covariance3d(s.logScale, s.rotation),
                                               glm::mat3(cam.worldToCamera), cam.intrinsics, glm::vec2(cam.size));
        REQUIRE(std::isfinite(cov.x));
        const float det = sm::det2(cov);
        const sm::Projected p = sm::projectSplat(s, cam, 0.2f, 0);
        if (l == 18.6f) {
            REQUIRE(std::isfinite(det));
            REQUIRE(3.0 * std::sqrt(double(cov.x)) > 4294967296.0);
            CHECK(sm::radiusPx(cov) == 1048576u);
            CHECK(p.radius == 1048576u);
            CHECK(p.tiles == glm::uvec4(0, 0, 5, 5));  // the whole 65x65 image
            CHECK(p.tileCount == 25);
            CHECK(std::isfinite(p.conic.x));
            CHECK(std::isfinite(p.conic.z));
        } else {
            REQUIRE(std::isinf(det));
            check_culled(p);
        }
    }
}

TEST_CASE("projectSplat with fx != fy, an off-centre principal point and a non-square image") {
    // Only this camera tells fx from fy, cx from cy and width from height, and puts the principal
    // point off the image centre. Expected values in double: the mean from the pinhole model, the
    // covariance J V J^T from a central-difference Jacobian, the radius from the oracle's formula,
    // the tiles by brute force over the 6x4 grid.
    const double fx = 120.0, fy = 90.0, cx = 40.0, cy = 30.0;
    const glm::uvec2 size(96, 64);
    std::mt19937 rng(13);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const Mat3d W = rodrigues({0.3, -0.8, 0.5}, 0.9);
    const double tc[3] = {0.2, -0.1, 0.4};
    sm::Camera cam{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) cam.worldToCamera[c][r] = static_cast<float>(W[r][c]);
        cam.worldToCamera[3][r] = static_cast<float>(tc[r]);
        cam.worldToCamera[r][3] = 0.0f;
    }
    cam.worldToCamera[3][3] = 1.0f;
    for (int r = 0; r < 3; ++r) cam.center[r] = static_cast<float>(-(W[0][r] * tc[0] + W[1][r] * tc[1] + W[2][r] * tc[2]));
    cam.intrinsics = glm::vec4(fx, fy, cx, cy);
    cam.size = size;
    cam.imageBatch = 0xFFFFFFFFu;
    REQUIRE(sm::tileGrid(size) == glm::uvec2(6, 4));

    // Pixel targets for the mean: random ones inside the image, and one in the bottom-right corner
    // tile, where swapping width and height changes the clamp
    std::vector<std::array<double, 2>> targets = {{91.0, 59.0}, {4.0, 60.0}, {93.0, 3.0}};
    for (int i = 0; i < 40; ++i) targets.push_back({48.0 + 44.0 * u(rng), 32.0 + 28.0 * u(rng)});
    int checked = 0;
    for (const auto& target : targets) {
        const double z = 3.0 + 1.5 * u(rng);
        const double t[3] = {(target[0] - cx) * z / fx, (target[1] - cy) * z / fy, z};
        // The world position: p = W^T (t - tc)
        double p[3];
        for (int r = 0; r < 3; ++r) p[r] = W[0][r] * (t[0] - tc[0]) + W[1][r] * (t[1] - tc[1]) + W[2][r] * (t[2] - tc[2]);
        const std::array<double, 3> axis = {u(rng), u(rng), u(rng) + 0.01};
        const double angle = 3.0 * u(rng);
        const Mat3d R = rodrigues(axis, angle);
        const std::array<double, 3> s = {std::exp(-2.5 + u(rng)), std::exp(-2.5 + u(rng)), std::exp(-2.5 + u(rng))};

        sm::Splat k{};
        k.position = glm::vec3(p[0], p[1], p[2]);
        k.opacity = 0.7f;
        k.logScale = glm::vec3(std::log(s[0]), std::log(s[1]), std::log(s[2]));
        k.rotation = axis_angle_quat(axis, angle, 1.7);
        k.sh.fill(0.0f);

        // V = W R S^2 R^T W^T, and Sigma' = J V J^T + 0.3 I with J numerical
        double V[3][3] = {};
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                for (int m = 0; m < 3; ++m)
                    for (int l = 0; l < 3; ++l)
                        for (int n = 0; n < 3; ++n) V[a][b] += W[a][m] * R[m][n] * s[n] * s[n] * R[l][n] * W[b][l];
        auto proj = [&](const double q[3], int c) { return c == 0 ? fx * q[0] / q[2] + cx : fy * q[1] / q[2] + cy; };
        double J[2][3];
        for (int c = 0; c < 2; ++c)
            for (int d = 0; d < 3; ++d) {
                const double h = 1e-4 * z;
                double qp[3] = {t[0], t[1], t[2]}, qm[3] = {t[0], t[1], t[2]};
                qp[d] += h;
                qm[d] -= h;
                J[c][d] = (proj(qp, c) - proj(qm, c)) / (2 * h);
            }
        double cov[2][2] = {};
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b)
                for (int m = 0; m < 3; ++m)
                    for (int l = 0; l < 3; ++l) cov[a][b] += J[a][m] * V[m][l] * J[b][l];
        const oracle::Sym2 sigma2{cov[0][0] + oracle::kLowPass, cov[0][1], cov[1][1] + oracle::kLowPass};
        const double v = 3.0 * std::sqrt(oracle::radius_eigenvalue(sigma2));
        const double mx = proj(t, 0), my = proj(t, 1);
        const int radius = oracle::radius_px(sigma2);
        // Skip cases where float noise could move the radius or a tile edge across an integer
        auto near_int = [](double x) { return std::abs(x - std::round(x)) < 1e-3; };
        if (near_int(v) || near_int((mx - radius) / 16) || near_int((mx + radius) / 16) ||
            near_int((my - radius) / 16) || near_int((my + radius) / 16))
            continue;
        ++checked;

        INFO("target (" << target[0] << ", " << target[1] << "), z " << z);
        const sm::Projected pr = sm::projectSplat(k, cam, 0.2f, 0);
        REQUIRE(pr.tileCount > 0);
        CHECK_CLOSE(pr.mean.x, mx);
        CHECK_CLOSE(pr.mean.y, my);
        CHECK_CLOSE(pr.depth, z);
        CHECK_SYM2(pr.conic, oracle::inverse(sigma2));
        CHECK(pr.radius == static_cast<uint32_t>(radius));
        // Tiles: those of the 6x4 grid meeting (m - r, m + r) on each axis
        const double lo[2] = {mx - radius, my - radius}, hi[2] = {mx + radius, my + radius};
        const uint32_t gridSize[2] = {6, 4};
        for (int axis = 0; axis < 2; ++axis) {
            uint32_t first = gridSize[axis], last = 0;
            for (uint32_t t16 = 0; t16 < gridSize[axis]; ++t16)
                if (16.0 * t16 < hi[axis] && 16.0 * t16 + 16 > lo[axis]) {
                    first = std::min(first, t16);
                    last = t16 + 1;
                }
            CHECK(pr.tiles[axis] == first);
            CHECK(pr.tiles[axis + 2] == last);
        }
        CHECK(pr.tileCount == (pr.tiles.z - pr.tiles.x) * (pr.tiles.w - pr.tiles.y));
    }
    CHECK(checked >= 30);
}

// ---------------------------------------------------------------------------------------------
// Spherical harmonics, bands 1-3
// ---------------------------------------------------------------------------------------------
namespace {

// The real SH basis in the reference code's order and signs (m = -l..l), from its closed forms with
// the normalizations written out, independent of the module's constants.
double sh_basis(int k, double x, double y, double z) {
    const double pi = kPi;
    switch (k) {
        case 0: return 0.5 * std::sqrt(1 / pi);
        case 1: return -std::sqrt(3 / (4 * pi)) * y;
        case 2: return std::sqrt(3 / (4 * pi)) * z;
        case 3: return -std::sqrt(3 / (4 * pi)) * x;
        case 4: return 0.5 * std::sqrt(15 / pi) * x * y;
        case 5: return -0.5 * std::sqrt(15 / pi) * y * z;
        case 6: return 0.25 * std::sqrt(5 / pi) * (2 * z * z - x * x - y * y);
        case 7: return -0.5 * std::sqrt(15 / pi) * x * z;
        case 8: return 0.25 * std::sqrt(15 / pi) * (x * x - y * y);
        case 9: return -0.25 * std::sqrt(35 / (2 * pi)) * y * (3 * x * x - y * y);
        case 10: return 0.5 * std::sqrt(105 / pi) * x * y * z;
        case 11: return -0.25 * std::sqrt(21 / (2 * pi)) * y * (4 * z * z - x * x - y * y);
        case 12: return 0.25 * std::sqrt(7 / pi) * z * (2 * z * z - 3 * x * x - 3 * y * y);
        case 13: return -0.25 * std::sqrt(21 / (2 * pi)) * x * (4 * z * z - x * x - y * y);
        case 14: return 0.25 * std::sqrt(105 / pi) * z * (x * x - y * y);
        case 15: return -0.25 * std::sqrt(35 / (2 * pi)) * x * (x * x - 3 * y * y);
    }
    return 0.0;
}

constexpr float kShProbe = 0.1f;  // small enough that 0.5 + 0.1 Y never clamps (|Y| < 1 for l <= 3)

// Y_k(d) through the module: shColour of coefficient k alone, in channel ch.
double module_basis(int k, int ch, glm::vec3 d, uint32_t degree) {
    std::array<float, 48> sh{};
    sh[3 * k + ch] = kShProbe;
    const glm::vec3 c = sm::shColour(sh, d, degree);
    return (c[ch] - 0.5) / kShProbe;
}

int band(int k) { return k == 0 ? 0 : k < 4 ? 1 : k < 9 ? 2 : 3; }

}  // namespace

TEST_CASE("shColour evaluates each coefficient's basis function in the reference's order and signs") {
    std::mt19937 rng(3);
    std::normal_distribution<double> n(0.0, 1.0);
    for (int i = 0; i < 40; ++i) {
        double x = n(rng), y = n(rng), z = n(rng);
        const double len = std::sqrt(x * x + y * y + z * z);
        x /= len, y /= len, z /= len;
        const glm::vec3 d(x, y, z);
        for (int k = 0; k < 16; ++k)
            for (int ch = 0; ch < 3; ++ch) {
                INFO("k " << k << " ch " << ch);
                // float noise: 0.5 + 0.1 Y loses about 6e-8 / 0.1 of Y
                CHECK(std::abs(module_basis(k, ch, d, 3) - sh_basis(k, x, y, z)) <= 1e-5);
            }
    }
}

TEST_CASE("shColour ignores bands above the degree, and keeps channels apart") {
    const glm::vec3 d = glm::normalize(glm::vec3(0.3f, -0.5f, 0.8f));
    for (uint32_t degree = 0; degree <= 3; ++degree)
        for (int k = 0; k < 16; ++k) {
            const double y = module_basis(k, 1, d, degree);
            if (band(k) > static_cast<int>(degree)) CHECK(y == 0.0);
            else CHECK(std::abs(y - sh_basis(k, d.x, d.y, d.z)) <= 1e-5);
            std::array<float, 48> sh{};
            sh[3 * k + 1] = kShProbe;
            const glm::vec3 c = sm::shColour(sh, d, degree);
            CHECK(c.x == 0.5f);
            CHECK(c.z == 0.5f);
        }
}

TEST_CASE("the SH basis is orthonormal over the sphere") {
    // Gauss-Legendre in z (8 nodes: exact for the degree <= 6 products) times a uniform rule in the
    // azimuth (16 nodes: exact for trigonometric degree <= 6)
    constexpr int kNz = 8, kNphi = 16;
    std::array<double, kNz> nodes{}, weights{};
    for (int i = 0; i < kNz; ++i) {
        double x = std::cos(kPi * (i + 0.75) / (kNz + 0.5)), dp = 0.0;
        for (int it = 0; it < 100; ++it) {
            double p0 = 1.0, p1 = x;
            for (int k = 2; k <= kNz; ++k) {
                const double p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k;
                p0 = p1;
                p1 = p2;
            }
            dp = kNz * (x * p1 - p0) / (x * x - 1);
            x -= p1 / dp;
        }
        nodes[i] = x;
        weights[i] = 2.0 / ((1 - x * x) * dp * dp);
    }
    std::array<std::array<double, 16>, 16> gram{};
    for (int i = 0; i < kNz; ++i)
        for (int j = 0; j < kNphi; ++j) {
            const double z = nodes[i], r = std::sqrt(1 - z * z), phi = 2 * kPi * (j + 0.5) / kNphi;
            const glm::vec3 d(r * std::cos(phi), r * std::sin(phi), z);
            const double w = weights[i] * 2 * kPi / kNphi;
            std::array<double, 16> y{};
            for (int k = 0; k < 16; ++k) y[k] = module_basis(k, 0, d, 3);
            for (int a = 0; a < 16; ++a)
                for (int b = 0; b < 16; ++b) gram[a][b] += w * y[a] * y[b];
        }
    for (int a = 0; a < 16; ++a)
        for (int b = 0; b < 16; ++b) {
            INFO("<Y" << a << ", Y" << b << "> = " << gram[a][b]);
            CHECK(std::abs(gram[a][b] - (a == b ? 1.0 : 0.0)) <= 1e-5);
        }
}

TEST_CASE("projectSplat's view direction runs from the camera centre to the splat") {
    // R = I, t = (0, 0, 2): the centre is (0, 0, -2). A splat at (1, 0, 3) is seen along
    // (1, 0, 5) / sqrt(26); coefficient 3 is -C1 x.
    const sm::Camera cam = rotated_camera(kCam, 0.0, glm::vec3(0, 0, 2));
    sm::Splat k = kernel_splat(make_splat(0, 0, 0, {0.05, 0.05, 0.05}, 0.0, 0.0, {0, 0, 0}), glm::vec3(1, 0, 3));
    k.sh[3 * 3 + 0] = 0.5f;
    const sm::Projected p = sm::projectSplat(k, cam, 0.2f, 1);
    REQUIRE(p.tileCount > 0);
    CHECK_CLOSE(p.colour.x, 0.5 + 0.5 * sh_basis(3, 1 / std::sqrt(26.0), 0, 5 / std::sqrt(26.0)));
    CHECK_CLOSE(p.colour.y, 0.5);
    // Degree 0 ignores it
    CHECK_CLOSE(sm::projectSplat(k, cam, 0.2f, 0).colour.x, 0.5);
}

// ---------------------------------------------------------------------------------------------
// Tiles
// ---------------------------------------------------------------------------------------------
TEST_CASE("tileGrid rounds up") {
    CHECK(sm::tileGrid(glm::uvec2(16, 17)) == glm::uvec2(1, 2));
    CHECK(sm::tileGrid(glm::uvec2(748, 748)) == glm::uvec2(47, 47));
    CHECK(sm::tileGrid(glm::uvec2(160, 96)) == glm::uvec2(10, 6));
    CHECK(sm::tileGrid(glm::uvec2(1, 1)) == glm::uvec2(1, 1));
}

TEST_CASE("tileRect holds exactly the tiles meeting [mean - r, mean + r), clamped to the grid") {
    const glm::uvec2 size(70, 50);
    const glm::uvec2 grid = sm::tileGrid(size);
    REQUIRE(grid == glm::uvec2(5, 4));
    std::mt19937 rng(5);
    // Means on a 1/64 grid, so mean +- r is exact in float, plus edges exactly on tile boundaries
    std::uniform_int_distribution<int> coord(-40 * 64, 120 * 64);
    std::uniform_int_distribution<int> rad(0, 40);
    std::vector<std::pair<glm::vec2, uint32_t>> samples = {
        {{16.0f, 16.0f}, 0u}, {{20.0f, 20.0f}, 4u}, {{36.0f, 12.0f}, 4u}, {{-3.0f, 25.0f}, 3u}, {{80.0f, 60.0f}, 10u}};
    for (int i = 0; i < 2000; ++i)
        samples.push_back({{coord(rng) / 64.0f, coord(rng) / 64.0f}, static_cast<uint32_t>(rad(rng))});
    for (const auto& [mean, r] : samples) {
        INFO("mean (" << mean.x << ", " << mean.y << "), r " << r);
        const glm::uvec4 rect = sm::tileRect(mean, r, grid);
        // Expected: tile k meets [lo, hi) when 16k < hi and 16k + 16 > lo
        for (int axis = 0; axis < 2; ++axis) {
            const double lo = mean[axis] - double(r), hi = mean[axis] + double(r);
            uint32_t first = grid[axis], last = 0;
            for (uint32_t k = 0; k < grid[axis]; ++k)
                if (16.0 * k < hi && 16.0 * k + 16 > lo) {
                    first = std::min(first, k);
                    last = k + 1;
                }
            if (last == 0) {
                CHECK(rect[axis] == rect[axis + 2]);  // empty on this axis
            } else {
                CHECK(rect[axis] == first);
                CHECK(rect[axis + 2] == last);
            }
            CHECK(rect[axis] <= rect[axis + 2]);
            CHECK(rect[axis + 2] <= grid[axis]);
        }
        // Every pixel whose centre is in the closed square is in a tile of the rectangle
        uint64_t covered = 0;
        for (uint32_t j = 0; j < size.y; ++j)
            for (uint32_t i = 0; i < size.x; ++i)
                if (std::abs(i + 0.5 - mean.x) <= r && std::abs(j + 0.5 - mean.y) <= r) {
                    ++covered;
                    CHECK((i / 16 >= rect.x && i / 16 < rect.z && j / 16 >= rect.y && j / 16 < rect.w));
                }
        if (covered > 0) CHECK(sm::tileCountOf(rect) > 0);
    }
}

// ---------------------------------------------------------------------------------------------
// Sort keys
// ---------------------------------------------------------------------------------------------
TEST_CASE("tileBitsFor and depthBitsFor") {
    const std::vector<std::pair<uint32_t, uint32_t>> expected = {
        {0, 1}, {1, 1}, {2, 1}, {3, 2}, {4, 2}, {5, 3}, {2209, 12}, {4096, 12}, {4097, 13},
        {1u << 20, 20}, {(1u << 20) + 1, 21}, {1u << 31, 31}, {0xFFFFFFFFu, 31}};
    for (const auto& [count, bits] : expected) {
        INFO("tileCount " << count);
        CHECK(sm::tileBitsFor(count) == bits);
        CHECK(sm::depthBitsFor(count) == 32 - bits);
        if (count > 0 && count <= (1u << 31)) CHECK(uint64_t(count) <= (uint64_t(1) << bits));
    }
}

TEST_CASE("depthKey: tile in the high bits, log depth below, monotonic and clamped") {
    const float near = 0.2f, far = 100.0f;
    for (uint32_t tileCount : {1u, 2209u, 1u << 20}) {
        const uint32_t bits = sm::depthBitsFor(tileCount);
        const uint32_t maxQ = bits == 32 ? 0xFFFFFFFFu : (1u << bits) - 1;
        INFO("tileCount " << tileCount << ", depth bits " << bits);
        for (uint32_t tile : {0u, tileCount / 2, tileCount - 1}) {
            CHECK(sm::keyTile(sm::depthKey(tile, 1.0f, bits, near, far), bits) == tile);
            // Clamped at both ends
            CHECK((sm::depthKey(tile, near, bits, near, far) & maxQ) == 0);
            CHECK((sm::depthKey(tile, 0.01f, bits, near, far) & maxQ) == 0);
            CHECK((sm::depthKey(tile, far, bits, near, far) & maxQ) == maxQ);
            CHECK((sm::depthKey(tile, 1e6f, bits, near, far) & maxQ) == maxQ);
            // Log scale: the midpoint of the logs quantizes to half the range
            const float mid = std::sqrt(near * far);
            const double q = sm::depthKey(tile, mid, bits, near, far) & maxQ;
            CHECK(std::abs(q / maxQ - 0.5) <= std::max(1e-5, 2.0 / maxQ));
            // Front to back within a tile
            uint32_t previous = 0;
            for (int i = 0; i <= 200; ++i) {
                const float depth = near * std::pow(far / near, i / 200.0f);
                const uint32_t key = sm::depthKey(tile, depth, bits, near, far);
                CHECK(sm::keyTile(key, bits) == tile);
                if (i > 0) CHECK(key >= previous);
                previous = key;
            }
        }
        // Tile-major: any depth in tile t sorts before any in tile t + 1
        if (tileCount > 1)
            CHECK(sm::depthKey(0, far, bits, near, far) < sm::depthKey(1, near, bits, near, far));
    }
    // 2209 tiles leave 20 bits: depths 0.1% apart get different keys
    const uint32_t bits = sm::depthBitsFor(2209);
    CHECK(bits == 20);
    CHECK(sm::depthKey(7, 5.0f, bits, near, far) < sm::depthKey(7, 5.005f, bits, near, far));
}
