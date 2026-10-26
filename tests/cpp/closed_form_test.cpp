// Self-tests of the closed-form oracles (tests/oracle/closed_form.h): limit cases, symmetries, the
// closed forms against a direct evaluation of Eqs. 5, 6 and 10 (also through a rotated camera), and
// the gradients against central differences of the oracle's loss, which is tied to render().
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include "oracle/closed_form.h"

#include <array>
#include <cmath>
#include <functional>

using namespace oracle;

namespace {

constexpr double kPi = 3.14159265358979323846;

// An on-axis splat with log-scales from scales.
Splat on_axis(double z, double sx, double sy, double sz, double theta = 0.0, double opacity = 0.0) {
    return {0.0, 0.0, z, {std::log(sx), std::log(sy), std::log(sz)}, theta, opacity, {1.0, 0.0, -1.0}};
}

using Mat3 = std::array<std::array<double, 3>, 3>;
constexpr Mat3 kIdentity{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};

// R from a quaternion (Eq. 10).
Mat3 rotation(Quat q) {
    const double n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    const double r = q.w / n, i = q.x / n, j = q.y / n, k = q.z / n;
    return {{{1 - 2 * (j * j + k * k), 2 * (i * j - r * k), 2 * (i * k + r * j)},
             {2 * (i * j + r * k), 1 - 2 * (i * i + k * k), 2 * (j * k - r * i)},
             {2 * (i * k - r * j), 2 * (j * k + r * i), 1 - 2 * (i * i + j * j)}}};
}

// The general EWA projection, evaluated directly from the paper's equations rather than from the
// closed forms: a world-space splat at p with rotation q seen by a camera with world-to-camera
// rotation W (t = 0): camera-space mean W p, Sigma = R S S^T R^T (Eq. 6), Sigma' = J W Sigma W^T
// J^T (Eq. 5), plus the low-pass term.
Sym2 cov2d_direct(const Camera& cam, const Mat3& W, std::array<double, 3> p, const std::array<double, 3>& scale,
                  Quat q) {
    const Mat3 R = rotation(q);
    double sigma[3][3] = {};
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            for (int m = 0; m < 3; ++m) sigma[a][b] += R[a][m] * scale[m] * scale[m] * R[b][m];
    double cam_p[3] = {};
    double v[3][3] = {};
    for (int a = 0; a < 3; ++a)
        for (int m = 0; m < 3; ++m) cam_p[a] += W[a][m] * p[m];
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            for (int m = 0; m < 3; ++m)
                for (int l = 0; l < 3; ++l) v[a][b] += W[a][m] * sigma[m][l] * W[b][l];
    const double x = cam_p[0], y = cam_p[1], z = cam_p[2];
    const double J[2][3] = {{cam.f / z, 0.0, -cam.f * x / (z * z)}, {0.0, cam.f / z, -cam.f * y / (z * z)}};
    double out[2][2] = {};
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b)
            for (int m = 0; m < 3; ++m)
                for (int l = 0; l < 3; ++l) out[a][b] += J[a][m] * v[m][l] * J[b][l];
    return {out[0][0] + kLowPass, out[0][1], out[1][1] + kLowPass};
}

Sym2 cov2d_direct(const Camera& cam, double x, double y, double z, const std::array<double, 3>& scale, Quat q) {
    return cov2d_direct(cam, kIdentity, {x, y, z}, scale, q);
}

void check_sym2(const Sym2& got, const Sym2& want, double eps = 1e-12) {
    CHECK(got.a == doctest::Approx(want.a).epsilon(eps));
    CHECK(got.b == doctest::Approx(want.b).epsilon(eps).scale(1.0));
    CHECK(got.c == doctest::Approx(want.c).epsilon(eps));
}

// A camera, splat, background and target where every pixel is blended and none is clamped, so the
// loss is smooth and finite differences are exact up to O(h^2).
const Camera kGradCam{20.0, 15, 15};
const Rgb kBg{0.1, 0.2, 0.3};
const Rgb kTarget{0.7, 0.4, 0.05};

Splat grad_splat() {
    // f s / z = 20 * 0.45 / 2 = 4.5 px along x; a smaller y and z scale, a tilt, an offset mean
    return {0.15, -0.1, 2.0, {std::log(0.45), std::log(0.35), std::log(0.3)}, 0.4, 0.8, {0.8, -0.6, 0.2}};
}

double central_difference(const std::function<double(double)>& loss, double h) {
    return (loss(h) - loss(-h)) / (2.0 * h);
}

double loss_of(const Splat& s) { return single_splat_l2(kGradCam, s, kTarget, kBg).loss; }

}  // namespace

TEST_CASE("constants and activations") {
    CHECK(kShC0 == doctest::Approx(1.0 / (2.0 * std::sqrt(kPi))).epsilon(1e-15));
    CHECK(sigmoid(0.0) == 0.5);
    CHECK(sigmoid(3.0) + sigmoid(-3.0) == doctest::Approx(1.0));
    // A zero DC coefficient is mid-grey; colours are clamped from below only.
    CHECK(sh_dc_to_colour(0.0) == 0.5);
    CHECK(sh_dc_to_colour(-0.5 / kShC0) == doctest::Approx(0.0).scale(1.0));
    CHECK(sh_dc_to_colour(-10.0) == 0.0);
    CHECK(sh_dc_to_colour(10.0) == doctest::Approx(kShC0 * 10.0 + 0.5));
    CHECK(sh_dc_to_colour_derivative(-10.0) == 0.0);
    CHECK(sh_dc_to_colour_derivative(1.0) == kShC0);
}

TEST_CASE("quat_about_z gives Eq. 10's rotation about z") {
    const double theta = 0.7;
    const Quat q = quat_about_z(theta);
    CHECK(q.w * q.w + q.z * q.z == doctest::Approx(1.0));
    // Rotating a covariance by theta via the quaternion matches cov3d's closed form.
    const Camera cam{50.0, 33, 33};
    const Splat s = on_axis(3.0, 0.4, 0.1, 0.2, theta);
    check_sym2(cov2d(cam, s), cov2d_direct(cam, 0.0, 0.0, 3.0, {0.4, 0.1, 0.2}, q));
}

TEST_CASE("2D covariance: closed-form cases") {
    const Camera cam{50.0, 33, 33};
    const double z = 4.0;

    SUBCASE("isotropic on the axis: (f s / z)^2 + 0.3 for any rotation") {
        const double s = 0.2, want = (cam.f * s / z) * (cam.f * s / z) + 0.3;
        for (double theta : {0.0, 0.3, kPi / 2, 2.0}) {
            const Sym2 c = cov2d(cam, on_axis(z, s, s, s, theta));
            CHECK(c.a == doctest::Approx(want));
            CHECK(c.b == doctest::Approx(0.0).scale(1.0));
            CHECK(c.c == doctest::Approx(want));
        }
    }
    SUBCASE("anisotropic, axis-aligned, and its 90-degree rotation swaps the axes") {
        const double sx = 0.3, sy = 0.1;
        const double ax = (cam.f * sx / z) * (cam.f * sx / z) + 0.3;
        const double ay = (cam.f * sy / z) * (cam.f * sy / z) + 0.3;
        const Sym2 c0 = cov2d(cam, on_axis(z, sx, sy, 0.5));
        CHECK(c0.a == doctest::Approx(ax));
        CHECK(c0.b == 0.0);
        CHECK(c0.c == doctest::Approx(ay));
        const Sym2 c90 = cov2d(cam, on_axis(z, sx, sy, 0.5, kPi / 2));
        CHECK(c90.a == doctest::Approx(ay));
        CHECK(c90.b == doctest::Approx(0.0).scale(1.0));
        CHECK(c90.c == doctest::Approx(ax));
        // 180 degrees is the identity on a covariance
        check_sym2(cov2d(cam, on_axis(z, sx, sy, 0.5, kPi)), c0);
    }
    SUBCASE("a rotation keeps the trace and determinant (the eigenvalues)") {
        const Sym2 c0 = cov2d(cam, on_axis(z, 0.3, 0.1, 0.5));
        const Sym2 c = cov2d(cam, on_axis(z, 0.3, 0.1, 0.5, kPi / 4));
        CHECK(c.a + c.c == doctest::Approx(c0.a + c0.c));
        CHECK(det(c) == doctest::Approx(det(c0)));
        CHECK(c.b == doctest::Approx(0.5 * (c0.a - c0.c)));
    }
    SUBCASE("on the axis, sz has no effect; off the axis it does") {
        check_sym2(cov2d(cam, on_axis(z, 0.3, 0.1, 0.01)), cov2d(cam, on_axis(z, 0.3, 0.1, 5.0)));
        Splat a = on_axis(z, 0.3, 0.1, 0.01), b = on_axis(z, 0.3, 0.1, 5.0);
        a.x = b.x = 0.5;
        CHECK(cov2d(cam, b).a > cov2d(cam, a).a);
    }
    SUBCASE("matches Eqs. 5, 6 and 10 evaluated directly, on and off the axis") {
        for (double x : {0.0, 0.4, -0.7})
            for (double y : {0.0, 0.25, -0.5})
                for (double theta : {0.0, 0.9, -2.3}) {
                    Splat s = on_axis(z, 0.25, 0.08, 0.15, theta);
                    s.x = x;
                    s.y = y;
                    check_sym2(cov2d(cam, s), cov2d_direct(cam, x, y, z, {0.25, 0.08, 0.15}, quat_about_z(theta)));
                }
    }
    SUBCASE("the low-pass term bounds a tiny splat's covariance from below") {
        const Sym2 c = cov2d(cam, on_axis(z, 1e-6, 1e-6, 1e-6));
        CHECK(c.a == doctest::Approx(0.3));
        CHECK(c.c == doctest::Approx(0.3));
    }
}

TEST_CASE("conic, eigenvalue and radius") {
    const Sym2 m{5.0, 1.5, 2.0};
    const Sym2 q = inverse(m);
    // m q = I
    CHECK(m.a * q.a + m.b * q.b == doctest::Approx(1.0));
    CHECK(m.a * q.b + m.b * q.c == doctest::Approx(0.0).scale(1.0));
    CHECK(m.b * q.b + m.c * q.c == doctest::Approx(1.0));
    // eigenvalues: lambda^2 - 7 lambda + 7.75 = 0
    CHECK(max_eigenvalue(m) == doctest::Approx(3.5 + std::sqrt(3.5 * 3.5 - 7.75)));
    CHECK(max_eigenvalue(Sym2{2.0, 0.0, 9.0}) == doctest::Approx(9.0));
    CHECK(max_eigenvalue(Sym2{4.0, 0.0, 4.0}) == doctest::Approx(4.0));
    CHECK_THROWS_AS(inverse(Sym2{1.0, 2.0, 1.0}), std::invalid_argument);

    // The radius's eigenvalue floors mid^2 - det at 0.1 (reference code): no effect once the
    // discriminant is above it, sqrt(0.1) more for an isotropic covariance.
    CHECK(radius_eigenvalue(Sym2{2.0, 0.0, 9.0}) == doctest::Approx(9.0));
    CHECK(radius_eigenvalue(m) == doctest::Approx(max_eigenvalue(m)));
    CHECK(radius_eigenvalue(Sym2{4.0, 0.0, 4.0}) == doctest::Approx(4.0 + std::sqrt(0.1)));
    CHECK(radius_eigenvalue(Sym2{4.1, 0.0, 4.0}) == doctest::Approx(4.05 + std::sqrt(0.1)));

    const Camera cam{50.0, 33, 33};
    const double s = 0.2, z = 4.0;
    CHECK(radius_px(cov2d(cam, on_axis(z, s, s, s))) ==
          static_cast<int>(std::ceil(3.0 * std::sqrt((cam.f * s / z) * (cam.f * s / z) + 0.3 + std::sqrt(0.1)))));
    // Small isotropic splats, where the floor adds a pixel (the exact eigenvalue gives 3 and 2).
    CHECK(radius_px(cov2d(cam, on_axis(z, 0.05, 0.05, 0.05))) == 4);
    CHECK(radius_px(cov2d(cam, on_axis(z, 1e-6, 1e-6, 1e-6))) == 3);
    CHECK(radius_px(Sym2{4.0, 0.0, 1.0}) == 6);
    CHECK(radius_px(Sym2{4.01, 0.0, 1.0}) == 7);
}

TEST_CASE("mean projection follows COLMAP's pinhole") {
    const Camera cam{100.0, 64, 48};
    Splat s = on_axis(5.0, 0.1, 0.1, 0.1);
    CHECK(mean2d(cam, s).x == 32.0);
    CHECK(mean2d(cam, s).y == 24.0);
    s.x = 0.5;
    s.y = 1.0;  // +y is down: a larger row
    CHECK(mean2d(cam, s).x == doctest::Approx(42.0));
    CHECK(mean2d(cam, s).y == doctest::Approx(44.0));
    s.z = -1.0;
    CHECK_THROWS_AS(mean2d(cam, s), std::invalid_argument);
}

TEST_CASE("one splat: per-pixel value") {
    // An odd width puts pixel (16, 16)'s centre on the principal point.
    const Camera cam{50.0, 33, 33};
    const Rgb bg{0.1, 0.2, 0.3};
    const Splat s = on_axis(4.0, 0.3, 0.1, 0.2, 0.0, 0.5);
    const Rgb c = colour(s);
    const double o = sigmoid(0.5);

    SUBCASE("at the mean: c sigmoid(o) + bg (1 - sigmoid(o))") {
        CHECK(gaussian_weight(cam, s, pixel_centre(16, 16)) == 1.0);
        const Rgb v = one_splat_value(cam, s, pixel_centre(16, 16), bg);
        for (int ch = 0; ch < 3; ++ch) CHECK(v[ch] == doctest::Approx(c[ch] * o + bg[ch] * (1 - o)));
    }
    SUBCASE("exp(-1/2 d^T Sigma'^-1 d) along the axes") {
        const Sym2 cov = cov2d(cam, s);
        CHECK(gaussian_weight(cam, s, pixel_centre(19, 16)) == doctest::Approx(std::exp(-0.5 * 9.0 / cov.a)));
        CHECK(gaussian_weight(cam, s, pixel_centre(16, 14)) == doctest::Approx(std::exp(-0.5 * 4.0 / cov.c)));
    }
    SUBCASE("below 1/255 the pixel shows the background; the clamp caps alpha at 0.99") {
        const Rgb far = one_splat_value(cam, s, pixel_centre(0, 0), bg);
        CHECK(raw_alpha(cam, s, pixel_centre(0, 0)) < kAlphaMin);
        for (int ch = 0; ch < 3; ++ch) CHECK(far[ch] == bg[ch]);
        Splat opaque = s;
        opaque.opacity_logit = 12.0;
        const Rgb v = one_splat_value(cam, opaque, pixel_centre(16, 16), bg);
        for (int ch = 0; ch < 3; ++ch) CHECK(v[ch] == doctest::Approx(c[ch] * 0.99 + bg[ch] * 0.01));
    }
    SUBCASE("the threshold is exact: just above and below 1/255") {
        // At the mean G = 1, so alpha = sigmoid(o); pick o for alpha = 1/255 (1 +- 1e-9).
        for (double f : {1.0 + 1e-9, 1.0 - 1e-9}) {
            const double a = kAlphaMin * f;
            Splat t = s;
            t.opacity_logit = std::log(a / (1 - a));
            const Rgb v = one_splat_value(cam, t, pixel_centre(16, 16), bg);
            CHECK((v[0] != bg[0]) == (f > 1.0));
        }
    }
    SUBCASE("render agrees with one_splat_value, and is symmetric") {
        const Image img = render(cam, {s}, bg);
        for (int j = 0; j < cam.height; ++j)
            for (int i = 0; i < cam.width; ++i) {
                const Rgb v = one_splat_value(cam, s, pixel_centre(i, j), bg);
                for (int ch = 0; ch < 3; ++ch) CHECK(img.at(i, j)[ch] == doctest::Approx(v[ch]).epsilon(1e-14));
                // mirror symmetry in both axes for an axis-aligned splat on the axis
                CHECK(img.at(i, j)[0] == doctest::Approx(img.at(cam.width - 1 - i, j)[0]).epsilon(1e-14));
                CHECK(img.at(i, j)[0] == doctest::Approx(img.at(i, cam.height - 1 - j)[0]).epsilon(1e-14));
            }
        // a 90-degree rotation transposes the image (square camera)
        Splat r = s;
        r.theta = kPi / 2;
        const Image rot = render(cam, {r}, bg);
        for (int j = 0; j < cam.height; ++j)
            for (int i = 0; i < cam.width; ++i)
                CHECK(rot.at(i, j)[1] == doctest::Approx(img.at(j, i)[1]).epsilon(1e-12));
    }
}

TEST_CASE("footprint_covers_image") {
    const Camera cam{20.0, 15, 15};
    CHECK(footprint_covers_image(cam, grad_splat()));
    CHECK_FALSE(footprint_covers_image(Camera{50.0, 33, 33}, on_axis(4.0, 0.05, 0.05, 0.05)));
    // The square must reach the image's edges, not just the outer pixel centres: on the axis of a
    // 17-px image (mu = 8.5), radius 8 reaches the centres 0.5 and 16.5 but not the edges 0 and 17.
    const Camera small{20.0, 17, 17};
    Splat edge = on_axis(4.0, 0.485, 0.485, 0.485);  // 3 sqrt((20 * 0.485 / 4)^2 + 0.3 + sqrt(0.1)) = 7.65
    REQUIRE(radius_px(cov2d(small, edge)) == 8);
    CHECK_FALSE(footprint_covers_image(small, edge));
    edge.log_scale = {std::log(1.8), std::log(1.8), std::log(1.8)};
    REQUIRE(radius_px(cov2d(small, edge)) >= 9);
    CHECK(footprint_covers_image(small, edge));
}

TEST_CASE("a camera rotated about z: world_to_camera_about_z matches Eq. 5 with W = Rz(phi)") {
    const Camera cam{50.0, 33, 33};
    const std::array<double, 3> scale{0.3, 0.08, 0.15};
    for (double phi : {0.4, -1.1, 2.5})
        for (double theta : {0.0, 0.7}) {
            const Splat world{0.3, -0.2, 4.0, {std::log(scale[0]), std::log(scale[1]), std::log(scale[2])},
                              theta, 0.0, {0.0, 0.0, 0.0}};
            const Mat3 W = rotation(quat_about_z(phi));  // COLMAP's world-to-camera R, qvec = quat_about_z(phi)
            const Splat s = world_to_camera_about_z(world, phi);
            // the camera-space mean is W p
            CHECK(s.x == doctest::Approx(W[0][0] * world.x + W[0][1] * world.y));
            CHECK(s.y == doctest::Approx(W[1][0] * world.x + W[1][1] * world.y));
            CHECK(s.z == world.z);
            const Sym2 want = cov2d_direct(cam, W, {world.x, world.y, world.z}, scale, quat_about_z(theta));
            check_sym2(cov2d(cam, s), want);
            // W^T on the covariance (the old code's bug) rotates it by theta - phi instead: the test
            // can tell the two apart.
            Splat wrong = s;
            wrong.theta = theta - phi;
            CHECK(std::abs(cov2d(cam, wrong).b - want.b) > 1e-3);
        }
}

TEST_CASE("two splats on the axis composite in depth order") {
    const Camera cam{50.0, 33, 33};
    const Rgb bg{0.0, 0.1, 0.2};
    Splat front = on_axis(3.0, 0.2, 0.2, 0.2, 0.0, 0.4);
    front.dc = {1.5, -1.0, 0.0};
    Splat back = on_axis(6.0, 0.5, 0.3, 0.2, 0.0, 1.2);
    back.dc = {-1.0, 1.5, 0.5};
    const Image ab = render(cam, {front, back}, bg);
    const Image ba = render(cam, {back, front}, bg);
    for (int j = 0; j < cam.height; ++j)
        for (int i = 0; i < cam.width; ++i) {
            const Vec2 p = pixel_centre(i, j);
            const double a1 = std::min(raw_alpha(cam, front, p), kAlphaMax);
            const double a2 = std::min(raw_alpha(cam, back, p), kAlphaMax);
            // the input order doesn't matter: the depth does
            for (int ch = 0; ch < 3; ++ch) CHECK(ab.at(i, j)[ch] == ba.at(i, j)[ch]);
            if (a1 >= kAlphaMin && a2 >= kAlphaMin) {
                const Rgb want = two_splat_value(a1, colour(front), a2, colour(back), bg);
                for (int ch = 0; ch < 3; ++ch) CHECK(ab.at(i, j)[ch] == doctest::Approx(want[ch]).epsilon(1e-14));
                CHECK(ab.blended[static_cast<std::size_t>(j) * cam.width + i] == 2);
            }
        }
}

TEST_CASE("two-splat closed form: limits") {
    const Rgb c1{1.0, 0.0, 0.5}, c2{0.0, 1.0, 0.25}, bg{0.2, 0.2, 0.2};
    // equal colours: c (1 - (1 - a1)(1 - a2)) + bg (1 - a1)(1 - a2)
    const Rgb same = two_splat_value(0.3, c1, 0.6, c1, bg);
    for (int ch = 0; ch < 3; ++ch) CHECK(same[ch] == doctest::Approx(c1[ch] * (1 - 0.7 * 0.4) + bg[ch] * 0.7 * 0.4));
    // a transparent back splat leaves the one-splat value; an opaque-ish front one hides the back
    const Rgb one = two_splat_value(0.3, c1, 0.0, c2, bg);
    for (int ch = 0; ch < 3; ++ch) CHECK(one[ch] == doctest::Approx(c1[ch] * 0.3 + bg[ch] * 0.7));
    const Rgb hidden = two_splat_value(0.99, c1, 0.5, c2, bg);
    for (int ch = 0; ch < 3; ++ch) CHECK(std::abs(hidden[ch] - c1[ch]) < 0.02);
    // composite() agrees
    const PixelValue v = composite({{2.0, 0.6, c2}, {1.0, 0.3, c1}}, bg);
    const Rgb want = two_splat_value(0.3, c1, 0.6, c2, bg);
    for (int ch = 0; ch < 3; ++ch) CHECK(v.colour[ch] == doctest::Approx(want[ch]));
    CHECK(v.transmittance == doctest::Approx(0.7 * 0.4));
}

TEST_CASE("Appendix C thresholds in composite()") {
    const Rgb red{1.0, 0.0, 0.0}, green{0.0, 1.0, 0.0}, blue{0.0, 0.0, 1.0}, bg{0.0, 0.0, 0.0};
    SUBCASE("transmittance stop: the fragment that would take T below 1e-4 isn't blended, nor later ones") {
        // T: 1 -> 0.01 -> 2e-4 (blended) -> 2e-5 (< 1e-4: stop)
        const PixelValue v = composite({{1.0, 0.99, red}, {2.0, 0.98, green}, {3.0, 0.9, blue}, {4.0, 0.5, red}}, bg);
        CHECK(v.blended == 2);
        CHECK(v.transmittance == doctest::Approx(2e-4));
        CHECK(v.colour[0] == doctest::Approx(0.99));
        CHECK(v.colour[1] == doctest::Approx(0.98 * 0.01));
        CHECK(v.colour[2] == 0.0);
    }
    SUBCASE("alpha is clamped to 0.99 before the tests") {
        const PixelValue v = composite({{1.0, 5.0, red}}, Rgb{0.0, 1.0, 0.0});
        CHECK(v.colour[0] == doctest::Approx(0.99));
        CHECK(v.colour[1] == doctest::Approx(0.01));
    }
    SUBCASE("a skipped fragment doesn't stop the blend") {
        const PixelValue v = composite({{1.0, 0.001, red}, {2.0, 0.5, green}}, bg);
        CHECK(v.blended == 1);
        CHECK(v.colour[0] == 0.0);
        CHECK(v.colour[1] == doctest::Approx(0.5));
    }
    SUBCASE("no fragments: the background") {
        const PixelValue v = composite({}, Rgb{0.3, 0.4, 0.5});
        CHECK(v.colour[2] == 0.5);
        CHECK(v.transmittance == 1.0);
    }
}

TEST_CASE("single-splat L2 gradients match central differences") {
    const Splat s0 = grad_splat();
    REQUIRE(footprint_covers_image(kGradCam, s0));
    const L2Gradients g = single_splat_l2(kGradCam, s0, kTarget, kBg);
    // Every pixel is blended, so the loss is smooth: check the precondition the test relies on.
    for (int j = 0; j < kGradCam.height; ++j)
        for (int i = 0; i < kGradCam.width; ++i) REQUIRE(raw_alpha(kGradCam, s0, pixel_centre(i, j)) > kAlphaMin);

    const double h = 1e-5, eps = 1e-6;
    auto fd = [&](const std::function<void(Splat&, double)>& perturb) {
        return central_difference([&](double d) { Splat s = s0; perturb(s, d); return loss_of(s); }, h);
    };
    CHECK(g.opacity_logit == doctest::Approx(fd([](Splat& s, double d) { s.opacity_logit += d; })).epsilon(eps));
    CHECK(g.x == doctest::Approx(fd([](Splat& s, double d) { s.x += d; })).epsilon(eps));
    CHECK(g.y == doctest::Approx(fd([](Splat& s, double d) { s.y += d; })).epsilon(eps));
    for (int n = 0; n < 3; ++n) {
        CHECK(g.log_scale[n] == doctest::Approx(fd([n](Splat& s, double d) { s.log_scale[n] += d; })).epsilon(eps));
        CHECK(g.dc[n] == doctest::Approx(fd([n](Splat& s, double d) { s.dc[n] += d; })).epsilon(eps));
    }
    CHECK(g.log_scale_isotropic ==
          doctest::Approx(fd([](Splat& s, double d) { for (double& l : s.log_scale) l += d; })).epsilon(eps));
    // dL/dc = dL/ddc / C0 where the colour isn't clamped
    for (int ch = 0; ch < 3; ++ch) CHECK(g.colour[ch] * kShC0 == doctest::Approx(g.dc[ch]));
}

TEST_CASE("the gradients' loss is render()'s") {
    const Splat s0 = grad_splat();
    const Image img = render(kGradCam, {s0}, kBg);
    double loss = 0.0;
    for (const Rgb& c : img.colour)
        for (int ch = 0; ch < 3; ++ch) loss += (c[ch] - kTarget[ch]) * (c[ch] - kTarget[ch]);
    CHECK(single_splat_l2(kGradCam, s0, kTarget, kBg).loss == doctest::Approx(loss).epsilon(1e-13));
    CHECK(projected_l2(kGradCam, project(kGradCam, s0), kTarget, kBg).loss == doctest::Approx(loss).epsilon(1e-13));
}

TEST_CASE("projected-splat (2D) gradients match central differences, off the axis") {
    const Splat s0 = grad_splat();
    const Projected p0 = project(kGradCam, s0);
    const L2Gradients2d g = projected_l2(kGradCam, p0, kTarget, kBg);
    const double h = 1e-5, eps = 1e-6;
    auto fd = [&](const std::function<void(Projected&, double)>& perturb) {
        return central_difference(
            [&](double d) { Projected p = p0; perturb(p, d); return projected_l2(kGradCam, p, kTarget, kBg).loss; }, h);
    };
    CHECK(g.mean.x == doctest::Approx(fd([](Projected& p, double d) { p.mean.x += d; })).epsilon(eps));
    CHECK(g.mean.y == doctest::Approx(fd([](Projected& p, double d) { p.mean.y += d; })).epsilon(eps));
    // b is the shared off-diagonal value: perturbing it moves both entries
    CHECK(g.cov.a == doctest::Approx(fd([](Projected& p, double d) { p.cov.a += d; })).epsilon(eps));
    CHECK(g.cov.b == doctest::Approx(fd([](Projected& p, double d) { p.cov.b += d; })).epsilon(eps));
    CHECK(g.cov.c == doctest::Approx(fd([](Projected& p, double d) { p.cov.c += d; })).epsilon(eps));
    CHECK(g.opacity == doctest::Approx(fd([](Projected& p, double d) { p.opacity += d; })).epsilon(eps));
    for (int ch = 0; ch < 3; ++ch)
        CHECK(g.colour[ch] == doctest::Approx(fd([ch](Projected& p, double d) { p.colour[ch] += d; })).epsilon(eps));

    // The conic: perturb A and invert it back to Sigma'.
    const Sym2 A0 = inverse(p0.cov);
    auto fd_conic = [&](int entry) {
        return central_difference(
            [&](double d) {
                Sym2 A = A0;
                (entry == 0 ? A.a : entry == 1 ? A.b : A.c) += d;
                Projected p = p0;
                p.cov = inverse(A);
                return projected_l2(kGradCam, p, kTarget, kBg).loss;
            },
            h);
    };
    CHECK(g.conic.a == doctest::Approx(fd_conic(0)).epsilon(eps));
    CHECK(g.conic.b == doctest::Approx(fd_conic(1)).epsilon(eps));
    CHECK(g.conic.c == doctest::Approx(fd_conic(2)).epsilon(eps));
    // As full matrices (off-diagonal entries half the shared-value gradient): dL/dSigma' = -A dL/dA A.
    const double Gc[2][2] = {{g.conic.a, 0.5 * g.conic.b}, {0.5 * g.conic.b, g.conic.c}};
    const double Am[2][2] = {{A0.a, A0.b}, {A0.b, A0.c}};
    double want[2][2] = {};
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 2; ++c)
            for (int m = 0; m < 2; ++m)
                for (int l = 0; l < 2; ++l) want[r][c] -= Am[r][m] * Gc[m][l] * Am[l][c];
    CHECK(g.cov.a == doctest::Approx(want[0][0]));
    CHECK(g.cov.b == doctest::Approx(2.0 * want[0][1]));
    CHECK(g.cov.c == doctest::Approx(want[1][1]));

    // single_splat_l2 carries these, and off the axis its x gradient isn't the mean term alone.
    const L2Gradients g3 = single_splat_l2(kGradCam, s0, kTarget, kBg);
    CHECK(g3.screen.mean.x == g.mean.x);
    CHECK(g3.screen.cov.b == g.cov.b);
    const double mean_only = g.mean.x * kGradCam.f / s0.z;
    CHECK(std::abs(g3.x - mean_only) > 0.01 * std::abs(g3.x));
}

TEST_CASE("single-splat L2 gradients: symmetry and limit cases") {
    const Camera cam = kGradCam;
    Splat s = grad_splat();
    s.x = s.y = 0.0;
    s.theta = 0.0;

    SUBCASE("centred on the axis: no position gradient, no z-scale gradient") {
        const L2Gradients g = single_splat_l2(cam, s, kTarget, kBg);
        CHECK(g.x == doctest::Approx(0.0).scale(1.0).epsilon(1e-10));
        CHECK(g.y == doctest::Approx(0.0).scale(1.0).epsilon(1e-10));
        CHECK(g.log_scale[2] == doctest::Approx(0.0).scale(1.0).epsilon(1e-10));
        CHECK(g.log_scale[0] != 0.0);
    }
    SUBCASE("mirroring the splat in x flips the x gradient only") {
        s.x = 0.2;
        const L2Gradients a = single_splat_l2(cam, s, kTarget, kBg);
        s.x = -0.2;
        const L2Gradients b = single_splat_l2(cam, s, kTarget, kBg);
        CHECK(a.x == doctest::Approx(-b.x));
        CHECK(a.y == doctest::Approx(b.y).scale(1.0));
        CHECK(a.opacity_logit == doctest::Approx(b.opacity_logit));
        CHECK(a.log_scale[0] == doctest::Approx(b.log_scale[0]));
    }
    SUBCASE("a splat of the background colour over a target equal to it: only the colour moves L") {
        Splat bgs = s;
        for (int ch = 0; ch < 3; ++ch) bgs.dc[ch] = (kBg[ch] - 0.5) / kShC0;
        const L2Gradients g = single_splat_l2(cam, bgs, kBg, kBg);
        CHECK(g.loss == doctest::Approx(0.0).scale(1.0));
        CHECK(g.opacity_logit == doctest::Approx(0.0).scale(1.0));
        CHECK(g.x == doctest::Approx(0.0).scale(1.0));
        CHECK(g.log_scale[0] == doctest::Approx(0.0).scale(1.0));
        CHECK(g.colour[0] == doctest::Approx(0.0).scale(1.0));
    }
    SUBCASE("a dark target pulls opacity down and a bright one up") {
        Splat white = s;
        white.dc = {2.0, 2.0, 2.0};
        CHECK(single_splat_l2(cam, white, Rgb{0, 0, 0}, Rgb{0, 0, 0}).opacity_logit > 0.0);
        CHECK(single_splat_l2(cam, white, Rgb{2, 2, 2}, Rgb{0, 0, 0}).opacity_logit < 0.0);
    }
    SUBCASE("a clamped pixel is outside the oracle's domain") {
        s.opacity_logit = 10.0;
        CHECK_THROWS_AS(single_splat_l2(cam, s, kTarget, kBg), std::invalid_argument);
    }
}
