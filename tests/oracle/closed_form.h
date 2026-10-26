// Closed-form test oracles for the splatting kernels: single-splat renders, a two-splat composite and
// single-splat gradients, hand-derived for one canonical setup and independent of the kernels (they
// share no code with lib/ and don't evaluate the general EWA projection). Header-only, in double.
//
// Spec: Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023);
// "Eq. n" below is the paper's equation n. Values the paper doesn't give (the 0.3 low-pass term, the
// SH +0.5 offset, the ceil on the 3-sigma radius and the 0.1 floor in its eigenvalue) follow its
// reference code (graphdeco-inria/gaussian-splatting, diff-gaussian-rasterization's forward.cu and
// backward.cu) and are marked "(reference code)"; docs/paper-conformance.md lists them.
//
// ---------------------------------------------------------------------------------------------------
// Canonical setup
// ---------------------------------------------------------------------------------------------------
// - Camera: COLMAP's conventions (docs/architecture.md) with R = I and t = 0: the camera sits at the
//   origin looking down +z, +x right, +y down. Pinhole with one focal length f (fx = fy = f) and the
//   principal point at the image centre (cx, cy) = (width/2, height/2). Pixel (i, j) (column i,
//   row j) has its centre at (i + 0.5, j + 0.5). The reference rasterizer's convention is the same:
//   it puts pixel centres at integers but maps to pixels with ndc2Pix = ((ndc + 1) W - 1) / 2, a
//   -0.5 shift. The oracles are meant for square images; nothing below needs width == height
//   except the transpose symmetry the tests check.
// - A camera rotated about its z axis maps to this setup with world_to_camera_about_z().
// - Splat: mean (x, y, z) with z > 0, log-scales (lx, ly, lz) (scale s = exp(l), Sec. 5's exponential
//   activation), a rotation by theta about the camera's z axis (the quaternion is quat_about_z(theta)),
//   an opacity logit o (alpha = sigmoid(o), Sec. 5), and SH degree-0 (DC) coefficients per channel.
//   Higher SH bands are zero: the closed forms here cover only rotations about z and DC colour.
// - Image value per channel: front-to-back alpha blending over a background colour bg,
//   I = sum_i c_i alpha_i T_i + T_final bg, T_i = prod_{j<i} (1 - alpha_j) (Eq. 3, plus the
//   reference code's background term), with Appendix C's thresholds.
// - Loss for the gradients: L = sum over pixels p and channels ch of (I_p,ch - t_ch)^2, an L2 loss
//   against a constant target colour t. That is smooth, unlike the training loss (Eq. 7), so the
//   finite-difference tests can use it too.
//
// What the oracles leave out, on purpose:
// - The tile cutoff. A kernel blends a splat only into the 16x16 tiles its 3-sigma square
//   (mean +- radius_px) touches; render() blends every splat into every pixel. A full-image
//   comparison must use a splat whose square covers the whole image (footprint_covers_image()), or
//   mask out the pixels whose tile the splat doesn't touch.
// - The near plane and frustum guard band (Sec. 6), and the clamp of x/z and y/z in the Jacobian
//   (reference code): every setup here is well inside the view, so neither applies.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace oracle {

using Rgb = std::array<double, 3>;

// ---------------------------------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------------------------------
// The low-pass term added to the 2D covariance's diagonal (reference code; EWA's screen-space
// filter, Zwicker et al. 2001), in px^2: every splat covers at least about one pixel.
inline constexpr double kLowPass = 0.3;
// Appendix C: blending skips a splat whose alpha is below 1/255, clamps alpha to at most 0.99, and
// stops before the transmittance T would drop below 1e-4 (accumulated opacity above 0.9999).
inline constexpr double kAlphaMin = 1.0 / 255.0;
inline constexpr double kAlphaMax = 0.99;
inline constexpr double kTransmittanceMin = 1e-4;
// The degree-0 real spherical harmonic, Y_0^0 = 1 / (2 sqrt(pi)).
inline constexpr double kShC0 = 0.28209479177387814;

inline void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("oracle: ") + what);
}

// ---------------------------------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------------------------------
struct Camera {
    double f;            // focal length in pixels, fx = fy
    int width, height;   // in pixels
    double cx() const { return 0.5 * width; }
    double cy() const { return 0.5 * height; }
};

struct Splat {
    double x, y, z;                  // camera-space mean (z > 0)
    std::array<double, 3> log_scale; // (lx, ly, lz); s = exp(l)
    double theta;                    // rotation about the camera's z axis, radians
    double opacity_logit;            // alpha = sigmoid(opacity_logit)
    Rgb dc;                          // SH degree-0 coefficient per channel
};

struct Vec2 { double x, y; };

// A symmetric 2x2 matrix [[a, b], [b, c]].
struct Sym2 { double a, b, c; };

// A quaternion, real part first (COLMAP's and the representation's order).
struct Quat { double w, x, y, z; };

// ---------------------------------------------------------------------------------------------------
// Activations and colour
// ---------------------------------------------------------------------------------------------------
inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// SH degree 0 to colour: c = max(C0 * dc + 0.5, 0) (reference code's computeColorFromSH: the +0.5
// centres a zero coefficient on mid-grey; colours are clamped from below only).
// d c / d dc = C0 where C0 * dc + 0.5 > 0, else 0.
inline double sh_dc_to_colour(double dc) { return std::max(kShC0 * dc + 0.5, 0.0); }
inline double sh_dc_to_colour_derivative(double dc) { return kShC0 * dc + 0.5 > 0.0 ? kShC0 : 0.0; }
inline Rgb colour(const Splat& s) {
    return {sh_dc_to_colour(s.dc[0]), sh_dc_to_colour(s.dc[1]), sh_dc_to_colour(s.dc[2])};
}

// The unit quaternion of a rotation by theta about z: (cos(theta/2), 0, 0, sin(theta/2)). Eq. 10
// with q_i = q_j = 0 gives R = [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], so kernels fed this
// quaternion see the rotation the closed forms below assume.
inline Quat quat_about_z(double theta) { return {std::cos(0.5 * theta), 0.0, 0.0, std::sin(0.5 * theta)}; }

// ---------------------------------------------------------------------------------------------------
// Projection
// ---------------------------------------------------------------------------------------------------
// 2D mean (pinhole): mu = (f x / z + cx, f y / z + cy).
inline Vec2 mean2d(const Camera& cam, const Splat& s) {
    require(s.z > 0.0, "the splat must be in front of the camera (z > 0)");
    return {cam.f * s.x / s.z + cam.cx(), cam.f * s.y / s.z + cam.cy()};
}

// The camera-space 3D covariance V = W Sigma W^T with W = I, Sigma = R S S^T R^T (Eq. 6) and R the
// rotation by theta about z, S = diag(sx, sy, sz):
//   V00 = sx^2 cos^2 + sy^2 sin^2,  V01 = (sx^2 - sy^2) sin cos,  V11 = sx^2 sin^2 + sy^2 cos^2,
//   V22 = sz^2,  V02 = V12 = 0 (a rotation about z keeps z an eigenvector).
struct Cov3dAboutZ { double v00, v01, v11, v22; };
inline Cov3dAboutZ cov3d(const Splat& s) {
    const double sx2 = std::exp(2.0 * s.log_scale[0]);
    const double sy2 = std::exp(2.0 * s.log_scale[1]);
    const double sz2 = std::exp(2.0 * s.log_scale[2]);
    const double co = std::cos(s.theta), si = std::sin(s.theta);
    return {sx2 * co * co + sy2 * si * si, (sx2 - sy2) * si * co, sx2 * si * si + sy2 * co * co, sz2};
}

// The 2D covariance Sigma' (Eq. 5) plus the low-pass term. The Jacobian of the pinhole projection
// (x, y, z) -> (f x / z, f y / z) is J = (f/z) [[1, 0, -u], [0, 1, -v]] with u = x/z, v = y/z, and
// W = I. With V02 = V12 = 0, Sigma' = J V J^T is, with k = (f/z)^2:
//   a = k (V00 + u^2 V22) + 0.3,  b = k (V01 + u v V22),  c = k (V11 + v^2 V22) + 0.3.
// Cases the tests pin down:
// - isotropic (s on all axes), on the axis (u = v = 0): a = c = (f s / z)^2 + 0.3, b = 0, for any theta;
// - anisotropic, axis-aligned (theta = 0), on the axis: a = (f sx / z)^2 + 0.3, c = (f sy / z)^2 + 0.3;
// - the same rotated by 90 degrees about z: a and c swap;
// - on the axis, sz has no effect (the third column of J is zero there).
inline Sym2 cov2d(const Camera& cam, const Splat& s) {
    require(s.z > 0.0, "the splat must be in front of the camera (z > 0)");
    const Cov3dAboutZ v = cov3d(s);
    const double k = (cam.f / s.z) * (cam.f / s.z);
    const double u = s.x / s.z, w = s.y / s.z;
    return {k * (v.v00 + u * u * v.v22) + kLowPass, k * (v.v01 + u * w * v.v22),
            k * (v.v11 + w * w * v.v22) + kLowPass};
}

inline double det(const Sym2& m) { return m.a * m.c - m.b * m.b; }

// The conic: the inverse of Sigma', [[c, -b], [-b, a]] / det.
inline Sym2 inverse(const Sym2& m) {
    const double d = det(m);
    require(d > 0.0, "the 2D covariance must be positive definite");
    return {m.c / d, -m.b / d, m.a / d};
}

// The larger eigenvalue, exactly: mid + sqrt(mid^2 - det) with mid = (a + c) / 2 (the old code's
// precedence bug wrote 0.5 * a + c). For b = 0 it is max(a, c).
inline double max_eigenvalue(const Sym2& m) {
    const double mid = 0.5 * (m.a + m.c);
    return mid + std::sqrt(std::max(mid * mid - det(m), 0.0));
}

// The eigenvalue the radius uses: mid + sqrt(max(mid^2 - det, 0.1)) (reference code: forward.cu's
// lambda1). The floor only matters for a near-isotropic splat (mid^2 - det below 0.1), where it adds
// up to sqrt(0.1) px^2, so such a splat's radius can be a pixel larger than the exact eigenvalue's.
inline constexpr double kRadiusDiscriminantMin = 0.1;
inline double radius_eigenvalue(const Sym2& m) {
    const double mid = 0.5 * (m.a + m.c);
    return mid + std::sqrt(std::max(mid * mid - det(m), kRadiusDiscriminantMin));
}

// The splat's screen radius in pixels: ceil(3 sqrt(radius_eigenvalue)) (Sec. 6's 99% confidence
// interval along the major axis; the ceil and the floor from the reference code). Isotropic on the
// axis: ceil(3 sqrt((f s / z)^2 + 0.3 + sqrt(0.1))).
inline int radius_px(const Sym2& cov) {
    return static_cast<int>(std::ceil(3.0 * std::sqrt(radius_eigenvalue(cov))));
}

// True when the square mean +- radius_px covers the whole image, edges included ([0, width] x
// [0, height]), so a kernel touches every tile under any tile rule and its tile cutoff can't differ
// from render() anywhere.
inline bool footprint_covers_image(const Camera& cam, const Splat& s) {
    const Vec2 mu = mean2d(cam, s);
    const double r = radius_px(cov2d(cam, s));
    return mu.x - r <= 0.0 && mu.x + r >= cam.width && mu.y - r <= 0.0 && mu.y + r >= cam.height;
}

// A camera rotated by phi about its z axis (COLMAP's world-to-camera R = Rz(phi), qvec =
// quat_about_z(phi), t = 0) sees a world-space splat (mean p, rotation theta about z) as the
// canonical splat with mean Rz(phi) p and rotation theta + phi: V = W Sigma W^T with W = Rz(phi)
// (Eq. 5). The old code's W^T would give theta - phi, which differs for an anisotropic splat.
// Kernel tests feed the world splat and the rotated camera, and compare with the oracle on the
// returned splat.
inline Splat world_to_camera_about_z(const Splat& world, double phi) {
    const double co = std::cos(phi), si = std::sin(phi);
    Splat s = world;
    s.x = co * world.x - si * world.y;
    s.y = si * world.x + co * world.y;
    s.theta = world.theta + phi;
    return s;
}

// ---------------------------------------------------------------------------------------------------
// Per-pixel value
// ---------------------------------------------------------------------------------------------------
inline Vec2 pixel_centre(int i, int j) { return {i + 0.5, j + 0.5}; }

// The Gaussian's weight at a pixel centre p (Eq. 4 on the projected splat):
// G = exp(-1/2 d^T Sigma'^-1 d), d = p - mu. G = 1 at the mean.
inline double gaussian_weight(const Camera& cam, const Splat& s, Vec2 p) {
    const Vec2 mu = mean2d(cam, s);
    const Sym2 q = inverse(cov2d(cam, s));
    const double dx = p.x - mu.x, dy = p.y - mu.y;
    return std::exp(-0.5 * (q.a * dx * dx + 2.0 * q.b * dx * dy + q.c * dy * dy));
}

// The splat's alpha at p before Appendix C's clamp: sigmoid(o) * G.
inline double raw_alpha(const Camera& cam, const Splat& s, Vec2 p) {
    return sigmoid(s.opacity_logit) * gaussian_weight(cam, s, p);
}

// One splat's contribution to one pixel, for compositing.
struct Fragment {
    double depth;  // view-space z: the sort key's depth (Sec. 6, Appendix C)
    double alpha;  // raw alpha, before the clamp
    Rgb colour;
};

struct PixelValue {
    Rgb colour;
    double transmittance;  // T after the last blended fragment
    int blended;           // how many fragments were blended (passed the thresholds)
};

// Front-to-back blending (Eq. 3) of fragments in depth order (nearest first; equal depths keep
// their input order), over bg, with Appendix C's thresholds:
//   alpha = min(alpha, 0.99); skip the fragment if alpha < 1/255;
//   T' = T (1 - alpha); if T' < 1e-4, stop: this fragment and every later one are not blended;
//   C += c alpha T; T = T'.
// The value is C + T bg.
inline PixelValue composite(std::vector<Fragment> fragments, const Rgb& bg) {
    std::stable_sort(fragments.begin(), fragments.end(),
                     [](const Fragment& l, const Fragment& r) { return l.depth < r.depth; });
    Rgb c{0.0, 0.0, 0.0};
    double t = 1.0;
    int blended = 0;
    for (const Fragment& fr : fragments) {
        const double alpha = std::min(fr.alpha, kAlphaMax);
        if (alpha < kAlphaMin) continue;
        const double next_t = t * (1.0 - alpha);
        if (next_t < kTransmittanceMin) break;
        for (int ch = 0; ch < 3; ++ch) c[ch] += fr.colour[ch] * alpha * t;
        t = next_t;
        ++blended;
    }
    for (int ch = 0; ch < 3; ++ch) c[ch] += t * bg[ch];
    return {c, t, blended};
}

// One splat, closed form: with alpha = min(sigmoid(o) G, 0.99),
//   I = c alpha + bg (1 - alpha) if alpha >= 1/255, else I = bg.
// (A single splat never trips the transmittance stop: T >= 1 - 0.99 = 0.01.)
inline Rgb one_splat_value(const Camera& cam, const Splat& s, Vec2 p, const Rgb& bg) {
    const double alpha = std::min(raw_alpha(cam, s, p), kAlphaMax);
    if (alpha < kAlphaMin) return bg;
    const Rgb c = colour(s);
    return {c[0] * alpha + bg[0] * (1.0 - alpha), c[1] * alpha + bg[1] * (1.0 - alpha),
            c[2] * alpha + bg[2] * (1.0 - alpha)};
}

// Two splats, closed form, front (nearer) one first, both alphas already clamped and above 1/255:
//   I = c1 a1 + c2 a2 (1 - a1) + bg (1 - a1)(1 - a2)
// (Eq. 3 with two terms, plus the background). The transmittance stop can't trigger:
// (1 - a1)(1 - a2) >= 0.01^2 = 1e-4.
inline Rgb two_splat_value(double a1, const Rgb& c1, double a2, const Rgb& c2, const Rgb& bg) {
    Rgb out;
    for (int ch = 0; ch < 3; ++ch)
        out[ch] = c1[ch] * a1 + c2[ch] * a2 * (1.0 - a1) + bg[ch] * (1.0 - a1) * (1.0 - a2);
    return out;
}

// A rendered image, row-major: pixel (i, j) at j * width + i.
struct Image {
    int width = 0, height = 0;
    std::vector<Rgb> colour;
    std::vector<double> transmittance;
    std::vector<int> blended;
    const Rgb& at(int i, int j) const { return colour[static_cast<std::size_t>(j) * width + i]; }
};

// Every splat blended into every pixel (no tile cutoff; see the top of the file), in depth order.
inline Image render(const Camera& cam, const std::vector<Splat>& splats, const Rgb& bg) {
    Image img;
    img.width = cam.width;
    img.height = cam.height;
    const std::size_t n = static_cast<std::size_t>(cam.width) * cam.height;
    img.colour.resize(n);
    img.transmittance.resize(n);
    img.blended.resize(n);
    std::vector<Fragment> fragments(splats.size());
    for (int j = 0; j < cam.height; ++j)
        for (int i = 0; i < cam.width; ++i) {
            for (std::size_t k = 0; k < splats.size(); ++k)
                fragments[k] = {splats[k].z, raw_alpha(cam, splats[k], pixel_centre(i, j)), colour(splats[k])};
            const PixelValue v = composite(fragments, bg);
            const std::size_t idx = static_cast<std::size_t>(j) * cam.width + i;
            img.colour[idx] = v.colour;
            img.transmittance[idx] = v.transmittance;
            img.blended[idx] = v.blended;
        }
    return img;
}

// ---------------------------------------------------------------------------------------------------
// Gradients of one splat under the L2 loss
// ---------------------------------------------------------------------------------------------------
// L = sum_p sum_ch (I_p,ch - t_ch)^2 over the whole image, I from one_splat_value. Pixels with
// alpha < 1/255 show bg and contribute nothing to any gradient. Precondition: no pixel's alpha is
// clamped (sigmoid(o) G <= 0.99 everywhere), so I is smooth in every parameter at every blended
// pixel; the gradient functions throw otherwise. Kernels' backward passes may treat the clamp
// differently, so the oracle doesn't define a gradient there.
//
// Two stages, matching the kernels': projected_l2() gives the gradients with respect to the
// projected splat (2D mean, Sigma' or the conic, activated opacity, colour), which the backward
// rasterize produces; single_splat_l2() chains them to the splat's parameters, which the backward
// preprocess produces.
//
// Derivation, per blended pixel, with alpha = opacity G and the residual r_ch = I_ch - t_ch:
//   dL/dI_ch = 2 r_ch,  dI_ch/dalpha = c_ch - bg_ch,  so  g := dL/dalpha = sum_ch 2 r_ch (c_ch - bg_ch).
// - Colour: dI_ch/dc_ch = alpha, so dL/dc_ch = sum_p 2 r_ch alpha.
// - Activated opacity: dalpha/dopacity = G.
// - Through G = exp(P), P = -1/2 d^T A d, A = Sigma'^-1 (the conic), d = p - mu, dalpha = alpha dP,
//   with w = A d and dA = -A dSigma' A:
//     dP/dmu = w,  dP/dSigma'[M] = 1/2 w^T M w,  dP/dA[M] = -1/2 d^T M d  (M symmetric).
//   For a symmetric matrix [[a, b], [b, c]] the gradients are with respect to its three independent
//   entries a, b, c, so b counts both off-diagonal positions (M = [[0, 1], [1, 0]]):
//     dP/da = 1/2 w_x^2, dP/db = w_x w_y, dP/dc = 1/2 w_y^2 for Sigma';
//     dP/dA00 = -1/2 d_x^2, dP/dA01 = -d_x d_y, dP/dA11 = -1/2 d_y^2 for the conic.
//   (The reference's backward rasterize accumulates half of dL/dA01, per off-diagonal entry: only
//   its conic gradient dL_dconic2D.y is half of conic.b here. Its Sigma' gradient dL_db already
//   counts both entries and equals cov.b, so don't halve that.)
//   As full matrices, dL/dSigma' = -A (dL/dA) A. That is Appendix A's chain (Eqs. 8, 9).
struct L2Gradients2d {
    double loss;
    Vec2 mean;       // dL/dmu, per pixel of shift (the reference's dL_dmean2D is per NDC unit:
                     // 0.5 W, or 0.5 H for y, times this)
    Sym2 cov;        // dL/d(a, b, c) of Sigma' = [[a, b], [b, c]] (low-pass term included)
    Sym2 conic;      // dL/d(A00, A01, A11) of the conic A = Sigma'^-1
    double opacity;  // dL/dsigmoid(o)
    Rgb colour;
};

// The projected splat, as the rasterizer sees it.
struct Projected {
    Vec2 mean;       // px
    Sym2 cov;        // Sigma' including the low-pass term, px^2
    double opacity;  // sigmoid(o)
    Rgb colour;
};

inline Projected project(const Camera& cam, const Splat& s) {
    return {mean2d(cam, s), cov2d(cam, s), sigmoid(s.opacity_logit), colour(s)};
}

// The L2 loss of a projected splat over cam's image and its gradients (only cam's size is used).
inline L2Gradients2d projected_l2(const Camera& cam, const Projected& pr, const Rgb& target, const Rgb& bg) {
    const Sym2 A = inverse(pr.cov);
    L2Gradients2d g{};
    for (int j = 0; j < cam.height; ++j)
        for (int i = 0; i < cam.width; ++i) {
            const Vec2 p = pixel_centre(i, j);
            const double dx = p.x - pr.mean.x, dy = p.y - pr.mean.y;
            const double G = std::exp(-0.5 * (A.a * dx * dx + 2.0 * A.b * dx * dy + A.c * dy * dy));
            const double alpha = pr.opacity * G;
            require(alpha <= kAlphaMax, "a pixel's alpha is clamped at 0.99; lower the opacity");
            if (alpha < kAlphaMin) {
                for (int ch = 0; ch < 3; ++ch) g.loss += (bg[ch] - target[ch]) * (bg[ch] - target[ch]);
                continue;
            }
            double dl_dalpha = 0.0;
            for (int ch = 0; ch < 3; ++ch) {
                const double I = pr.colour[ch] * alpha + bg[ch] * (1.0 - alpha);
                const double r = I - target[ch];
                g.loss += r * r;
                g.colour[ch] += 2.0 * r * alpha;
                dl_dalpha += 2.0 * r * (pr.colour[ch] - bg[ch]);
            }
            g.opacity += dl_dalpha * G;
            const double dl_dp = dl_dalpha * alpha;
            const double wx = A.a * dx + A.b * dy, wy = A.b * dx + A.c * dy;
            g.mean.x += dl_dp * wx;
            g.mean.y += dl_dp * wy;
            g.cov.a += dl_dp * 0.5 * wx * wx;
            g.cov.b += dl_dp * wx * wy;
            g.cov.c += dl_dp * 0.5 * wy * wy;
            g.conic.a -= dl_dp * 0.5 * dx * dx;
            g.conic.b -= dl_dp * dx * dy;
            g.conic.c -= dl_dp * 0.5 * dy * dy;
        }
    return g;
}

// The splat's parameters, from the projected gradients (s = exp(l), k = (f/z)^2, u = x/z, v = y/z):
// - Opacity logit: dopacity/do = sigmoid(o) (1 - sigmoid(o)) (Sec. 5's sigmoid activation).
// - DC coefficient: dL/ddc_ch = dL/dc_ch * C0 (or 0 where the colour is clamped at 0; see
//   sh_dc_to_colour_derivative).
// - In-plane position x: dmu/dx = (f/z, 0) and, from cov2d's closed form,
//   dSigma'/dx = k V22 / z [[2u, v], [v, 0]], so dL/dx = dL/dmu_x f/z + k V22 / z (2u dL/da + v dL/db);
//   y: dmu/dy = (0, f/z), dSigma'/dy = k V22 / z [[0, u], [u, 2v]],
//   so dL/dy = dL/dmu_y f/z + k V22 / z (u dL/db + 2v dL/dc).
//   On the axis (u = v = 0) the covariance term vanishes and only the mean moves.
// - Log-scale l_n (n = x, y, z): with M = R S (Appendix A), dSigma/dl_n = 2 s_n^2 r_n r_n^T, r_n the
//   n-th column of R, so dSigma'/dl_n = 2 s_n^2 j j^T with j = J r_n, and
//   dL/dl_n = 2 s_n^2 (j_x^2 dL/da + j_x j_y dL/db + j_y^2 dL/dc),
//   with J r_x = (f/z)(cos, sin), J r_y = (f/z)(-sin, cos), J r_z = (f/z)(-u, -v).
//   The isotropic log-scale (one l for all three axes) has dL/dl = the sum of the three.
//   Limit: on the axis, dL/dl_z = 0.
struct L2Gradients {
    double loss;
    L2Gradients2d screen;  // the projected stage's gradients
    double opacity_logit;
    Rgb colour;
    Rgb dc;
    double x, y;
    std::array<double, 3> log_scale;
    double log_scale_isotropic;
};

inline L2Gradients single_splat_l2(const Camera& cam, const Splat& s, const Rgb& target, const Rgb& bg) {
    const L2Gradients2d sc = projected_l2(cam, project(cam, s), target, bg);
    const Cov3dAboutZ v = cov3d(s);
    const double fz = cam.f / s.z, k = fz * fz;
    const double u = s.x / s.z, vv = s.y / s.z;
    const double sig = sigmoid(s.opacity_logit);
    const double co = std::cos(s.theta), si = std::sin(s.theta);
    const std::array<Vec2, 3> jr = {Vec2{fz * co, fz * si}, Vec2{-fz * si, fz * co}, Vec2{-fz * u, -fz * vv}};

    L2Gradients g{};
    g.loss = sc.loss;
    g.screen = sc;
    g.opacity_logit = sc.opacity * sig * (1.0 - sig);
    g.colour = sc.colour;
    for (int ch = 0; ch < 3; ++ch) g.dc[ch] = sc.colour[ch] * sh_dc_to_colour_derivative(s.dc[ch]);
    const double cov_scale = k * v.v22 / s.z;
    g.x = sc.mean.x * fz + cov_scale * (2.0 * u * sc.cov.a + vv * sc.cov.b);
    g.y = sc.mean.y * fz + cov_scale * (u * sc.cov.b + 2.0 * vv * sc.cov.c);
    for (int n = 0; n < 3; ++n) {
        const double s2 = std::exp(2.0 * s.log_scale[n]);
        const Vec2 j = jr[n];
        g.log_scale[n] = 2.0 * s2 * (j.x * j.x * sc.cov.a + j.x * j.y * sc.cov.b + j.y * j.y * sc.cov.c);
    }
    g.log_scale_isotropic = g.log_scale[0] + g.log_scale[1] + g.log_scale[2];
    return g;
}

}  // namespace oracle
