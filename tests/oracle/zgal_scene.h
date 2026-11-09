// The closed-form oracle (closed_form.h) for a ZGAL program's S1 scene: the oracle's camera and
// splats from lib/splat_math.zgal's Camera and Splat (any generated namespace's, hence the
// templates), the borderline rule for a pixel whose threshold decisions sit within rounding of their
// bounds, and the FNV-1a hash the checks print. render_test's, grad_test's and splat_view's checks
// share it.
#pragma once

#include "closed_form.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace oracle {

// An S1 camera: at the origin, rotated by phi about z, with fx = fy
struct SceneCamera {
    Camera camera;
    double phi;
};

// The oracle's camera, or nothing if the camera isn't S1's. The principal point is the camera's,
// so an off-centre one (S1's off-centre) is the oracle's too.
template <class C>
std::optional<SceneCamera> sceneCamera(const C& cam) {
    const auto& m = cam.worldToCamera;
    if (cam.intrinsics.x != cam.intrinsics.y || m[3][0] != 0.0f || m[3][1] != 0.0f || m[3][2] != 0.0f ||
        m[3][3] != 1.0f || m[2][2] != 1.0f) {
        return std::nullopt;
    }
    const double width = cam.size.x, height = cam.size.y;
    return SceneCamera{{cam.intrinsics.x, int(cam.size.x), int(cam.size.y), cam.intrinsics.z - 0.5 * width,
                        cam.intrinsics.w - 0.5 * height},
                       std::atan2(double(m[0][1]), double(m[0][0]))};
}

// The oracle's splat for an S1 splat (rotated about z only) seen by a camera rotated by phi about z,
// or nothing if the splat is rotated about another axis
template <class S>
std::optional<Splat> sceneSplat(const S& s, double phi) {
    if (s.rotation.y != 0.0f || s.rotation.z != 0.0f) return std::nullopt;
    Splat w{};
    w.x = s.position.x;
    w.y = s.position.y;
    w.z = s.position.z;
    w.log_scale = {s.logScale.x, s.logScale.y, s.logScale.z};
    // (w, x, y, z) in (.x, .y, .z, .w): quat_about_z(theta) = (cos(theta/2), 0, 0, sin(theta/2))
    w.theta = 2.0 * std::atan2(double(s.rotation.w), double(s.rotation.x));
    w.opacity_logit = s.opacity;
    w.dc = {s.sh[0], s.sh[1], s.sh[2]};
    return world_to_camera_about_z(w, phi);
}

// Whether value is within a relative margin of bound
inline bool near(double value, double bound, double margin) { return std::abs(value - bound) <= margin * bound; }

// A threshold decision within rounding of its bound, which the GPU may take the other way.
// Flipping one changes T by at most `flip` and the colour by at most 2 flip M, M the largest
// colour channel among the pixel's splats and the background: for alpha (or the exponent's sign),
// flip = alpha T, the splat's own term plus the (1 - alpha) scaling of everything after it; for
// the transmittance stop, flip = T, since everything after it (at most T M) and the background's
// T bg change.
struct Borderline {
    bool any = false;
    double flip = 0.0;  // the sum over the pixel's borderline decisions
    double maxColour = 0.0;

    void add(double f) {
        any = true;
        flip += f;
    }
    void see(const Rgb& c) {
        for (double v : c) maxColour = std::max(maxColour, v);
    }
    double colourTolerance(double base) const { return base + 2.0 * flip * maxColour; }
    double tTolerance(double base) const { return base + flip; }
};

// The borderline decisions of composite(fragments, bg): an alpha or a running T within a relative
// margin of its threshold (the oracle's Gaussian has no exponent test)
inline Borderline borderline(std::vector<Fragment> fragments, const Rgb& bg, double margin) {
    Borderline border;
    border.see(bg);
    std::stable_sort(fragments.begin(), fragments.end(),
                     [](const Fragment& l, const Fragment& r) { return l.depth < r.depth; });
    double t = 1.0;
    for (const Fragment& fr : fragments) {
        border.see(fr.colour);
        const double alpha = std::min(fr.alpha, kAlphaMax);
        if (near(alpha, kAlphaMin, margin)) border.add(alpha * t);
        if (alpha < kAlphaMin) continue;
        const double next = t * (1.0 - alpha);
        if (near(next, kTransmittanceMin, margin)) border.add(t);
        if (next < kTransmittanceMin) break;
        t = next;
    }
    return border;
}

inline constexpr uint64_t kFnvBasis = 0xcbf29ce484222325ull;

inline uint64_t fnv1a(uint64_t h, const void* bytes, size_t size) {
    const auto* p = static_cast<const unsigned char*>(bytes);
    for (size_t i = 0; i < size; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

}  // namespace oracle
