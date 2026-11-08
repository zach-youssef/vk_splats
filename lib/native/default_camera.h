// The default camera of a stream (docs/data-format.md, "Stream files"): a deterministic camera
// framing the scene, which the encoder writes into the header, so a viewer opening a scene without a
// cameras file still sees it (and the web computes nothing).
//
// Header-only, on top of the C++ that ZGAL generates for a module importing lib/splat_math.zgal
// (its Splat and Camera). The generated code lands in the namespace of the .zgal file importing it,
// so the includer names that namespace first:
//   #include "my_program.zgal.h"
//   namespace splat_native { namespace gen = ::my_program; }
//   #include "<relative path>/lib/native/default_camera.h"
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>
#include <span>
#include <vector>

namespace splat_native {

// The image: 1280x720 with a vertical field of view of 60 degrees. tan(30 deg) = 1/sqrt(3) and
// sin(30 deg) = 1/2 are literals, so the camera doesn't depend on a libm's tan.
inline constexpr uint32_t kDefaultCameraWidth = 1280;
inline constexpr uint32_t kDefaultCameraHeight = 720;
inline constexpr float kDefaultCameraTanHalfFov = 0.57735026918962576f;
// fx = fy = (height / 2) / tan(30 deg) = 360 sqrt(3)
inline constexpr float kDefaultCameraFocal = 623.53829072479585f;
// The box's percentiles per axis: robust to a few floaters
inline constexpr float kDefaultCameraLowPercentile = 0.05f;
inline constexpr float kDefaultCameraHighPercentile = 0.95f;
// The distance from the box's centre, in radii (half the box's diagonal). r / sin(30 deg) puts the
// box's bounding sphere exactly in the vertical field of view; the margin makes it 10% further.
inline constexpr float kDefaultCameraDistance = 1.1f * 2.0f;

// The order statistic at index floor(p (n - 1)) of `values` (reordered), so it's deterministic for
// any sort algorithm. `values` must be non-empty.
inline float orderStatistic(std::vector<float>& values, float p) {
    const auto k = static_cast<std::size_t>(std::floor(p * static_cast<float>(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(k), values.end());
    return values[k];
}

// defaultCamera's results
inline constexpr uint32_t kDefaultCameraOk = 0;
inline constexpr uint32_t kDefaultCameraOutOfMemory = 1;  // the positions' copy couldn't be allocated

// The camera framing the splats' positions: per axis, the 5th and 95th percentiles bound a box with
// centre c and radius r (half its diagonal; 1 when the box is a point or the scene is empty, whose c
// is 0). The camera looks down world +z (identity rotation, COLMAP: image y down) from c - d e_z,
// d = 2.2 r, so the box is inside the image. All in float32. The positions must be finite (the
// encoder checks). Never throws, so a C ABI can wrap it: kDefaultCameraOk with `camera` set, or
// kDefaultCameraOutOfMemory with `camera` unchanged.
inline uint32_t defaultCamera(std::span<const gen::Splat> splats, gen::Camera& camera) noexcept {
    glm::vec3 centre(0.0f);
    float radius = 1.0f;
    if (!splats.empty()) {
        std::vector<float> axis;
        try {
            axis.resize(splats.size());
        } catch (const std::bad_alloc&) {
            return kDefaultCameraOutOfMemory;
        }
        glm::vec3 low, high;
        for (int a = 0; a < 3; ++a) {
            for (std::size_t i = 0; i < splats.size(); ++i) axis[i] = splats[i].position[a];
            low[a] = orderStatistic(axis, kDefaultCameraLowPercentile);
            high[a] = orderStatistic(axis, kDefaultCameraHighPercentile);
        }
        centre = 0.5f * (low + high);
        const glm::vec3 half = 0.5f * (high - low);
        const float r = std::sqrt(half.x * half.x + half.y * half.y + half.z * half.z);
        if (r > 0.0f) radius = r;
    }
    const glm::vec3 eye = centre - glm::vec3(0.0f, 0.0f, kDefaultCameraDistance * radius);
    glm::mat4 worldToCamera(1.0f);
    worldToCamera[3] = glm::vec4(-eye, 1.0f);  // R = I, t = -R eye
    camera = gen::Camera{
        .worldToCamera = worldToCamera,
        .intrinsics = glm::vec4(kDefaultCameraFocal, kDefaultCameraFocal, 0.5f * kDefaultCameraWidth,
                                0.5f * kDefaultCameraHeight),
        .center = eye,
        .pixelOffset = 0,
        .size = glm::uvec2(kDefaultCameraWidth, kDefaultCameraHeight),
        .imageBatch = 0xFFFFFFFFu,
        .heldOut = 0,
    };
    return kDefaultCameraOk;
}

}  // namespace splat_native
