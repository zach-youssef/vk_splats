// The default camera (lib/native/default_camera.h), through the C++ ZGAL generates for the splat
// math in default_camera_test.zgal.h: its constants, an empty scene and a point, and boxes of every
// shape (with floaters) whose 5th-95th percentile corners must all project inside the image, for any
// splat order.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include "default_camera_test.zgal.h"
namespace splat_native { namespace gen = ::default_camera_test; }
#include "../../lib/native/default_camera.h"

#include "test_random.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace dc = default_camera_test;
namespace sn = splat_native;

TEST_CASE("the default camera frames the scene") {
    SUBCASE("an empty scene: centre 0, radius 1") {
        dc::Camera cam;
        REQUIRE(sn::defaultCamera({}, cam) == sn::kDefaultCameraOk);
        CHECK(cam.center == glm::vec3(0.0f, 0.0f, -2.2f));
        CHECK(cam.worldToCamera == glm::mat4(glm::vec4(1, 0, 0, 0), glm::vec4(0, 1, 0, 0), glm::vec4(0, 0, 1, 0),
                                             glm::vec4(0, 0, 2.2f, 1)));
        CHECK(cam.intrinsics == glm::vec4(sn::kDefaultCameraFocal, sn::kDefaultCameraFocal, 640, 360));
        CHECK(cam.size == glm::uvec2(1280, 720));
        CHECK(cam.imageBatch == 0xFFFFFFFFu);
        CHECK(cam.pixelOffset == 0);
        CHECK(cam.heldOut == 0);
        // 60 degrees vertically
        CHECK(std::abs(360.0 / sn::kDefaultCameraFocal - std::tan(3.14159265358979323846 / 6)) < 1e-7);
        CHECK(std::abs(double(sn::kDefaultCameraTanHalfFov) - std::tan(3.14159265358979323846 / 6)) < 1e-7);
    }
    SUBCASE("a point: radius 1 around it") {
        std::vector<dc::Splat> splats(3);
        for (dc::Splat& s : splats) s.position = glm::vec3(1, 2, 3);
        dc::Camera cam;
        REQUIRE(sn::defaultCamera(splats, cam) == sn::kDefaultCameraOk);
        CHECK(cam.center == glm::vec3(1.0f, 2.0f, 3.0f - 2.2f));
    }
    SUBCASE("boxes of every shape are inside the image, whatever the splats' order") {
        TestRandom rng(7);
        for (int scene = 0; scene < 60; ++scene) {
            CAPTURE(scene);
            const int n = rng.integer(1, 400);
            // Half extents, some flat: the worst case for the vertical fit is a box thin in x
            glm::vec3 half{float(rng.uniform(0, 10)), float(rng.uniform(0, 10)), float(rng.uniform(0, 10))};
            if (scene % 3 == 0) half[scene % 9 / 3] = 0.0f;
            if (scene == 1) half = glm::vec3(0.0f, 7.0f, 7.0f);
            const glm::vec3 centre{float(rng.uniform(-50, 50)), float(rng.uniform(-50, 50)), float(rng.uniform(-50, 50))};
            std::vector<dc::Splat> splats(n);
            for (int i = 0; i < n; ++i)
                splats[i].position = centre + half * glm::vec3{float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1)),
                                                               float(rng.uniform(-1, 1))};
            // A few floaters far away, beyond the percentiles
            if (n >= 100) splats[0].position = glm::vec3(1e5f), splats[1].position = glm::vec3(-1e5f);
            dc::Camera cam;
            REQUIRE(sn::defaultCamera(splats, cam) == sn::kDefaultCameraOk);
            // The box, from sorted positions
            glm::vec3 lo, hi;
            for (int a = 0; a < 3; ++a) {
                std::vector<float> v(n);
                for (int i = 0; i < n; ++i) v[i] = splats[i].position[a];
                std::sort(v.begin(), v.end());
                lo[a] = v[std::size_t(std::floor(0.05f * float(n - 1)))];
                hi[a] = v[std::size_t(std::floor(0.95f * float(n - 1)))];
            }
            for (int corner = 0; corner < 8; ++corner) {
                const glm::vec3 p{(corner & 1) ? hi.x : lo.x, (corner & 2) ? hi.y : lo.y, (corner & 4) ? hi.z : lo.z};
                const glm::vec3 t = dc::toCamera(cam.worldToCamera, p);
                CAPTURE(corner);
                REQUIRE(t.z > 0.0f);
                const glm::vec2 px = dc::projectMean(t, cam.intrinsics);
                CHECK(px.x >= 0.0f);
                CHECK(px.x <= 1280.0f);
                CHECK(px.y >= 0.0f);
                CHECK(px.y <= 720.0f);
            }
            // Deterministic: reversed and rotated input orders give the same camera, bit for bit
            std::vector<dc::Splat> reordered(splats.rbegin(), splats.rend());
            std::rotate(reordered.begin(), reordered.begin() + n / 3, reordered.end());
            dc::Camera again;
            REQUIRE(sn::defaultCamera(reordered, again) == sn::kDefaultCameraOk);
            CHECK(std::memcmp(&again, &cam, sizeof cam) == 0);
        }
    }
}
