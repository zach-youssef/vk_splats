// The view camera (lib/view_camera.zgal, through the C++ ZGAL generates for it in
// view_camera_test.zgal.h): the untouched pose gives the start camera exactly, fitting to another size
// scales the image about its centre with square pixels, looks and moves go the documented ways and
// round-trip, the camera never rolls (its x axis stays perpendicular to the start's up), and speeds
// come back exactly when scrolled back.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include "view_camera_test.zgal.h"
namespace splat_native { namespace gen = ::view_camera_test; }
#include "../../lib/native/default_camera.h"

#include "test_random.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace vc = view_camera_test;

namespace {

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------
// A camera with rotation from a unit quaternion (w, x, y, z), centre c and its intrinsics and size
vc::Camera makeCamera(glm::vec4 q, glm::vec3 c, glm::vec4 intrinsics, glm::uvec2 size) {
    const glm::mat3 r = vc::quatToMatrix(q);
    const glm::vec3 t = -(r * c);
    glm::mat4 w(r);
    w[3] = glm::vec4(t, 1.0f);
    return vc::Camera{.worldToCamera = w, .intrinsics = intrinsics, .center = c, .pixelOffset = 7,
                      .size = size, .imageBatch = 2, .heldOut = 1};
}

// Random cameras: any rotation, off-centre principal points, odd and non-square sizes
std::vector<vc::Camera> randomCameras(uint32_t seed, int n) {
    TestRandom rng(seed);
    std::vector<vc::Camera> cameras;
    for (int i = 0; i < n; ++i) {
        const glm::vec4 q(float(rng.normal()), float(rng.normal()), float(rng.normal()), float(rng.normal()));
        const glm::vec3 c(float(rng.uniform(-5, 5)), float(rng.uniform(-5, 5)), float(rng.uniform(-5, 5)));
        const glm::uvec2 size(uint32_t(rng.integer(17, 1999)), uint32_t(rng.integer(9, 1501)));
        const float f = float(rng.uniform(0.5, 2.0)) * float(size.y);
        const glm::vec4 k(f, f * float(rng.uniform(0.9, 1.1)), float(rng.uniform(0.0, 1.0)) * float(size.x),
                          float(rng.uniform(0.0, 1.0)) * float(size.y));
        cameras.push_back(makeCamera(glm::normalize(q), c, k, size));
    }
    return cameras;
}

// The camera's axes in world space (rows of R): x right, y down, z forward
glm::vec3 axis(const vc::Camera& cam, int a) { return glm::transpose(glm::mat3(cam.worldToCamera))[a]; }
glm::vec3 up(const vc::Camera& cam) { return -axis(cam, 1); }

// A world point's pixel through a camera, and its camera-space depth
glm::vec2 project(const vc::Camera& cam, glm::vec3 p) {
    return vc::projectMean(vc::toCamera(cam.worldToCamera, p), cam.intrinsics);
}
float depth(const vc::Camera& cam, glm::vec3 p) { return vc::toCamera(cam.worldToCamera, p).z; }

float maxDiff(glm::vec3 a, glm::vec3 b) {
    const glm::vec3 d = glm::abs(a - b);
    return std::max(d.x, std::max(d.y, d.z));
}
float maxDiff(const glm::mat4& a, const glm::mat4& b) {
    float m = 0.0f;
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) m = std::max(m, std::abs(a[c][r] - b[c][r]));
    return m;
}

// The fitted camera is a rigid camera: R orthonormal, the last row (0, 0, 0, 1), center = -R^T t
void checkRigid(const vc::Camera& cam) {
    const glm::mat3 r(cam.worldToCamera);
    const glm::mat3 i = glm::transpose(r) * r;
    for (int c = 0; c < 3; ++c)
        for (int k = 0; k < 3; ++k) CHECK(std::abs(i[c][k] - (c == k ? 1.0f : 0.0f)) < 1e-5f);
    CHECK(glm::determinant(r) > 0.0f);
    for (int c = 0; c < 3; ++c) CHECK(cam.worldToCamera[c][3] == 0.0f);
    CHECK(cam.worldToCamera[3][3] == 1.0f);
    const glm::vec3 centre = -(glm::transpose(r) * glm::vec3(cam.worldToCamera[3]));
    CHECK(maxDiff(centre, cam.center) < 1e-4f * (1.0f + glm::length(cam.center)));
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------
TEST_CASE("the untouched pose at the start's size is the start camera exactly") {
    const vc::ViewPose pose = vc::viewPose();
    CHECK(pose.offset == glm::vec3(0.0f));
    CHECK(pose.yaw == 0.0f);
    CHECK(pose.pitch == 0.0f);
    // The stream header's default camera (lib/native/default_camera.h), then random ones
    std::vector<vc::Camera> starts(1);
    REQUIRE(splat_native::defaultCamera({}, starts[0]) == splat_native::kDefaultCameraOk);
    for (const vc::Camera& c : randomCameras(1, 200)) starts.push_back(c);
    for (const vc::Camera& start : starts) {
        const vc::Camera cam = vc::fitCamera(pose, start, start.size);
        CHECK(cam.worldToCamera == start.worldToCamera);
        CHECK(cam.intrinsics == start.intrinsics);
        CHECK(cam.center == start.center);
        CHECK(cam.size == start.size);
        CHECK(cam.pixelOffset == start.pixelOffset);
        CHECK(cam.imageBatch == start.imageBatch);
        CHECK(cam.heldOut == start.heldOut);
    }
}

TEST_CASE("fitting to another size scales the image about its centre, with square pixels") {
    const std::vector<glm::uvec2> sizes = {{640, 480}, {1920, 1080}, {300, 900}, {1, 1}, {4097, 33}};
    TestRandom rng(2);
    for (const vc::Camera& start : randomCameras(3, 50)) {
        for (glm::uvec2 size : sizes) {
            const vc::Camera cam = vc::fitCamera(vc::viewPose(), start, size);
            CHECK(cam.size == size);
            CHECK(cam.worldToCamera == start.worldToCamera);
            CHECK(cam.center == start.center);
            // One scale s = H'/H for both focal lengths, so the pixel aspect is the start's
            const double s = double(size.y) / double(start.size.y);
            CHECK(std::abs(cam.intrinsics.x - s * start.intrinsics.x) <= 1e-6 * s * start.intrinsics.x);
            CHECK(std::abs(cam.intrinsics.y - s * start.intrinsics.y) <= 1e-6 * s * start.intrinsics.y);
            CHECK(std::abs(double(cam.intrinsics.x) / cam.intrinsics.y -
                           double(start.intrinsics.x) / start.intrinsics.y) < 1e-6);
            // cx' = W'/2 + s (cx - W/2), cy' = H'/2 + s (cy - H/2)
            const double cx = size.x / 2.0 + s * (start.intrinsics.z - start.size.x / 2.0);
            const double cy = size.y / 2.0 + s * (start.intrinsics.w - start.size.y / 2.0);
            const double tol = 1e-5 * (1.0 + s * (start.size.x + start.size.y));
            CHECK(std::abs(cam.intrinsics.z - cx) < tol);
            CHECK(std::abs(cam.intrinsics.w - cy) < tol);
            // So every point lands at u' = W'/2 + s (u - W/2), with the image's centre fixed
            for (int i = 0; i < 4; ++i) {
                const glm::vec3 p = start.center + 3.0f * axis(start, 2) +
                                    glm::vec3(float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1)),
                                              float(rng.uniform(-1, 1)));
                if (depth(start, p) < 0.5f) continue;
                const glm::dvec2 u(project(start, p));
                const glm::dvec2 expected = glm::dvec2(size) / 2.0 + s * (u - glm::dvec2(start.size) / 2.0);
                const glm::dvec2 got(project(cam, p));
                CHECK(glm::length(got - expected) < 1e-4 * (1.0 + s * glm::length(u)));
            }
            // And back to the start's size: the start's intrinsics, to float precision at the start's
            // scale (a 1x1 fit rounds the principal point at a 1-pixel scale, so allow a few ulps of
            // that, scaled back up)
            const vc::Camera back = vc::fitCamera(vc::viewPose(), cam, start.size);
            const float scale = start.intrinsics.x + float(start.size.x + start.size.y);
            for (int c = 0; c < 4; ++c)
                CHECK(std::abs(back.intrinsics[c] - start.intrinsics[c]) < 1e-5f * scale);
        }
    }
}

TEST_CASE("look turns right and up, about the start's up") {
    for (const vc::Camera& start : randomCameras(4, 50)) {
        const glm::vec3 right0 = axis(start, 0), forward0 = axis(start, 2), up0 = up(start);
        // Right: forward goes to (sin a, 0, cos a) in the start's frame
        const float a = 0.3f;
        const vc::ViewPose turned = vc::viewLook(vc::viewPose(), glm::vec2(a / 0.002f, 0.0f), 0.002f);
        CHECK(std::abs(turned.yaw - a) < 1e-6f);
        CHECK(turned.pitch == 0.0f);
        const vc::Camera t = vc::fitCamera(turned, start, start.size);
        CHECK(maxDiff(axis(t, 2), std::sin(a) * right0 + std::cos(a) * forward0) < 1e-5f);
        // Turning only about up leaves the camera's y axis exactly the start's
        CHECK(axis(t, 1) == axis(start, 1));
        checkRigid(t);
        // Dragging down (+y) looks down: forward tips away from up
        const vc::ViewPose lowered = vc::viewLook(vc::viewPose(), glm::vec2(0.0f, 100.0f), 0.002f);
        CHECK(std::abs(lowered.pitch + 0.2f) < 1e-6f);
        const vc::Camera l = vc::fitCamera(lowered, start, start.size);
        CHECK(maxDiff(axis(l, 2), std::cos(0.2f) * forward0 - std::sin(0.2f) * up0) < 1e-5f);
        CHECK(maxDiff(axis(l, 0), right0) < 1e-6f);
        checkRigid(l);
        // Looking never moves the camera
        CHECK(l.center == start.center);
    }
}

TEST_CASE("look round-trips, wraps the yaw and clamps the pitch") {
    TestRandom rng(5);
    for (const vc::Camera& start : randomCameras(6, 20)) {
        vc::ViewPose pose = vc::viewPose();
        for (int i = 0; i < 50; ++i) {
            const glm::vec2 d(float(rng.uniform(-300, 300)), float(rng.uniform(-100, 100)));
            const vc::ViewPose there = vc::viewLook(pose, d, 0.003f);
            if (std::abs(there.pitch) < vc::viewPitchLimit && std::abs(there.yaw) < 3.0f) {
                const vc::ViewPose back = vc::viewLook(there, -d, 0.003f);
                CHECK(std::abs(back.yaw - pose.yaw) < 1e-5f);
                CHECK(std::abs(back.pitch - pose.pitch) < 1e-5f);
                const vc::Camera a = vc::fitCamera(back, start, start.size);
                const vc::Camera b = vc::fitCamera(pose, start, start.size);
                CHECK(maxDiff(a.worldToCamera, b.worldToCamera) < 1e-4f * (1.0f + glm::length(start.center)));
            }
            pose = there;
        }
    }
    // A full turn in steps comes back to the same view, with the yaw kept in [-pi, pi]
    vc::ViewPose pose = vc::viewPose();
    for (int i = 0; i < 1000; ++i) {
        pose = vc::viewLook(pose, glm::vec2(17.0f, 0.0f), 0.01f);
        CHECK(pose.yaw >= -vc::viewPi);
        CHECK(pose.yaw <= vc::viewPi);
    }
    CHECK(std::abs(std::remainder(double(pose.yaw) - 170.0, 2.0 * 3.14159265358979323846)) < 1e-3);
    // The pitch stops at the limit, both ways
    CHECK(vc::viewLook(vc::viewPose(), glm::vec2(0.0f, -1e6f), 0.01f).pitch == vc::viewPitchLimit);
    CHECK(vc::viewLook(vc::viewPose(), glm::vec2(0.0f, 1e6f), 0.01f).pitch == -vc::viewPitchLimit);
}

TEST_CASE("up doesn't drift under look: the camera never rolls or flips") {
    TestRandom rng(7);
    for (const vc::Camera& start : randomCameras(8, 10)) {
        const glm::vec3 up0 = up(start);
        vc::ViewPose pose = vc::viewPose();
        for (int i = 0; i < 2000; ++i) {
            pose = vc::viewLook(pose, glm::vec2(float(rng.uniform(-500, 500)), float(rng.uniform(-500, 500))),
                                0.004f);
            const vc::Camera cam = vc::fitCamera(pose, start, start.size);
            // The x axis stays level: perpendicular to the start's up
            CHECK(std::abs(glm::dot(axis(cam, 0), up0)) < 1e-5f);
            // and the image's up is never below the horizon: up . (-y) = cos pitch > 0
            CHECK(glm::dot(up(cam), up0) >= std::cos(vc::viewPitchLimit) - 1e-5f);
            CHECK(std::abs(glm::dot(up(cam), up0) - std::cos(pose.pitch)) < 1e-5f);
        }
        // The pose is the state, so the view after any looks depends on (yaw, pitch) only: looking back
        // to pitch 0 gives the start's up exactly
        const vc::ViewPose level = vc::viewLook(pose, glm::vec2(0.0f, pose.pitch / 0.004f), 0.004f);
        if (level.pitch == 0.0f) CHECK(up(vc::fitCamera(level, start, start.size)) == up0);
        CHECK(up(vc::fitCamera(vc::ViewPose{pose.offset, pose.yaw, 0.0f}, start, start.size)) == up0);
    }
}

TEST_CASE("move goes along the current right and forward and the start's up") {
    TestRandom rng(9);
    for (const vc::Camera& start : randomCameras(10, 30)) {
        const glm::vec3 up0 = up(start);
        const vc::ViewPose turned =
            vc::viewLook(vc::viewPose(), glm::vec2(float(rng.uniform(-500, 500)), float(rng.uniform(-200, 200))),
                         0.005f);
        const vc::Camera t = vc::fitCamera(turned, start, start.size);
        const float span = float(rng.uniform(0.1, 3.0));
        // W: forward, with the pitch
        const vc::ViewPose w = vc::viewMove(turned, start, glm::vec3(0, 0, 1), span);
        CHECK(maxDiff(w.offset, span * axis(t, 2)) < 1e-5f);
        CHECK(w.yaw == turned.yaw);
        CHECK(w.pitch == turned.pitch);
        // D: right
        CHECK(maxDiff(vc::viewMove(turned, start, glm::vec3(1, 0, 0), span).offset, span * axis(t, 0)) < 1e-5f);
        // E: the start's up, whatever the pitch; Q: down
        CHECK(maxDiff(vc::viewMove(turned, start, glm::vec3(0, 1, 0), span).offset, span * up0) < 1e-5f);
        CHECK(maxDiff(vc::viewMove(turned, start, glm::vec3(0, -1, 0), span).offset, -span * up0) < 1e-5f);
        // A diagonal is as fast as one key
        CHECK(std::abs(glm::length(vc::viewMove(turned, start, glm::vec3(1, 0, 1), span).offset) - span) < 1e-5f);
        // No keys, or keys that cancel: no move
        CHECK(vc::viewMove(turned, start, glm::vec3(0), span).offset == glm::vec3(0));
        // The fitted camera moved, and is still a rigid camera at the moved centre
        const vc::Camera m = vc::fitCamera(w, start, start.size);
        CHECK(maxDiff(m.center, start.center + span * axis(t, 2)) < 1e-5f * (1.0f + glm::length(start.center)));
        CHECK(maxDiff(glm::mat4(glm::mat3(m.worldToCamera)), glm::mat4(glm::mat3(t.worldToCamera))) == 0.0f);
        checkRigid(m);
        // A point ahead gets closer by span
        const glm::vec3 p = start.center + 10.0f * axis(t, 2);
        CHECK(std::abs(depth(t, p) - depth(m, p) - span) < 1e-4f * (1.0f + glm::length(start.center)));
        // Round trip: back the same way
        const vc::ViewPose back = vc::viewMove(w, start, glm::vec3(0, 0, -1), span);
        CHECK(maxDiff(back.offset, glm::vec3(0)) < 1e-5f);
        CHECK(maxDiff(vc::fitCamera(back, start, start.size).worldToCamera, t.worldToCamera) <
              1e-4f * (1.0f + glm::length(start.center)));
    }
}

TEST_CASE("speed scales by the step factor and comes back exactly") {
    CHECK(vc::viewSpeed(2.5f, 0.0f, 1.25f) == 2.5f);
    CHECK(std::abs(vc::viewSpeed(2.5f, 3.0f, 1.25f) - 2.5f * 1.25f * 1.25f * 1.25f) < 1e-6f);
    CHECK(std::abs(vc::viewSpeed(2.5f, -2.0f, 1.25f) - 2.5f / (1.25f * 1.25f)) < 1e-6f);
    CHECK(vc::viewSpeed(1.0f, 0.5f, 4.0f) == 2.0f);
    // Scrolling up and back down: the step count, and so the speed, is the same
    float steps = 0.0f;
    for (int i = 0; i < 7; ++i) steps += 1.0f;
    for (int i = 0; i < 7; ++i) steps -= 1.0f;
    CHECK(vc::viewSpeed(0.3f, steps, 1.1f) == 0.3f);
    // Monotone in the steps
    for (int i = -20; i < 20; ++i)
        CHECK(vc::viewSpeed(1.0f, float(i), 1.1f) < vc::viewSpeed(1.0f, float(i + 1), 1.1f));
}
