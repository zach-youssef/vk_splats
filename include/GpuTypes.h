// Zach Youssef, 4/21/26
// Header defining structs for data used in our shaders

#pragma once

#include <glm/glm.hpp>

///////////////////////
// These two aren't actually uploaded to the GPU, just here for convenience
struct CameraPosition {
    glm::mat4 view;
    int32_t imageId;
};

struct CameraIntrinsics {
    glm::mat4 projection;
    int32_t image_size;
    float focal_length;
};
///////////////////////

////
// All the structs that are used as storage buffer arrays must be padded out to a multiple of 16 bytes
// (See std140 standard)
////

struct CameraProps {
    glm::mat4 view;
    glm::mat4 projection;
    int image_size;
    float focal_length;
    float tan_fov;
};

struct Splat {
    glm::vec4 mean;
    glm::vec4 scale;
    glm::vec4 rotation;
    glm::vec4 alpha;
    glm::vec4 sh[16];
};
static_assert(sizeof(Splat) % 16 == 0);

struct ImagespaceSplat {
    glm::vec2 mean;
    float depth;
    float _padding0;
    glm::vec3 conic;
    float radius;
    glm::uvec4 extent;
    glm::vec3 rgb;
    float _padding1;
    glm::vec3 rgbClamped;
    float alpha;
};
static_assert(sizeof(ImagespaceSplat) == 0x50);

struct PreprocessControls {
    int max_index;
    int sh_degree;
    float near_plane;
};

struct SplatEntry {
    uint index;
    uint depth;
    glm::vec2 _padding0;
};
static_assert(sizeof(SplatEntry) % 16 == 0);

struct RasterGrad {
    glm::vec4 dl_drgb;
    glm::vec2 dl_dmean;
    float dl_dconicx;
    float dl_dconicy;
    float dl_dconicz;
    float dl_dalpha;
    glm::vec2 _padding0;
};
static_assert(sizeof(RasterGrad) % 16 == 0);

struct TileSplatCount {
    uint numSplats[16];
};
static_assert(sizeof(TileSplatCount) % 16 == 0);

struct RasterControls {
    glm::uvec4 tile_extents[16];
    int max_splat_idx;
    int max_splats_per_tile;
};

struct LearningRates {
    float mean;
    float scale;
    float rotation;
    float alpha;
    float sh_0;
    float sh_h;
};