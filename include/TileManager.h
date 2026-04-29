// Zach Youssef, 4/22/26
// Header defining a class for managing per-tile GPU objects

#pragma once

#include <VulkanApp.h>

#include "GpuTypes.h"

class TileManager {
    static const int T = 16;
public:
    TileManager(size_t maxSplatsPerTile): maxSplatsPerTile_(maxSplatsPerTile) {}

    // Initialize all per-tile GPU buffers
    void initBuffers(VulkanApp<1>& app, int maxSplatIdx, int image_size) {
        // Initialize raster controls
        RasterControls control;
        control.max_splat_idx = maxSplatIdx;
        control.max_splats_per_tile = maxSplatsPerTile_;
        // Compute extents for each tile
        int tile_size = std::ceil(image_size / 4.0);
        for(int i = 0; i < 16; ++i) {
            uint minX = (i % 4) * tile_size;
            uint minY = (i / 4) * tile_size;
            control.tile_extents[i] = glm::uvec4(minX, minY, minX + tile_size, minY + tile_size);
        }

        // Initialize buffers
        Buffer<SplatEntry>::create(entryBuffer0_,
                                    maxSplatsPerTile_ * 16,
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                    app.getDevice(),
                                    app.getPhysicalDevice());
        Buffer<SplatEntry>::create(entryBuffer1_,
                                    maxSplatsPerTile_ * 16,
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                    app.getDevice(),
                                    app.getPhysicalDevice());
        Buffer<TileSplatCount>::create(countBuffer_,
                                        1,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                        app.getDevice(),
                                        app.getPhysicalDevice());
        Buffer<RasterControls>::createAndInitialize(controlBuffer_,
                                                    std::vector<RasterControls>{control},
                                                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                    app.getDevice(),
                                                    app.getPhysicalDevice(),
                                                    app.getGraphicsQueue(),
                                                    app.getCommandPool());
    }

    VkBuffer tileEntries0() {
        return entryBuffer0_->getBuffer();
    }
    VkBuffer tileEntries1() {
        return entryBuffer1_->getBuffer();
    }

    VkBuffer count() {
        return countBuffer_->getBuffer();
    }

    VkBuffer control() {
        return controlBuffer_->getBuffer();
    }

    void setMaxIndex(VulkanApp<1>& app, int maxIndex) {
        controlBuffer_->mapAndExecute(offsetof(RasterControls, max_splat_idx), sizeof(int), [maxIndex](void* mapped){
            *(int*)mapped = maxIndex;
        });
    }

private:
    const int maxSplatsPerTile_;

    std::unique_ptr<Buffer<SplatEntry>> entryBuffer0_;
    std::unique_ptr<Buffer<SplatEntry>> entryBuffer1_;
    std::unique_ptr<Buffer<TileSplatCount>> countBuffer_;
    std::unique_ptr<Buffer<RasterControls>> controlBuffer_;
};