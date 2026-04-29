// Zach Youssef, 4/22/26
// Header defining a class for managing training parameters on the GPU

#pragma once

#include <VulkanApp.h>

#include "GpuTypes.h"

class TrainingManager {
public:
    TrainingManager(const LearningRates& learningRates, int maxIndex, float near_plane)
    :learningRates_(learningRates) {
        controls_ = PreprocessControls {maxIndex, 0, near_plane};
    }

    // Intialize GPU objects for controlling splat processing & learning rates
    void initBuffers(VulkanApp<1>& app, int numSamples) {
        // Create buffers
        Buffer<PreprocessControls>::create(controlBuffer_,
                                           1,
                                           VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                           app.getDevice(),
                                           app.getPhysicalDevice());
        Buffer<LearningRates>::create(rateBuffer_,
                                      1,
                                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                      app.getDevice(),
                                      app.getPhysicalDevice());
        Buffer<DensityControl>::create(densitySwitch_,
                                       1,
                                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                       app.getDevice(),
                                       app.getPhysicalDevice());

        // Initialize them to our starting values
        controlBuffer_->mapAndExecute(0, sizeof(PreprocessControls), [this](void* mappedBuffer){
            memcpy(mappedBuffer, &controls_, sizeof(PreprocessControls));
        });
        rateBuffer_->mapAndExecute(0, sizeof(LearningRates), [this](void* mappedBuffer){
            memcpy(mappedBuffer, &learningRates_, sizeof(LearningRates));
        });
        densitySwitch_->mapAndExecute(0, sizeof(DensityControl), [numSamples](void* mappedBuffer){
            DensityControl* ubo = (DensityControl*) mappedBuffer;
            ubo->numSamples = numSamples;
            ubo->performDensityControl = false;
        });
    }

    // Updates the number of SH degrees in use
    void setSHDegree(VulkanApp<1>& app, int shDegree) {
        controls_.sh_degree = shDegree; 
        controlBuffer_->mapAndExecute(offsetof(PreprocessControls, sh_degree), sizeof(int), [shDegree](void* mapped){
            *(int*)mapped = shDegree;
        });
    }

    // Update the learning rate for the splat means
    void setMeanLr(VulkanApp<1>& app, float meanLr) {
        learningRates_.mean = meanLr;
        rateBuffer_->mapAndExecute(offsetof(LearningRates, mean), sizeof(float), [meanLr](void* mapped){
            *(float*)mapped = meanLr;
        });
    }

    // Update the max splat index
    void setMaxIndex(VulkanApp<1>& app, int maxIndex) {
        controls_.max_index = maxIndex; 
        controlBuffer_->mapAndExecute(offsetof(PreprocessControls, max_index), sizeof(int), [maxIndex](void* mapped){
            *(int*)mapped = maxIndex;
        });
    }

    void setDensitySwitch(VulkanApp<1>& app, bool enableDensityControl) {
        densitySwitch_->mapAndExecute(offsetof(DensityControl, performDensityControl), sizeof(bool), [enableDensityControl](void* mappedBuffer){
            *((bool*) mappedBuffer) = enableDensityControl;
        });
    }

    VkBuffer control() {
        return controlBuffer_->getBuffer();
    }

    VkBuffer learningRates() {
        return rateBuffer_->getBuffer();
    }

    VkBuffer densitySwitch() {
        return densitySwitch_->getBuffer();
    }

private:
    LearningRates learningRates_;
    PreprocessControls controls_;

    // Uniform buffers
    std::unique_ptr<Buffer<PreprocessControls>> controlBuffer_;
    std::unique_ptr<Buffer<LearningRates>> rateBuffer_;
    std::unique_ptr<Buffer<DensityControl>> densitySwitch_;
};