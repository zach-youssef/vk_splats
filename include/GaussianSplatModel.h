// Zach Youssef, 4/20/26
// Header defining class abstration for our gaussian splat data

#pragma once

#include <VulkanApp.h>

#include <fstream>

#include "GpuTypes.h"
#include "ColmapImportUtils.h"
#include "KdTreeDist.h"

namespace {
    // Helper for parsing floats from a csv file
    float nextFloat(std::string& val, std::stringstream& lineStream) {
        std::getline(lineStream, val, ',');
        return std::stof(val);
    }
}

class GaussianSplatModel {
public:
    GaussianSplatModel(){}

    // Initializes a splat model from a colmap export
    void initFromSparsePointFile(const std::string& file) {
        // Cache the positions as a vector for initial scale computation
        std::vector<std::vector<float>> positions;
        // Read the initial means and colors from the point cloud
        splats = readColmapFile<Splat>(file, [&positions](const std::vector<std::string>& line){
            // Parse position / splat mean
            glm::vec3 mean;
            for (int i = 0; i < 3; ++i) {
                mean[i] = std::stof(line[i + 1]);
            }
            // Cache position for scale computation
            positions.push_back(std::vector<float>{mean.x, mean.y, mean.z});

            // Parse color
            glm::vec3 rgb;
            for (int i = 0; i < 3; ++i) {
                rgb[i] = std::stof(line[i + 4]);
            }

            // Construct splat
            Splat splat;
            splat.mean = glm::vec4(mean, 0);
            splat.rotation = glm::vec4(1, 0, 0, 0);
            splat.sh[0] = glm::vec4(rgb * (1.0f / 0.28209479177387814f), 0);
            for (int i = 1; i < 16; i++){
                splat.sh[i] = glm::vec4(0);
            }
            splat.alpha = glm::vec4(0.1);
            return splat;
        });

        // Initialize the scale of each splat to be the average distance
        // to it's 3 closest neighbors
        KdTree kd = KdTree(positions, 3);
        for(auto& splat : splats) {
            auto initScale = kd.avgDistToKNearest(std::vector<float>{splat.mean.x, splat.mean.y, splat.mean.z}, 3);
            splat.scale = glm::vec4{initScale[0], initScale[1], initScale[2], 0};
        }
    }

    // Intializes the model with saved results
    void initFromSplatCSV(const std::string& file) {
        std::ifstream csv;
        csv.open(file);
        std::string line;
        while(std::getline(csv, line)) {
            std::stringstream lineStream(line);
            Splat splat{};
            std::string val;
            splat.mean.x = nextFloat(val, lineStream);
            splat.mean.y = nextFloat(val, lineStream);
            splat.mean.z = nextFloat(val, lineStream);
            splat.scale.x = nextFloat(val, lineStream);
            splat.scale.y = nextFloat(val, lineStream);
            splat.scale.z = nextFloat(val, lineStream);
            splat.rotation.x = nextFloat(val, lineStream);
            splat.rotation.y = nextFloat(val, lineStream);
            splat.rotation.z = nextFloat(val, lineStream);
            splat.rotation.w = nextFloat(val, lineStream);
            splat.alpha[0] = nextFloat(val, lineStream);
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 3; ++j) {
                    splat.sh[i][j] = nextFloat(val, lineStream);
                }
            }
            splats.push_back(splat);
        }
    }

    // Creates the required buffers to render/train the splat model
    // and uploads the currently loaded splats
    void initializeBuffers(VulkanApp<1>& app) {
        // Create GPU buffers
        Buffer<Splat>::create(stagingBuffer_, 
                              splats.size(), 
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              app.getDevice(),
                              app.getPhysicalDevice());
        Buffer<Splat>::create(splatBuffer_, 
                              splats.size(), 
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                              app.getDevice(),
                              app.getPhysicalDevice());
        Buffer<ImagespaceSplat>::create(imagespaceBuffer_, 
                                        splats.size(), 
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                        app.getDevice(),
                                        app.getPhysicalDevice());
        Buffer<RasterGrad>::create(rasterGradBuffer_, 
                                   splats.size(), 
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                   app.getDevice(),
                                   app.getPhysicalDevice());
        Buffer<Splat>::create(gradBuffer_, 
                              splats.size(), 
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                              app.getDevice(),
                              app.getPhysicalDevice());
        Buffer<float>::createAndInitialize(dposMagBuffer_, 
                                           std::vector<float>(splats.size(), 0), 
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                           app.getDevice(),
                                           app.getPhysicalDevice(),
                                           app.getGraphicsQueue(),
                                           app.getCommandPool());
        Buffer<glm::vec4>::create(densityFlagBuffer_, 
                                  splats.size(), 
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                  app.getDevice(),
                                  app.getPhysicalDevice());

        // Upload initial splat data to the staging buffer
        stagingBuffer_->mapAndExecute(0, sizeof(Splat) * splats.size(), [this](void* mappedBuffer){
            memcpy(mappedBuffer, splats.data(), sizeof(Splat) * splats.size());
        });

        // Copy the staging buffer into the GPU splat buffer
        Buffer<Splat>::copyBuffer(stagingBuffer_->getBuffer(),
                                  splatBuffer_->getBuffer(),
                                  sizeof(Splat) * splats.size(),
                                  app.getGraphicsQueue(),
                                  app.getDevice(),
                                  app.getCommandPool());
    }

    void saveModel(const std::string& outFile, VulkanApp<1>& app) {
        // Copy the current GPU data back to the CPU
        syncCPUSplats(app);

        // Save splats to a CSV, with each line represenitng a single splat
        std::ofstream csv;
        csv.open(outFile, std::ios::trunc);
        for (const auto& splat : splats) {
            csv << splat.mean.x << "," << splat.mean.y << "," << splat.mean.z << ",";
            csv << splat.scale.x << "," << splat.scale.y << "," << splat.scale.z << ",";
            csv << splat.rotation.x << "," << splat.rotation.y << "," << splat.rotation.z << "," << splat.rotation.w << ",";
            csv << splat.alpha[0] << ",";
            for (int i = 0; i < 16; ++i) {
                glm::vec3 sh = splat.sh[i];
                csv << sh.x << "," << sh.y << "," << sh.z;
                if (i != 15) {
                    csv << ",";
                }
            }
            csv << std::endl;
        }
        csv.close();
    }

    void performDensityOptimization(VulkanApp<1>& app){
        // Copy current splats to CPU
        syncCPUSplats(app);

        // Grab the density operation for each splat
        std::vector<glm::vec4> densityFlags(splats.size(), glm::vec4(0));
        densityFlagBuffer_->mapAndExecute(0, splats.size() * sizeof(glm::vec4), [&](void* mapped){
            memcpy(densityFlags.data(), mapped, sizeof(glm::vec4) * splats.size());
        });

        // Perform density control
        std::vector<Splat> newSplats;
        newSplats.reserve(splats.size());
        for (int idx = 0; idx < splats.size(); ++idx) {
            const auto& splat = splats.at(idx);
            const glm::vec4& flag = densityFlags.at(idx);

            if (flag.w == 0) {
                // Prune
                continue;
            } else if (flag.w == 1) {
                // Unmodified
                newSplats.push_back(splat);
            } else {
                glm::vec3 delta{flag.x, flag.y, flag.z};
                if (flag.w == 2) {
                    // Split
                    Splat s0(splat);
                    s0.scale /= 2;
                    s0.mean += glm::vec4(delta, 0);
                    newSplats.push_back(s0);

                    Splat s1(splat);
                    s1.scale /= 2;
                    s1.mean -= glm::vec4(delta, 0);
                    newSplats.push_back(s1);
                } else if (flag.w == 3) {
                    // Clone
                    newSplats.push_back(splat);
                    Splat clone(splat);
                    clone.mean += glm::vec4(delta, 0);
                    newSplats.push_back(clone);
                }
            }
        }

        // Reinitialize GPU objects with our new splat size
        this->splats = newSplats;
        initializeBuffers(app);
    }

    // Returns the maximum valid index into the splat buffer
    int maxSplatIndex() {
        return splats.size() - 1;
    }

    VkBuffer splatBuffer() {
        return splatBuffer_->getBuffer();
    }

    VkBuffer imagespaceSplats() {
        return imagespaceBuffer_->getBuffer();
    }

    VkBuffer rasterGradients() {
        return rasterGradBuffer_->getBuffer();
    }

    VkBuffer finalGradients() {
        return gradBuffer_->getBuffer();
    }

    VkBuffer positionGradientMagnitudes() {
        return dposMagBuffer_->getBuffer();
    }

    VkBuffer densityFlags() {
        return densityFlagBuffer_->getBuffer();
    }

private:
    void syncCPUSplats(VulkanApp<1>& app) {
        // Copy the current GPU data back to the CPU
        Buffer<Splat>::copyBuffer(splatBuffer_->getBuffer(),
                                  stagingBuffer_->getBuffer(),
                                  sizeof(Splat) * splats.size(),
                                  app.getGraphicsQueue(),
                                  app.getDevice(),
                                  app.getCommandPool());
        stagingBuffer_->mapAndExecute(0, sizeof(Splat) * splats.size(), [this](void* mappedBuffer){
            memcpy(splats.data(), mappedBuffer, sizeof(Splat) * splats.size());
        });
    }

private:
    // CPU Copy of splat data
    std::vector<Splat> splats;

    ///
    // Per splat buffers
    ///
    // Staging buffer (used for initial upload & saving out)
    std::unique_ptr<Buffer<Splat>> stagingBuffer_;
    //  Actual Splats
    std::unique_ptr<Buffer<Splat>> splatBuffer_;
    //  Imagepsace splats
    std::unique_ptr<Buffer<ImagespaceSplat>> imagespaceBuffer_;
    //  Rasterization gradients
    std::unique_ptr<Buffer<RasterGrad>> rasterGradBuffer_;
    //  Preprocessing gradients
    std::unique_ptr<Buffer<Splat>> gradBuffer_;
    // Density control bookeeping
    std::unique_ptr<Buffer<float>> dposMagBuffer_;
    // Density control flags per splat
    std::unique_ptr<Buffer<glm::vec4>> densityFlagBuffer_;
};