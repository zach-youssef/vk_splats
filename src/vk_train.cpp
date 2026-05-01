// Zach Yousef, 4/20/26
// Entrypoint for vulkan implementation of training a gaussian splat model
#define STB_IMAGE_IMPLEMENTATION
#define VK_WRAP_UTIL_IMPL

#include <VulkanApp.h>
#include <Descriptor.h>
#include <Renderable.h>
#include <ComputeNode.h>
#include <RenderableNode.h>
#include <AcquireImageNode.h>
#include <PresentNode.h>
#include <FileUtil.h>
#include <BasicMaterial.h>

#include "CameraImageLoader.h"
#include "GaussianSplatModel.h"
#include "TileManager.h"
#include "TrainingManager.h"
#include "ImageToScreenRenderable.h"

#include <iostream>

// Must match shader side value
const int DENSITY_ITERATION_GAP = 100;

// TODO for the love of god find a better pattern than this
bool descriptorsDirty = false;

// Helpers to make descriptor construction easier
std::shared_ptr<StorageImageDescriptor<1>> storageImageDescriptor(VkImageView image) {
    return std::make_shared<StorageImageDescriptor<1>>(VK_SHADER_STAGE_COMPUTE_BIT, std::array<VkImageView, 1>{image});
}
std::shared_ptr<CombinedImageSamplerDescriptor<1>> sampledImageDescriptor(VkImageView image, VkSampler sampler) {
    return std::make_shared<CombinedImageSamplerDescriptor<1>>(VK_SHADER_STAGE_FRAGMENT_BIT, std::array<VkImageView, 1>{image}, sampler);
}
template<typename Data>
std::shared_ptr<BufferDescriptor<Data, 1>> uboDescriptor(VkBuffer buffer) {
    return std::make_shared<BufferDescriptor<Data, 1>>(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                                       VK_SHADER_STAGE_COMPUTE_BIT,
                                                       std::array<VkBuffer, 1>{buffer}, 1);
}
template<typename Data>
std::shared_ptr<BufferDescriptor<Data, 1>> storageDescriptor(VkBuffer buffer, int size) {
    return std::make_shared<BufferDescriptor<Data, 1>>(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                       VK_SHADER_STAGE_COMPUTE_BIT,
                                                       std::array<VkBuffer, 1>{buffer}, size);
}

// Generic compute shader material that is instantiated with dispatch dimensions & 
class Compute : public ComputeMaterial<1> {
public:
    Compute(VulkanApp<1>& app, 
            std::vector<std::shared_ptr<Descriptor>> descriptors, 
            glm::vec3* dispatchDimensions,
            const std::vector<char>& shaderCode): ComputeMaterial<1>(app.getDevice(), 
                                                               app.getPhysicalDevice(),
                                                               descriptors,
                                                               shaderCode), dispatchDimensions_(dispatchDimensions){}
    glm::vec3 getDispatchDimensions() override {
        return *dispatchDimensions_;
    }

    void update(uint32_t currentImage, VkExtent2D swapChainExtent) override {
        if (descriptorsDirty) {
            populateDescriptorSet(currentImage);
        }
    }
private: 
    glm::vec3* dispatchDimensions_;
};
// Helper function for creating compute nodes for the render graph
RenderGraph<1>::NodeHandle computeNode(VulkanApp<1>& app,
                                       RenderGraph<1>& graph,
                                       std::vector<std::shared_ptr<Descriptor>> descriptors,
                                       glm::vec3* dispatchDimensions,
                                       const std::vector<char>& shaderCode) {
    auto shader = std::make_unique<Compute>(app, descriptors, dispatchDimensions, shaderCode);
    auto node = std::make_unique<ComputeNode<1>>(std::move(shader), 
                                                 app.getDevice(), 
                                                 app.getComputeQueue(), 
                                                 app.getComputeCommandBuffers());
    return graph.addNode(std::move(node));
}

int main(int argc, char** argv) {
    ///
    // Parse arguments
    std::cout << "Parsing aruments" << std::endl;
    ///
    if (argc < 3 || argc > 4) {
        std::cout << "Usage: vk_train_splats <colmap_dir> <image_dr> <optional_path_to_splat_csv>" << std::endl;
        return EXIT_FAILURE;
    }
    std::string colmapDir = argv[1];
    std::string imageDir = argv[2];
    std::optional<std::string> splatFile;
    if (argc == 4) {
        splatFile = argv[3];
    }

    ///
    // Load image, camera, and splat data
    std::cout << "Loading data" << std::endl;
    ///
    std::cout << "Loading frames" << std::endl;
    CameraImageLoader imageData(std::format("{}/images.txt", colmapDir), std::format("{}/cameras.txt", colmapDir), imageDir);
    std::cout << "Loading splats" << std::endl;
    GaussianSplatModel splatModel{};
    if (splatFile.has_value()) {
        splatModel.initFromSplatCSV(splatFile.value());
    } else {
        splatModel.initFromSparsePointFile(std::format("{}/points3D.txt", colmapDir));
    }
    int image_size = imageData.getImageSize();

    ///
    // Intitialize Vuklan
    std::cout << "Initializing Vulkan" << std::endl;
    ///
    VulkanApp<1> app(1080, 2160); // We are going to render example & output side by side
    app.init();

    ///
    // Set initial learning rates
    // Taken from the paper's implementation
    ///
    LearningRates lr;
    lr.mean = 0.00016;
    lr.scale = 0.0005;
    lr.rotation = 0.0001;
    lr.alpha = 0.0025;
    lr.sh_0 = 0.00025;
    lr.sh_h = 0.00025 / 20.0;

    ///
    // Setup GPU data buffers
    std::cout << "Setting up GPU buffers" << std::endl;
    ///
    splatModel.initializeBuffers(app);
    imageData.initBuffers(app);

    int maxSplatIndex = splatModel.maxSplatIndex();
    int numSplats = maxSplatIndex + 1;
    int maxSplatsPerTile = maxSplatIndex;

    TileManager tileManager(maxSplatsPerTile);
    tileManager.initBuffers(app, maxSplatIndex, image_size);

    TrainingManager trainManager(lr, maxSplatIndex, 0.01 /* near plane for frustum culling*/);
    trainManager.initBuffers(app, imageData.numSamples());

    ///
    // Create image sampler for graphics pipeline
    //
    std::unique_ptr<VulkanSampler> sampler;
    VulkanSampler::createWithAddressMode(sampler,
                                         VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                         app.getDevice(),
                                         app.getPhysicalDevice());

    ///
    // Create descriptors for each buffer / image
    std::cout << "Creating descriptors" << std::endl;
    //
    auto rasterStorage = storageImageDescriptor(imageData.rasterImage());
    auto rasterSampler = sampledImageDescriptor(imageData.rasterImage(), **sampler);
    auto trainingStorage = storageImageDescriptor(imageData.trainingImage());
    auto trainingSampler = sampledImageDescriptor(imageData.trainingImage(), **sampler);
    auto gradImage = storageImageDescriptor(imageData.gradientImage());
    auto cameraProps = uboDescriptor<CameraProps>(imageData.cameraProps());

    auto splatBuffer = storageDescriptor<Splat>(splatModel.splatBuffer(), numSplats);
    auto imagespaceSplats = storageDescriptor<ImagespaceSplat>(splatModel.imagespaceSplats(), numSplats);
    auto rasterGradients = storageDescriptor<RasterGrad>(splatModel.rasterGradients(), numSplats);
    auto finalGradients = storageDescriptor<Splat>(splatModel.finalGradients(), numSplats);
    auto posGradientMags = storageDescriptor<float>(splatModel.positionGradientMagnitudes(), numSplats);
    auto densityFlags = storageDescriptor<glm::vec4>(splatModel.densityFlags(), numSplats);

    auto tileEntries0 = storageDescriptor<SplatEntry>(tileManager.tileEntries0(), maxSplatsPerTile);
    auto tileEntries1 = storageDescriptor<SplatEntry>(tileManager.tileEntries1(), maxSplatsPerTile);
    auto splatCountWrite = storageDescriptor<TileSplatCount>(tileManager.count(), 1);
    auto splatCountRead = uboDescriptor<TileSplatCount>(tileManager.count());
    auto rasterControl = uboDescriptor<RasterControls>(tileManager.control());

    auto preprocessControl = uboDescriptor<PreprocessControls>(trainManager.control());
    auto learningRates = uboDescriptor<LearningRates>(trainManager.learningRates());
    auto densityControl = uboDescriptor<DensityControl>(trainManager.densitySwitch());

    ///
    // Calculate dispatch sizes for our various compute operations
    ///
    glm::vec3 perSplatDispatch{std::ceil(static_cast<float>(numSplats) / 256.0), 1, 1};
    glm::vec3 tileAssignmentDispatch{1, 16, 1};
    // Each tile covers 1/4 of the width/height, and each workgroup has 32x32 threads
    int rasterSize = std::ceil(static_cast<float>(image_size) / (4.f * 32.f));
    glm::vec3 rasterDispatch{rasterSize, rasterSize, 16};

    ///
    // Create a render graph and add our compute stages to it
    std::cout << "Creating render graph" << std::endl;
    //
    ///
    auto renderGraph = std::make_unique<RenderGraph<1>>(app.getDevice());

    // Preprocess splats
    auto preprocess = computeNode(app, *renderGraph, std::vector<std::shared_ptr<Descriptor>>{
        splatBuffer, imagespaceSplats, cameraProps, preprocessControl
    }, &perSplatDispatch, readFile("spirv/preprocessSplats.spv"));

    // Assign splats to tiles
    auto assignment = computeNode(app, *renderGraph, std::vector<std::shared_ptr<Descriptor>>{
        imagespaceSplats, tileEntries0, rasterControl, splatCountWrite
    }, &tileAssignmentDispatch, readFile("spirv/assignSplatsToTile.spv"));

    // Sort by depth
    auto sorting = computeNode(app, *renderGraph, std::vector<std::shared_ptr<Descriptor>>{
        tileEntries0, tileEntries1, splatCountRead, rasterControl
    }, &tileAssignmentDispatch, readFile("spirv/single_radixsort.spv"));

    // Rasterize
    auto rasterize = computeNode(app, *renderGraph, std::vector<std::shared_ptr<Descriptor>>{
        tileEntries1, imagespaceSplats, rasterStorage, gradImage, rasterControl, splatCountRead
    }, &rasterDispatch, readFile("spirv/rasterizeSplats.spv"));

#ifdef BACKPROP
    // Backprop rasterization
    auto rasterizeBackprop = computeNode(app, *renderGraph, std::vector<std::shared_ptr<Descriptor>>{
        tileEntries1, imagespaceSplats, rasterStorage, trainingStorage, gradImage, rasterGradients, rasterControl
    }, &rasterDispatch, readFile("spirv/rasterizeBackwards.spv"));

    // Backprop preprocessing
    auto preprocessBackprop = computeNode(app, *renderGraph, std::vector<std::shared_ptr<Descriptor>>{
        splatBuffer, imagespaceSplats, cameraProps, preprocessControl, rasterGradients, finalGradients, posGradientMags
    }, &perSplatDispatch, readFile("spirv/preprocessBackwards.spv"));

    // Update model
    auto updateModel = computeNode(app, *renderGraph, std::vector<std::shared_ptr<Descriptor>>{
        splatBuffer, finalGradients, learningRates, densityFlags, densityControl, posGradientMags
    }, &perSplatDispatch, readFile("spirv/updateModel.spv"));
#endif
    
    ///
    // Create graphics pipeline nodes
    std::cout << "Creating graphcs pipeline" << std::endl;
    ///
    // Create MVP matrix for rendering
    auto model = glm::identity<glm::mat4>();
    auto view = glm::identity<glm::mat4>();
    auto projection = glm::ortho(-1.0, 1.0, -1.0, 1.0);
    // GLM was originally designed for OpenGL where the Y coordinate of the clip coordinates is inverted
    projection[1][1] *= -1;
    auto ubo = UniformBufferObject::fromModelViewProjection(model, view, projection);
    // Put mvp in a buffer
    std::unique_ptr<Buffer<UniformBufferObject>> uboBuffer;
    VK_SUCCESS_OR_THROW(Buffer<UniformBufferObject>::create(uboBuffer,
                                                            1,
                                                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                                                            app.getDevice(),
                                                            app.getPhysicalDevice()),
                            "Failed to create uniform buffer.");
    // Map our UBO to the buffer
    uboBuffer->mapAndExecute(0, sizeof(UniformBufferObject), [&ubo](void* mappedMemory) {
        memcpy(mappedMemory, &ubo, sizeof(UniformBufferObject));
    });
    // Create descriptor
    auto mvp = std::make_shared<UniformBufferDescriptor<UniformBufferObject, 1>>(std::array<VkBuffer, 1>{uboBuffer->getBuffer()}, VK_SHADER_STAGE_VERTEX_BIT);


    auto renderMaterial = std::make_unique<BasicMaterial<1,2>>(app.getDevice(), app.getPhysicalDevice(),
                                                               std::vector<std::shared_ptr<Descriptor>>{
                                                                mvp,
                                                                trainingSampler, 
                                                                rasterSampler},
                                                               app.getSwapchainExtent(),
                                                               app.getRenderPass(),
                                                               readFile("spirv/vert.spv"),
                                                               readFile("spirv/frag.spv"),
                                                               SimpleVertex::getBindingDescription(),
                                                               SimpleVertex::getAttributeDescriptions());
    auto renderable = std::make_unique<ImageToScreenRenderable<1>>(std::move(renderMaterial), 
                                                                   app.getDevice(),
                                                                   app.getPhysicalDevice(),
                                                                   app.getGraphicsQueue(),
                                                                   app.getCommandPool());
    auto graphics = renderGraph->addNode(std::make_unique<RenderableNode<1>>(std::move(renderable),
                                                                             app.getDevice(),
                                                                             app.getGraphicsQueue(),
                                                                             app.getRenderPass(),
                                                                             app.getGraphicsCommandBuffers()));
    auto acquireImageNode = renderGraph->addNode(std::make_unique<AcquireImageNode<1>>(app.getDevice()));
    auto presentNode = renderGraph->addNode(std::make_unique<PresentNode<1>>(app.getDevice(), app.getPresentQueue()));

    ///
    // Set up edges in our render graph
    ///

    // Set up dependencies for image acquisition -> graphics -> presentation
    renderGraph->addEdge(acquireImageNode, graphics);
    renderGraph->addEdge(graphics, presentNode);

    // Set up forward pass edges
    renderGraph->addEdge(preprocess, assignment);
    renderGraph->addEdge(assignment, sorting);
    renderGraph->addEdge(sorting, rasterize);

#ifdef BACKPROP
    // Set up backward pass edges
    renderGraph->addEdge(rasterize, rasterizeBackprop);
    renderGraph->addEdge(rasterizeBackprop, preprocessBackprop);
    renderGraph->addEdge(preprocessBackprop, updateModel);

    // Render once all compute is complete
    // Ideally, we could render once rasterization is complete,
    // but my render "graph" system really only handles straight lines
    renderGraph->addEdge(updateModel, graphics);
#else
    renderGraph->addEdge(rasterize, graphics);
#endif

    ///
    // Set up a callback at the beggining of each iteration to update things
    ///
    int iteration = 0;
    int currentSample = 0;
    int sh_degree = 0;
    bool done = false;
    bool applyDensityControl = false;
    auto preDrawCallback = std::make_shared<std::function<void(VulkanApp<1>&, uint32_t)>>();
    *preDrawCallback = [&](VulkanApp<1>& a, uint32_t) {
        // Nothing more to do if training is completed
        if (done) {
            return;
        }

        // On the last sample of every 100 iterations, compute density adjustments
        if ((iteration + 1) % DENSITY_ITERATION_GAP == 0 && currentSample == imageData.numSamples() - 1) {
            trainManager.setDensitySwitch(a, true);
            applyDensityControl = true;
        }

        
        // If we have run out of samples, increment the iteration and start again,
        // applying any updates to our training regimen
        if (currentSample >= imageData.numSamples()) {
            currentSample = 0;
            // Update iteration
            iteration++;
            // Maybe update sh degree
            if (iteration % 1000 == 0) {
                sh_degree = std::min(sh_degree + 1, 3);
                trainManager.setSHDegree(a, sh_degree);
            }
            // Maybe update mean learning rate
            if (iteration % 100 == 0) {
                lr.mean *= .25;
                trainManager.setMeanLr(a, lr.mean);
            }
            // Maybe save splats to file
            if (iteration % 100 == 0) {
                splatModel.saveModel(std::format("splats/{}", iteration), a);
            }
            // Stop training after 35K iterations
            if (iteration == 35000) {
                issueSingleTimeCommand([&](VkCommandBuffer cb) {
                    vkCmdFillBuffer(cb, trainManager.learningRates(), 0, sizeof(LearningRates), 0);
                }, a.getGraphicsQueue(), a.getDevice(), a.getCommandPool());
                done = true;
                splatModel.saveModel("splats/final", a);
                return;
            }

            // If we kicked off density control processing for the last iteration, apply those updates
            if (applyDensityControl) {
                // Performs optimization and creates new resized splat buffers
                splatModel.performDensityOptimization(app);
                std::cout << "Density control: Splat count modified from " << numSplats << " to " 
                          << splatModel.maxSplatIndex() + 1 << std::endl;
                // Update our splat count
                maxSplatIndex = splatModel.maxSplatIndex();
                numSplats = maxSplatIndex + 1;
                perSplatDispatch = glm::vec3{std::ceil(static_cast<float>(numSplats) / 256.0), 1, 1};
                // Update our controls
                trainManager.setDensitySwitch(a, false);
                trainManager.setMaxIndex(a,maxSplatIndex);
                tileManager.setMaxIndex(a, maxSplatIndex);
                // Update all descriptors that relied on the resized buffers
                splatBuffer->bindBuffer(splatModel.splatBuffer(), numSplats);
                imagespaceSplats->bindBuffer(splatModel.imagespaceSplats(), numSplats);
                rasterGradients->bindBuffer(splatModel.rasterGradients(), numSplats);
                finalGradients->bindBuffer(splatModel.finalGradients(), numSplats);
                posGradientMags->bindBuffer(splatModel.positionGradientMagnitudes(), numSplats);
                densityFlags->bindBuffer(splatModel.densityFlags(), numSplats);
                // Update control flags
                applyDensityControl = false;
                descriptorsDirty = true;
            }
        } else if (descriptorsDirty) {
            // If we aren't in the branch executed at the beggining of an iteration, then our descriptors
            // should no longer be dirty
            descriptorsDirty = false;
        }

        // Zero out the gradients from the previous iteration
        issueSingleTimeCommand([&](VkCommandBuffer cb) {
            vkCmdFillBuffer(cb, splatModel.finalGradients(), 0, sizeof(Splat) * numSplats, 0);
        }, a.getGraphicsQueue(), a.getDevice(), a.getCommandPool());

        // Upload the next training sample
    #ifdef DEBUG_FIRST
        if (iteration == 0 && currentSample == 0) {
    #endif
        std::cout << "Iteration " << iteration << ": ";
        imageData.uploadTrainingSample(currentSample, a);
    #ifdef DEBUG_FIRST
        }
    #endif

        currentSample++;
    };
    app.addPreDrawCallback(preDrawCallback);

    ///
    // Run the training we've set up
    ///
    std::cout << "Running" << std::endl;
    app.setRenderGraph(std::move(renderGraph));
    app.run();

    return EXIT_SUCCESS;
}