// Zach Youssef, 4/20/26
// Header defining class abstration for our camera & image data

#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include <VulkanApp.h>
#include <Image.h>

#include "GpuTypes.h"
#include "ColmapImportUtils.h"
#include <vector>
#include <glm/gtx/quaternion.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/viz/types.hpp>

class CameraImageLoader {
public:
    CameraImageLoader(const std::string& imageFile, const std::string& cameraFile, const std::string& imageDirectory) {
        // Read list of camera positions
        cameraFrames = readColmapFile<CameraPosition>(imageFile, [this, &imageDirectory](const std::vector<std::string>& line){
            // Parse rotation
            glm::quat q;
            for (int i = 0; i < 4; ++i) {
                q[i] = std::stof(line[i + 1]);
            }
            glm::mat4 rotation = glm::toMat4(q);
            glm::mat4 view = glm::inverse(rotation);

            // Parse translation
            glm::vec3 t;
            for (int i = 0; i < 3; ++i) {
                t[i] = std::stof(line[i + 5]);
            }

            /////////
            // It should be:
            view[3] = {t[0], -t[1], -t[2], 1};
            // But according to the XCode debugger, I need to:
            //view[0][3] = t[0];
            //view[1][3] = -t[1];
            //view[2][3] = -t[2];
            /////////

            imageNames.push_back(imageDirectory + "/" + line[9]);
            return CameraPosition{
                view, 
                static_cast<int32_t>(imageNames.size() - 1)
            };
        });

        // Read the properties of the camera itself
        intrinsics = readColmapFile<CameraIntrinsics>(cameraFile, [](const std::vector<std::string>& line){
            int image_size = std::stoi(line[2]);
            float focal_length = std::stof(line[4]);
            float distortion = std::stof(line[7]);
            // Using OpenCV to construct the camera projection
            // First use the params to build a calibration matrix
            cv::Mat calibrationMatrix(3, 3, CV_32FC1);
            calibrationMatrix = 0;
            calibrationMatrix.at<float32_t>(0, 0) = focal_length;
            calibrationMatrix.at<float32_t>(1, 1) = focal_length;
            calibrationMatrix.at<float32_t>(0, 2) = image_size / 2;
            calibrationMatrix.at<float32_t>(1, 2) = image_size / 2;
            calibrationMatrix.at<float32_t>(2, 2) = 1;
            // Adjust for distortion
            cv::Size2i frameSize{image_size, image_size};
            cv::Mat cameraMatrix = cv::getOptimalNewCameraMatrix(calibrationMatrix, cv::Scalar(distortion), frameSize, 1);
            // Construct camera projection matrix
            cv::Matx33d K;
            cameraMatrix.convertTo(K, CV_64F);
            cv::viz::Camera cam(K, frameSize);
            cv::Matx44d cameraProjection;
            cam.computeProjectionMatrix(cameraProjection);
            // Convert to glm matrix (glm/vulkan use column major indexing)
            glm::mat4 projection;
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    projection[col][row] = cameraProjection(row, col);
                }
            }
            // GLM/OpenGL have the opposite Y orientation to Vulkan
            projection[1][1] *= -1;

            return CameraIntrinsics{
                projection,
                image_size,
                focal_length
            };
        })[0];
    }

    // Returns the number of training frames loaded
    size_t numSamples() {
        return cameraFrames.size();
    }

    // Returns the size of the image
    int getImageSize() {
        return intrinsics.image_size;
    }

    // Intitializes the buffers needed for rasterization / training
    void initBuffers(VulkanApp<1>& app) {
        Image::createEmptyRGBA(trainingImage_, 
                               intrinsics.image_size, 
                               intrinsics.image_size,
                               app.getGraphicsQueue(),
                               app.getCommandPool(),
                               app.getDevice(),
                               app.getPhysicalDevice());
        Image::createEmptyRGBA(rasterImage_, 
                               intrinsics.image_size, 
                               intrinsics.image_size,
                               app.getGraphicsQueue(),
                               app.getCommandPool(),
                               app.getDevice(),
                               app.getPhysicalDevice());
        Image::createEmpty(gradImage_, 
                           VK_FORMAT_R16G16_SFLOAT,
                           intrinsics.image_size, 
                           intrinsics.image_size,
                           app.getGraphicsQueue(),
                           app.getCommandPool(),
                           app.getDevice(),
                           app.getPhysicalDevice());
        Buffer<CameraProps>::create(propBuffer_,
                                    1,
                                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                    app.getDevice(),
                                    app.getPhysicalDevice());
    }

    // Uploads the image & camera props for the training sample at the given index
    void uploadTrainingSample(int idx, VulkanApp<1>& app) {
        std::cout << "Uploading sample " << idx << std::endl;
        // Load the image for this frame
        int width,height,channels;
        stbi_uc* pixels = stbi_load(imageNames[idx].c_str(), &width, &height, &channels, STBI_rgb_alpha);
        if (!pixels) {
            std::cout << "Image didn't load (" << imageNames[idx].c_str() << ")" << std::endl;
            std::cout << stbi_failure_reason() << std::endl;
        }
        trainingImage_->uploadUcharBufferToImage(pixels, 
                                                 width * height * STBI_rgb_alpha,
                                                 app.getGraphicsQueue(),
                                                 app.getCommandPool(),
                                                 app.getPhysicalDevice());
        stbi_image_free(pixels);

        // Load the camera props for this frame
        propBuffer_->mapAndExecute(0, sizeof(CameraProps), [this, idx](void* mappedBuffer) {
            CameraProps* props = (CameraProps*) mappedBuffer;
            props->view = cameraFrames[idx].view;
            props->projection = intrinsics.projection;
            props->image_size = intrinsics.image_size;
            props->focal_length = intrinsics.focal_length;
            props->tan_fov = (intrinsics.image_size)/ (2 * intrinsics.focal_length);
        });
    }

    VkImageView trainingImage() {
        return trainingImage_->getImageView();
    }

    VkImageView rasterImage() {
        return rasterImage_->getImageView();
    }
 
    VkImageView gradientImage() {
        return gradImage_->getImageView();
    }

    VkBuffer cameraProps() {
        return propBuffer_->getBuffer();
    }

private:
    std::vector<CameraPosition> cameraFrames;
    CameraIntrinsics intrinsics;
    std::vector<std::string> imageNames;

    // Images needed for rasterization & traning
    std::unique_ptr<Image> trainingImage_;
    std::unique_ptr<Image> rasterImage_;
    std::unique_ptr<Image> gradImage_;

    // Uniform buffer for camera props
    std::unique_ptr<Buffer<CameraProps>> propBuffer_;
};