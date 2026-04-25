// Zach Youssef, 4/22/2026
// Renderable class for drawing images to screen, reused from my extensions for projects 1 & 4

#pragma once

#include <Renderable.h>
#include <VkUtil.h>

struct SimpleVertex {
    glm::vec2 pos;
    glm::vec2 texCoord;

public:
    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(SimpleVertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
        
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(SimpleVertex, pos);
        
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(SimpleVertex, texCoord);

        return attributeDescriptions;
    }
};

template<uint MAX_FRAMES>
class ImageToScreenRenderable : public MeshRenderable<SimpleVertex, MAX_FRAMES> {
public:
    ImageToScreenRenderable<MAX_FRAMES>(
        std::unique_ptr<Material<MAX_FRAMES>>&& material,
        VkDevice device, 
        VkPhysicalDevice physicalDevice, 
        VkQueue graphicsQueue, 
        VkCommandPool commandPool): MeshRenderable<SimpleVertex, MAX_FRAMES>(
            std::vector<SimpleVertex>{
                SimpleVertex{glm::vec2{-1.0f, -1.0f}, glm::vec2{0.0f, 1.0f}},
                SimpleVertex{glm::vec2{1.0f, -1.0f}, glm::vec2{1.0f, 1.0f}},
                SimpleVertex{glm::vec2{1.0f, 1.0f}, glm::vec2{1.0f, 0.0f}},
                SimpleVertex{glm::vec2{-1.0f, 1.0f}, glm::vec2{0.0f, 0.0f}}
            }, 
            std::vector<uint16_t>{
                0, 1, 2,
                2, 3, 0
            }, 
            std::move(material), 
            device, 
            physicalDevice, 
            graphicsQueue, 
            commandPool) {}
};