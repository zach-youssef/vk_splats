#version 450

// Zach Youssef, 4/22/26
// Simple vertex shader for displaying a 2D image

layout (binding = 0) uniform UBO {
    mat4 mvp;
} ubo;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inTexCoord;

layout(location = 0) out vec2 fragTexCoord;


void main() {
    gl_Position = ubo.mvp * vec4(inPosition, 0.0, 1.0);
    fragTexCoord = inTexCoord;
}