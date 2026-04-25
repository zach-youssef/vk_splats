# version 450

// Zach Youssef, 4/22/26
// Fragment shader for displaying training images next to model results

layout(location = 0) in vec2 fragTexCoord;

layout(binding = 1) uniform sampler2D trainingSampler;
layout(binding = 2) uniform sampler2D rasterSampler;

layout(location = 0) out vec4 outColor;

void main() {
    if (fragTexCoord.x > 0.5) {
        outColor = texture(rasterSampler, vec2((fragTexCoord.x * 2.), fragTexCoord.y));
    } else {
        outColor = texture(trainingSampler, vec2((fragTexCoord.x * 2.), fragTexCoord.y));
    }
}
