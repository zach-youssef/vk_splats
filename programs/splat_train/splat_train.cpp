// splat_train's MainInputs: the PLY export and the failure exit. An extern can't take a
// path option, so the PLY's path comes from $SPLAT_TRAIN_PLY (splat_train.ply by default).
#include "splat_train.zgal.h"

#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace {

// The INRIA PLY's vertex properties, in its order (tools/splatdata/splats.py's
// ply_property_names): x y z nx ny nz f_dc_0-2 f_rest_0-44 opacity scale_0-2 rot_0-3
constexpr int kProperties = 62;

std::string header(size_t count) {
    std::string h = "ply\nformat binary_little_endian 1.0\nelement vertex " + std::to_string(count) + "\n";
    const char* names[] = {"x", "y", "z", "nx", "ny", "nz"};
    for (const char* name : names) h += std::string("property float ") + name + "\n";
    for (int c = 0; c < 3; ++c) h += "property float f_dc_" + std::to_string(c) + "\n";
    for (int k = 0; k < 45; ++k) h += "property float f_rest_" + std::to_string(k) + "\n";
    h += "property float opacity\n";
    for (int k = 0; k < 3; ++k) h += "property float scale_" + std::to_string(k) + "\n";
    for (int k = 0; k < 4; ++k) h += "property float rot_" + std::to_string(k) + "\n";
    return h + "end_header\n";
}

// A splat's properties: the SH's coefficient k of channel c is sh[3k + c] (Splat's layout), and
// f_rest is channel-major (coefficient k >= 1 of channel c at f_rest_{15c + k - 1}); the rotation
// is (w, x, y, z), stored in (.x, .y, .z, .w)
std::array<float, kProperties> properties(const splat_train::Splat& s) {
    std::array<float, kProperties> p{};
    p[0] = s.position.x;
    p[1] = s.position.y;
    p[2] = s.position.z;
    for (int c = 0; c < 3; ++c) p[6 + c] = s.sh[c];
    for (int c = 0; c < 3; ++c)
        for (int k = 1; k < 16; ++k) p[9 + 15 * c + k - 1] = s.sh[3 * k + c];
    p[54] = s.opacity;
    p[55] = s.logScale.x;
    p[56] = s.logScale.y;
    p[57] = s.logScale.z;
    p[58] = s.rotation.x;
    p[59] = s.rotation.y;
    p[60] = s.rotation.z;
    p[61] = s.rotation.w;
    return p;
}

}  // namespace

Result<splat_train::MainInputs> splat_train::mainInputs(VulkanContext&) {
    return MainInputs{
        .exportPly =
            [](const zgal::Data<Splat>& splats) {
                const char* env = std::getenv("SPLAT_TRAIN_PLY");
                std::string path = env && *env ? env : "splat_train.ply";
                std::ofstream out(path, std::ios::binary);
                out << header(splats.size());
                for (const Splat& s : splats) {
                    auto p = properties(s);  // little-endian floats: every platform the wrapper runs on
                    out.write(reinterpret_cast<const char*>(p.data()), sizeof(p));
                }
                if (!out) {
                    std::cout << "splat_train: FAILED, couldn't write " << path << std::endl;
                    std::exit(1);
                }
                std::cout << "splat_train: wrote " << splats.size() << " splats to " << path << std::endl;
            },
        .fail = [] {
            std::cout.flush();
            std::exit(1);
        },
    };
}
