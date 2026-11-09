// splat_view's MainInputs: the failure exit (every check is in the app), and --inspect's accum
// written to $SPLAT_VIEW_ACCUM for the closed-form oracle (web-check/view_oracle_check.cpp)
#include "splat_view.zgal.h"

#include <cstdlib>
#include <fstream>
#include <iostream>

Result<splat_view::MainInputs> splat_view::mainInputs(VulkanContext&) {
    return MainInputs{
        .fail = [] {
            std::cout.flush();
            std::exit(1);
        },
        .saveAccum = [](const zgal::Data<glm::uvec2>& accum) {
            const char* path = std::getenv("SPLAT_VIEW_ACCUM");
            if (path == nullptr || *path == '\0') return;
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(accum.elements().data()),
                      std::streamsize(accum.size() * sizeof(glm::uvec2)));
            if (!out) {
                std::cout << "splat_view: FAILED, can't write accum to " << path << std::endl;
                std::exit(1);
            }
            std::cout << "splat_view: wrote accum to " << path << std::endl;
        },
    };
}
