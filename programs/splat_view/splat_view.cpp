// splat_view's MainInputs: the failure exit (every check is in the app)
#include "splat_view.zgal.h"

#include <cstdlib>
#include <iostream>

Result<splat_view::MainInputs> splat_view::mainInputs(VulkanContext&) {
    return MainInputs{
        .fail = [] {
            std::cout.flush();
            std::exit(1);
        },
    };
}
