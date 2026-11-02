// ssim_test's MainInputs: the failure exit (every check is in the app)
#include "ssim_test.zgal.h"

#include <cstdlib>
#include <iostream>

Result<ssim_test::MainInputs> ssim_test::mainInputs(VulkanContext&) {
    return MainInputs{
        .fail = [] {
            std::cout.flush();
            std::exit(1);
        },
    };
}
