// densify_test's MainInputs: the failure exit (every check is in the app)
#include "densify_test.zgal.h"

#include <cstdlib>
#include <iostream>

Result<densify_test::MainInputs> densify_test::mainInputs(VulkanContext&) {
    return MainInputs{
        .fail = [] {
            std::cout.flush();
            std::exit(1);
        },
    };
}
