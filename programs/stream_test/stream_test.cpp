// stream_test's MainInputs: the failure exit (every check is in the app)
#include "stream_test.zgal.h"

#include <cstdlib>
#include <iostream>

Result<stream_test::MainInputs> stream_test::mainInputs(VulkanContext&) {
    return MainInputs{
        .fail = [] {
            std::cout.flush();
            std::exit(1);
        },
    };
}
