// grad_test's MainInputs: the checks are grad_check.h's (shared with the web's checker,
// web-check/grad_test_check.cpp)
#include "grad_test.zgal.h"

namespace render_types = grad_test;
#include "grad_check.h"

#include <cstring>
#include <iostream>

Result<grad_test::MainInputs> grad_test::mainInputs(VulkanContext&) {
    return MainInputs{
        .check = grad_check::check,
        .sameGradients =
            [](const zgal::Data<ProjectedGradient>& a, const zgal::Data<ProjectedGradient>& b) {
                return a.size() == b.size() &&
                       std::memcmp(a.elements().data(), b.elements().data(), a.size() * sizeof(ProjectedGradient)) == 0;
            },
        .fail = [] {
            std::cout.flush();
            std::exit(1);
        },
    };
}
