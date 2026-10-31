// render_test's MainInputs: the checks are render_check.h's (shared with the web's checker,
// web-check/render_test_check.cpp)
#include "render_test.zgal.h"

namespace render_types = render_test;
#include "render_check.h"

#include <cstring>
#include <iostream>

namespace {

template<typename T>
bool same(const zgal::Data<T>& a, const zgal::Data<T>& b) {
    return a.size() == b.size() && std::memcmp(a.elements().data(), b.elements().data(), a.size() * sizeof(T)) == 0;
}

}  // namespace

Result<render_test::MainInputs> render_test::mainInputs(VulkanContext&) {
    return MainInputs{
        .check = render_check::check,
        .sameColours = same<glm::vec4>,
        .sameEntries = same<uint32_t>,
        .fail = [] {
            std::cout.flush();
            std::exit(1);
        },
    };
}
