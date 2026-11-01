// grad_test_check <readbacks dir>: grad_test's checks (../grad_check.h) on the frame a web run of
// grad_test wrote (grad_test.inputs.ts, with GRAD_TEST_READBACKS=<dir>): the same checks the
// native program runs in-process, so the web's backward pass is held to the closed-form oracle as
// the native one is. It prints what the native program prints (the hash too, which test.sh
// compares between runs) and exits 1 on a mismatch.
//
// The directory holds each readback's elements as std430 bytes, <name>.bin (splats, cameras,
// projected, gradients, losses), and check.txt: the camera index, the background's r, g and b, the
// target's r, g and b, and 1 or 0 for --l1 and for --oracle, separated by spaces.
#include "grad_test_check.zgal.h"

namespace render_types = grad_test_check;
using namespace grad_test_check;
#include "../grad_check.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string& what) {
    std::cout << "grad_test_check: " << what << std::endl;
    std::exit(1);
}

template<typename T>
zgal::Data<T> read(const std::filesystem::path& dir, const std::string& name) {
    const std::filesystem::path path = dir / (name + ".bin");
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("can't read " + path.string());
    const std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (bytes.size() % sizeof(T) != 0) {
        fail(path.string() + " has " + std::to_string(bytes.size()) + " bytes, not a whole number of " +
             std::to_string(sizeof(T)) + "-byte elements");
    }
    std::vector<T> elements(bytes.size() / sizeof(T));
    std::memcpy(elements.data(), bytes.data(), bytes.size());
    return zgal::Data<T>(std::move(elements));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) fail("usage: grad_test_check <readbacks dir>");
    const std::filesystem::path dir = argv[1];
    std::ifstream parameters(dir / "check.txt");
    uint32_t cameraIndex = 0, l1 = 2, oracle = 2;
    glm::vec3 background{}, target{};
    if (!(parameters >> cameraIndex >> background.x >> background.y >> background.z >> target.x >> target.y >>
          target.z >> l1 >> oracle) ||
        l1 > 1 || oracle > 1) {
        fail("can't read " + (dir / "check.txt").string());
    }
    const bool complete = grad_check::check(read<Splat>(dir, "splats"), read<Camera>(dir, "cameras"),
                                            read<Projected>(dir, "projected"), read<ProjectedGradient>(dir, "gradients"),
                                            read<float>(dir, "losses"), cameraIndex, background, target, l1 == 1,
                                            oracle == 1);
    // The web run writes a frame only once check() would take it
    if (!complete) fail("the readbacks are of different frames or sizes");
    return 0;
}
