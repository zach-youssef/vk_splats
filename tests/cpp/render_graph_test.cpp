// lib/render_graph.zgal's host-side functions, as ZGAL compiles them to C++: the dispatch sizing
// (groupsFor, which keeps every dispatch within the devices' 65535 workgroups per dimension; the
// kernels' grid-stride loops cover the rest, which tile_test runs on the GPU with --max-groups)
// and the entries' capacity growth (grownCapacity).
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include "render_graph_test.zgal.h"

#include <cstdint>

namespace rg = render_graph_test;

TEST_CASE("groupsFor: enough groups to cover the elements, at most maxGroups") {
    CHECK(rg::groupLimit == 65535u);
    CHECK(rg::groupsFor(0, 256, rg::groupLimit) == 0);
    CHECK(rg::groupsFor(1, 256, rg::groupLimit) == 1);
    CHECK(rg::groupsFor(256, 256, rg::groupLimit) == 1);
    CHECK(rg::groupsFor(257, 256, rg::groupLimit) == 2);
    // The most one dimension holds uncapped, then capped: entryCapacityMax needs 131072 groups
    CHECK(rg::groupsFor(65535u * 256u, 256, rg::groupLimit) == 65535u);
    CHECK(rg::groupsFor(65535u * 256u + 1, 256, rg::groupLimit) == 65535u);
    CHECK(rg::groupsFor(rg::entryCapacityMax, rg::entryGroupSize, rg::groupLimit) == 65535u);
    // The preprocess's 64-invocation groups over 2^24 splats
    CHECK(rg::groupsFor(1u << 24, rg::splatGroupSize, rg::groupLimit) == 65535u);
    // A test's small cap, and a cap of 0 taken as 1
    CHECK(rg::groupsFor(100000, 256, 3) == 3);
    CHECK(rg::groupsFor(100000, 256, 0) == 1);
    for (uint32_t count : {1u, 255u, 1000u, 1u << 20, 1u << 22, rg::entryCapacityMax}) {
        for (uint32_t cap : {1u, 3u, rg::groupLimit}) {
            const uint32_t groups = rg::groupsFor(count, 256, cap);
            CHECK(groups >= 1);
            CHECK(groups <= cap);
            // Uncapped, the groups cover the count exactly
            if (groups < cap) CHECK((uint64_t(groups) * 256 >= count && uint64_t(groups - 1) * 256 < count));
        }
    }
}

TEST_CASE("grownCapacity: unchanged when the total fits, else total + 25%, at most entryCapacityMax") {
    CHECK(rg::grownCapacity(0, 0) == 1);
    CHECK(rg::grownCapacity(16, 0) == 16);
    CHECK(rg::grownCapacity(16, 16) == 16);
    CHECK(rg::grownCapacity(16, 17) == 21);
    CHECK(rg::grownCapacity(1000, 400) == 1000);  // never shrinks
    CHECK(rg::grownCapacity(16, 1000000) == 1250000);
    // Saturates: frames needing more stay incomplete
    CHECK(rg::grownCapacity(16, rg::entryCapacityMax - 1) == rg::entryCapacityMax);
    CHECK(rg::grownCapacity(rg::entryCapacityMax, 0xFFFFFFFFu) == rg::entryCapacityMax);
    for (uint32_t total : {1u, 17u, 4096u, 1u << 24, (1u << 25) - (1u << 23) - 1}) {
        const uint32_t grown = rg::grownCapacity(1, total);
        CHECK(grown >= total);
        CHECK(grown <= rg::entryCapacityMax);
    }
}
