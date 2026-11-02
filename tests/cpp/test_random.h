// Seeded random numbers that are the same on every platform, for the C++ tests. std::mt19937's
// output is fixed by the standard, but std::uniform_real_distribution, normal_distribution and
// uniform_int_distribution aren't: libstdc++ (Linux CI) and libc++ (macOS) turn the same engine
// into different numbers, so a seeded test would check other cases on each
// (docs/building-and-testing.md). These map the engine's 32-bit words with fixed formulas instead.
#pragma once

#include <cmath>
#include <cstdint>
#include <random>

class TestRandom {
public:
    explicit TestRandom(uint32_t seed) : engine_(seed) {}

    // Uniform in [lo, hi), from 53 bits of two words
    double uniform(double lo, double hi) {
        const uint64_t a = engine_() >> 5, b = engine_() >> 6;  // 27 and 26 bits
        return lo + (hi - lo) * (double(a * 67108864ull + b) / 9007199254740992.0);
    }

    // Standard normal (Box-Muller; one of the pair, so each call takes four words)
    double normal() {
        const double u1 = 1.0 - uniform(0.0, 1.0);  // (0, 1], so the log is finite
        const double u2 = uniform(0.0, 1.0);
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }

    // Uniform in [lo, hi], inclusive (the modulo's bias is below 3e-6 for the tests' ranges: at most
    // 10241 values of 2^32)
    int integer(int lo, int hi) { return lo + int(engine_() % uint32_t(hi - lo + 1)); }

private:
    std::mt19937 engine_;
};
