#include "hash.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>

int main()
{
        constexpr auto max_u64 = std::numeric_limits<std::uint64_t>::max();
        struct Vector {
                std::uint64_t input;
                std::uint64_t seed;
                std::uint64_t expected;
        };
        // Expected values are XXH64 of the 8-byte little-endian input, with the given seed.
        // They were computed with the Python xxhash package (4.0.1). That package gives
        // 0xef46db3751d8e999 for the empty string with seed 0, the published XXH64 value.
        constexpr std::array vectors {
                Vector {       0,       0, 0x34c96acdcadb1bbbULL },
                Vector {       0,       1, 0x22c76afd15f0110fULL },
                Vector {       1,       1, 0x8afb45d6a8b39709ULL },
                Vector {      42,       1, 0x602f275014cd5d53ULL },
                Vector {      42,      64, 0x55ddc6c48d5ad7cdULL },
                Vector { max_u64,       0, 0x85d136adb773c6c9ULL },
                Vector { max_u64,       1, 0x0ad7f61289875125ULL },
                Vector {       0, max_u64, 0x60b899029d747159ULL },
                Vector { max_u64, max_u64, 0x1a158c94abf6a8b1ULL },
        };
        for (const auto& v : vectors) {
                if (hash64(v.input, v.seed) != v.expected) {
                        std::cerr << "hash output mismatch: input=" << v.input << " seed=" << v.seed
                                  << '\n';
                        return 1;
                }
        }

        std::array<std::size_t, 1024> buckets { };
        for (std::uint64_t key = 0; key < 1'000'000; ++key)
                ++buckets[hash64(key, 1) & 1023];
        constexpr double expected = 1'000'000.0 / 1024;
        double chi_square = 0;
        for (auto count : buckets) {
                const double difference = static_cast<double>(count) - expected;
                chi_square += difference * difference / expected;
        }

        // Pearson chi-square on the 1024 bucket counts (1023 degrees of freedom, nothing fitted).
        // https://www.itl.nist.gov/div898/handbook/eda/section3/eda35f.htm
        //
        // Both tails count as failures at alpha = 0.01. Too high means the buckets are uneven.
        // Too low means they are suspiciously even.
        //
        // The bounds are the 0.005 and 0.995 quantiles for 1023 degrees of freedom. They were
        // computed by inverting the chi-square CDF numerically. scipy.stats.chi2.ppf agrees with
        // them to within 2e-5. cpp/tests/verify_chi2_bounds.py repeats that check.
        //
        // The seed is fixed, so the result is the same on every run. If this fails, the hash
        // changed.
        constexpr double lower_critical = 910.2463;
        constexpr double upper_critical = 1143.2653;
        std::cout << "bucket chi-square=" << chi_square << " (valid range: [" << lower_critical
                  << ", " << upper_critical << "])\n";
        if (chi_square <= lower_critical || chi_square >= upper_critical) {
                std::cerr << "fixed-seed bucket uniformity check failed (two-sided p <= 0.01)\n";
                return 1;
        }
        return 0;
}
