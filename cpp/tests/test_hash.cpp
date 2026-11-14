#include "test_helpers.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace {

struct Vector {
        std::uint64_t input;
        std::uint64_t seed;
        std::uint64_t expected;
};

void test_known_vectors(sketch_test::Context& context)
{
        constexpr auto max_u64 = std::numeric_limits<std::uint64_t>::max();
        // Expected values are XXH64 of the 8-byte little-endian input, with the given seed.
        // They were computed with the Python xxhash package (4.0.1). That package gives
        // 0xef46db3751d8e999 for the empty string with seed 0, the published XXH64 value.
        constexpr std::array vectors {
                Vector { 0, 0, 0x34c96acdcadb1bbbULL },
                Vector { 0, 1, 0x22c76afd15f0110fULL },
                Vector { 1, 1, 0x8afb45d6a8b39709ULL },
                Vector { 42, 1, 0x602f275014cd5d53ULL },
                Vector { 42, 64, 0x55ddc6c48d5ad7cdULL },
                Vector { max_u64, 0, 0x85d136adb773c6c9ULL },
                Vector { max_u64, 1, 0x0ad7f61289875125ULL },
                Vector { 0, max_u64, 0x60b899029d747159ULL },
                Vector { max_u64, max_u64, 0x1a158c94abf6a8b1ULL },
        };

        for (const Vector& vector : vectors) {
                const std::uint64_t actual = hash64(vector.input, vector.seed);
                if (actual != vector.expected) {
                        std::cerr << "  input=" << vector.input << " seed=" << vector.seed
                                  << " actual=0x" << std::hex << actual << " expected=0x"
                                  << vector.expected << std::dec << '\n';
                }
                context.expect(actual == vector.expected,
                    "hash output matches the known vector");
        }
}

void test_bucket_uniformity(sketch_test::Context& context)
{
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

        context.note("bucket chi-square=" + std::to_string(chi_square)
            + " (valid range: [" + std::to_string(lower_critical) + ", "
            + std::to_string(upper_critical) + "])");
        context.expect(chi_square > lower_critical && chi_square < upper_critical,
            "fixed-seed bucket uniformity check (two-sided p > 0.01)");
}

} // namespace

int main(int argc, char** argv)
{
        std::vector<sketch_test::TestGroup> groups;

        groups.emplace_back("Hash")
            .add("known vectors", test_known_vectors)
            .add("bucket uniformity", test_bucket_uniformity);

        return sketch_test::run_all(
            argc, argv, std::span<const sketch_test::TestGroup>(groups));
}
