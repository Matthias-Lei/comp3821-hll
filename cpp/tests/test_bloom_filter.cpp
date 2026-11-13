#include "bloom_filter.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace {

constexpr std::size_t kBits = 8192;
constexpr std::uint64_t kSeed = 1;

BloomFilter build(std::span<const std::uint64_t> values, std::size_t m = kBits)
{
        return sketch_test::build_sketch<BloomFilter>(values, m, kSeed);
}

// If we pass 0..99 to update(), the double hashing sets exactly 100 bits.
// The estimate must equal -(m/k) ln((m - 100) / m) exactly.
void test_formula(sketch_test::Context& context)
{
        BloomFilter sketch(kBits);
        for (std::uint64_t h = 0; h < 100; ++h)
                sketch.update(h);

        // m = 8192, k = 3, set_bits = 100
        const double expected = -(8192.0 / 3.0) * std::log((8192.0 - 100.0) / 8192.0);
        const std::optional<double> estimate = sketch.estimate();
        context.expect(estimate.has_value() && std::abs(*estimate - expected) < 1e-9,
            "estimate matches -m/k ln(V)");
}

void test_exact_on_small_streams(sketch_test::Context& context)
{
        const std::vector<std::uint64_t> empty;
        context.expect(build(empty).estimate() == 0.0, "empty stream estimates 0");

        const std::vector<std::vector<std::uint64_t>> streams = {
                { 10, 20, 10, 30, 20 },
                sketch_test::range(0, 20),
                { 7, 7, 7, 7 },
        };
        for (const std::vector<std::uint64_t>& stream : streams) {
                const std::uint64_t truth = sketch_test::distinct_count(stream);
                const std::optional<double> estimate = build(stream).estimate();
                context.expect(estimate.has_value()
                        && std::llround(*estimate) == static_cast<long long>(truth),
                    "small stream rounds to the exact distinct count");
        }
}

void test_repeats_do_not_change_state(sketch_test::Context& context)
{
        const std::vector<std::uint64_t> first { 1, 2, 3, 4, 5 };
        const std::vector<std::uint64_t> repeated { 1, 1, 2, 3, 3, 3, 4, 5, 5 };
        context.expect(build(first).estimate() == build(repeated).estimate(),
            "repeats leave the estimate unchanged");
}

void test_reordering_does_not_change_state(sketch_test::Context& context)
{
        std::vector<std::uint64_t> values = sketch_test::range(0, 1000);
        const std::optional<double> forward = build(values).estimate();
        std::reverse(values.begin(), values.end());
        context.expect(forward == build(values).estimate(),
            "reordering leaves the estimate unchanged");
}

// m = 100 and m = 64 will saturate very quickly with k=3 hashes per element.
void test_full_bitmap_has_no_estimate(sketch_test::Context& context)
{
        const std::vector<std::uint64_t> values = sketch_test::range(0, 10000);
        const std::size_t sizes[] = { 64, 100 };
        for (std::size_t m : sizes) {
                context.expect(!build(values, m).estimate().has_value(),
                    "full bitmap gives nullopt");
        }
}

void test_memory_bits(sketch_test::Context& context)
{
        context.expect(BloomFilter(kBits).memory_bits() == kBits, "memory_bits is m");
}

} // namespace

int main(int argc, char** argv)
{
        std::vector<sketch_test::TestGroup> groups;

        groups.emplace_back("Bloom Filter")
            .add("formula", test_formula)
            .add("small streams", test_exact_on_small_streams)
            .add("repeats", test_repeats_do_not_change_state)
            .add("reordering", test_reordering_does_not_change_state)
            .add("full bitmap", test_full_bitmap_has_no_estimate)
            .add("memory bits", test_memory_bits);

        return sketch_test::run_all(
            argc, argv, std::span<const sketch_test::TestGroup>(groups));
}
