#include "bloom_filter.hpp"
#include "hash.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kBits = 8192;
constexpr std::uint64_t kSeed = 1;

BloomFilter build(std::span<const std::uint64_t> values, std::size_t m = kBits)
{
        return sketch_test::build_sketch<BloomFilter>(values, m, kSeed);
}

void test_bias(sketch_test::Context& context)
{
        const std::vector<std::uint64_t> values = sketch_test::range(0, 1000);
        sketch_test::expect_no_bias<BloomFilter>(context, values, kBits, 1000.0);
}

void test_different_sketches_are_not_equal(sketch_test::Context& context)
{
        const BloomFilter small(64);
        const BloomFilter large(128);
        context.expect(!(small == large), "different m compare unequal");

        BloomFilter one(64);
        BloomFilter two(64);
        one.update(1);
        two.update(2);
        context.expect(!(one == two), "different bitmaps compare unequal");

        BloomFilter one_copy(64);
        one_copy.update(1);
        context.expect(one == one_copy, "identical bitmaps compare equal");
}

void test_merge_equals_union(sketch_test::Context& context)
{
        const std::vector<std::uint64_t> left = sketch_test::range(0, 500);
        const std::vector<std::uint64_t> right = sketch_test::range(500, 1000);
        const std::vector<std::uint64_t> all = sketch_test::range(0, 1000);

        BloomFilter merged = build(left);
        const BloomFilter right_sketch = build(right);
        const BloomFilter right_copy = right_sketch;
        merged.merge(right_sketch);

        const BloomFilter direct = build(all);
        context.expect(merged == direct, "disjoint merge equals the union sketch");
        context.expect(merged.estimate() == direct.estimate(),
            "merge estimates the same as the union");
        context.expect(right_sketch == right_copy, "merge leaves the source unchanged");
}

void test_overlapping_merge(sketch_test::Context& context)
{
        const std::vector<std::uint64_t> left = sketch_test::range(0, 600);
        const std::vector<std::uint64_t> right = sketch_test::range(400, 1000);
        const std::vector<std::uint64_t> all = sketch_test::range(0, 1000);

        BloomFilter merged = build(left);
        merged.merge(build(right));
        context.expect(merged == build(all), "overlapping merge equals the union sketch");
}

void test_merge_self(sketch_test::Context& context)
{
        const std::vector<std::uint64_t> values = sketch_test::range(0, 100);
        BloomFilter sketch = build(values);
        const BloomFilter copy = sketch;
        sketch.merge(sketch);
        context.expect(sketch == copy, "self-merge is idempotent");
}

void test_merge_rejects_different_sizes(sketch_test::Context& context)
{
        const std::size_t sizes[][2] = { { 64, 128 }, { 100, 120 } };
        for (const auto& size : sizes) {
                const std::size_t a = size[0];
                const std::size_t b = size[1];
                context.expect_throws<std::invalid_argument>(
                    [a, b] {
                            BloomFilter left(a);
                            BloomFilter right(b);
                            left.merge(right);
                    },
                    "merging sketches of different m throws");
        }
}

// A non-word-aligned bitmap exercises the final partial word during the OR.
void test_merge_word_boundary(sketch_test::Context& context)
{
        const std::vector<std::uint64_t> left = sketch_test::range(0, 50);
        const std::vector<std::uint64_t> right = sketch_test::range(50, 100);
        const std::vector<std::uint64_t> all = sketch_test::range(0, 100);

        BloomFilter merged = build(left, 100);
        merged.merge(build(right, 100));
        context.expect(merged == build(all, 100),
            "merge equals the union at m = 100");
}

void test_variance_matches_theory(sketch_test::Context& context)
{
        constexpr std::size_t m = 8192;
        constexpr std::size_t n = 1000;
        constexpr std::size_t trials = 400;
        const std::vector<std::uint64_t> values = sketch_test::range(0, n);

        const sketch_test::MonteCarlo samples
            = sketch_test::monte_carlo<BloomFilter>(values, m, trials);
        if (!context.expect(samples.saturated == 0, "no saturation in the variance test"))
                return;

        const sketch_test::Summary summary = sketch_test::summarize(samples.estimates);
        const double load = 3.0 * static_cast<double>(n) / static_cast<double>(m);
        const double theory = static_cast<double>(m) / 9.0
            * (std::exp(load) - load - 1.0);
        const double observed = summary.sd * summary.sd;

        context.note("variance=" + std::to_string(observed)
            + " theory=" + std::to_string(theory));
        context.expect(std::abs(observed - theory) <= 0.35 * theory,
            "sample variance is within 35% of theory");
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

// One empty bit must give a finite estimate; setting the last bit saturates.
void test_saturation_boundaries(sketch_test::Context& context)
{
        const std::size_t sizes[] = { 64, 100 };
        for (std::size_t m : sizes) {
                BloomFilter sketch(m);

                // h = bit has h2 = 0, so each update sets exactly bit `bit`.
                for (std::uint64_t bit = 0; bit + 1 < m; ++bit)
                        sketch.update(bit);

                const double expected
                    = -(static_cast<double>(m) / 3.0)
                    * std::log(1.0 / static_cast<double>(m));
                const std::optional<double> estimate = sketch.estimate();
                context.expect(estimate.has_value() && std::abs(*estimate - expected) < 1e-9,
                    "one empty bit gives the exact finite estimate");

                sketch.update(static_cast<std::uint64_t>(m - 1));
                context.expect(!sketch.estimate().has_value(),
                    "zero empty bits gives nullopt");
        }
}

// h2 == 0 collapses the three double-hashing indices to one. One element must set one
// bit, not three.
void test_double_hashing_degenerate(sketch_test::Context& context)
{
        BloomFilter sketch(64);
        sketch.update(7); // h1 = 7, h2 = 0 => all three indices are 7.

        const double expected = -(64.0 / 3.0) * std::log((64.0 - 1.0) / 64.0);
        const std::optional<double> estimate = sketch.estimate();
        context.expect(estimate.has_value() && std::abs(*estimate - expected) < 1e-12,
            "h2 = 0 collapses the three indices to one bit");
}

// Compare the packed bitmap against a naive reference at a non-word-aligned size.
void test_matches_reference_bitmap(sketch_test::Context& context)
{
        constexpr std::size_t m = 100;
        const std::vector<std::uint64_t> values = sketch_test::range(0, 50);

        BloomFilter sketch(m);
        std::vector<bool> reference(m, false);
        for (std::uint64_t value : values) {
                const std::uint64_t hashed = hash64(value, kSeed);
                const std::uint32_t h1 = static_cast<std::uint32_t>(hashed);
                const std::uint32_t h2 = static_cast<std::uint32_t>(hashed >> 32);
                for (std::size_t i = 0; i < 3; ++i) {
                        const std::size_t index = (h1 + i * h2) % m;
                        reference[index] = true;
                }
                sketch.update(hashed);
        }

        std::size_t empty = 0;
        for (bool bit : reference) {
                if (!bit)
                        ++empty;
        }
        const double expected = -(static_cast<double>(m) / 3.0)
            * std::log(static_cast<double>(empty) / static_cast<double>(m));
        const std::optional<double> estimate = sketch.estimate();
        context.expect(estimate.has_value() && std::abs(*estimate - expected) < 1e-9,
            "estimate matches a reference bitmap at m = 100");
}

void test_zero_memory_throws(sketch_test::Context& context)
{
        context.expect_throws<std::invalid_argument>(
            [] { (void)BloomFilter(0); }, "m = 0 throws");
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
        context.expect(BloomFilter(100).memory_bits() == 100,
            "memory_bits is m for a non-word-aligned bitmap");
}

} // namespace

int main(int argc, char** argv)
{
        std::vector<sketch_test::TestGroup> groups;

        groups.emplace_back("Bloom Filter")
            .add("formula", test_formula)
            .add("bias", test_bias)
            .add("small streams", test_exact_on_small_streams)
            .add("different sketches differ", test_different_sketches_are_not_equal)
            .add("repeats", test_repeats_do_not_change_state)
            .add("reordering", test_reordering_does_not_change_state)
            .add("saturation boundaries", test_saturation_boundaries)
            .add("degenerate double hashing", test_double_hashing_degenerate)
            .add("reference bitmap", test_matches_reference_bitmap)
            .add("merge equals union", test_merge_equals_union)
            .add("overlapping merge", test_overlapping_merge)
            .add("self merge", test_merge_self)
            .add("merge rejects different sizes", test_merge_rejects_different_sizes)
            .add("merge word boundary", test_merge_word_boundary)
            .add("variance", test_variance_matches_theory)
            .add("zero memory throws", test_zero_memory_throws)
            .add("full bitmap", test_full_bitmap_has_no_estimate)
            .add("memory bits", test_memory_bits);

        return sketch_test::run_all(
            argc, argv, std::span<const sketch_test::TestGroup>(groups));
}
