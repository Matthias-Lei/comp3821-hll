#include "linear_counting.hpp"
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

LinearCounting build(std::span<const std::uint64_t> values, std::size_t m = kBits)
{
        return sketch_test::build_sketch<LinearCounting>(values, m, kSeed);
}

// With bits 0..n-1 set, the estimate must equal -m ln((m - n) / m) exactly.
void test_formula(sketch_test::Context& context)
{
        LinearCounting sketch(kBits);
        for (std::uint64_t h = 0; h < 100; ++h)
                sketch.update(h);

        const double expected = -8192.0 * std::log((8192.0 - 100.0) / 8192.0);
        const std::optional<double> estimate = sketch.estimate();
        context.expect(estimate.has_value() && std::abs(*estimate - expected) < 1e-9,
            "estimate matches -m ln(V)");
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

// The tests below rely on ==, so it must also tell different sketches apart.
void test_different_sketches_are_not_equal(sketch_test::Context& context)
{
        const std::vector<std::uint64_t> one { 1 };
        const std::vector<std::uint64_t> two { 2 };

        context.expect(!(build(one) == build(two)), "different bitmaps compare unequal");
        context.expect(!(LinearCounting(64) == LinearCounting(128)),
            "different m compare unequal");
        context.expect(!(LinearCounting(100) == LinearCounting(120)),
            "same word count, different m");
}

void test_repeats_do_not_change_state(sketch_test::Context& context)
{
        const std::vector<std::uint64_t> first { 1, 2, 3, 4, 5 };
        const std::vector<std::uint64_t> repeated { 1, 1, 2, 3, 3, 3, 4, 5, 5 };
        context.expect(build(first) == build(repeated), "repeats leave the bitmap unchanged");
}

void test_reordering_does_not_change_state(sketch_test::Context& context)
{
        std::vector<std::uint64_t> values = sketch_test::range(0, 1000);
        const LinearCounting forward = build(values);
        std::reverse(values.begin(), values.end());
        context.expect(forward == build(values), "reordering leaves the bitmap unchanged");
}

void test_merge_equals_union(sketch_test::Context& context)
{
        const std::vector<std::uint64_t> left = sketch_test::range(0, 500);
        const std::vector<std::uint64_t> right = sketch_test::range(500, 1000);
        const std::vector<std::uint64_t> all = sketch_test::range(0, 1000);

        LinearCounting merged = build(left);
        const LinearCounting right_sketch = build(right);
        const LinearCounting right_copy = right_sketch;
        merged.merge(right_sketch);

        const LinearCounting direct = build(all);
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

        LinearCounting merged = build(left);
        merged.merge(build(right));
        context.expect(merged == build(all), "overlapping merge equals the union sketch");
}

void test_merge_self(sketch_test::Context& context)
{
        const std::vector<std::uint64_t> values = sketch_test::range(0, 100);
        LinearCounting sketch = build(values);
        const LinearCounting copy = sketch;
        sketch.merge(sketch);
        context.expect(sketch == copy, "self-merge is idempotent");
}

void test_merge_rejects_different_sizes(sketch_test::Context& context)
{
        // 100 and 120 both fit in two words, so the check must compare m itself.
        const std::size_t sizes[][2] = { { 64, 128 }, { 100, 120 } };
        for (const auto& size : sizes) {
                const std::size_t a = size[0];
                const std::size_t b = size[1];
                context.expect_throws<std::invalid_argument>(
                    [a, b] {
                            LinearCounting left(a);
                            left.merge(LinearCounting(b));
                    },
                    "merging sketches of different m throws");
        }
}

// m = 100 leaves 28 unused bits in the last word, which must not count as empty.
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
        context.expect(LinearCounting(kBits).memory_bits() == kBits, "memory_bits is m");
}

void test_zero_memory_throws(sketch_test::Context& context)
{
        context.expect_throws<std::invalid_argument>(
            [] { (void)LinearCounting(0); }, "m = 0 throws");
}

// Exactly one empty bit must still give a finite estimate; one more bit saturates it.
void test_one_empty_bit(sketch_test::Context& context)
{
        const std::size_t sizes[] = { 64, 100 };
        for (std::size_t m : sizes) {
                LinearCounting sketch(m);
                for (std::uint64_t bit = 0; bit + 1 < m; ++bit)
                        sketch.update(bit);

                const double expected
                    = -static_cast<double>(m) * std::log(1.0 / static_cast<double>(m));
                const std::optional<double> estimate = sketch.estimate();
                context.expect(estimate.has_value() && std::abs(*estimate - expected) < 1e-9,
                    "one empty bit gives the exact finite estimate");

                sketch.update(static_cast<std::uint64_t>(m - 1));
                context.expect(!sketch.estimate().has_value(),
                    "zero empty bits gives nullopt");
        }
}

// The LC estimator's variance is m(e^t - t - 1); at t = 1 that is m(e - 2). The
// tolerance is deliberately loose; this is a sanity check, not a precision test.
void test_variance_matches_theory(sketch_test::Context& context)
{
        constexpr std::size_t m = 4096;
        constexpr std::size_t trials = 400;
        const std::vector<std::uint64_t> values = sketch_test::range(0, m);

        const sketch_test::MonteCarlo samples
            = sketch_test::monte_carlo<LinearCounting>(values, m, trials);
        if (!context.expect(samples.saturated == 0, "no saturation at t = 1"))
                return;

        const sketch_test::Summary summary = sketch_test::summarize(samples.estimates);
        const double theory = static_cast<double>(m) * (std::exp(1.0) - 2.0);
        const double observed = summary.sd * summary.sd;

        context.note("variance=" + std::to_string(observed)
            + " theory=" + std::to_string(theory));
        context.expect(std::abs(observed - theory) <= 0.35 * theory,
            "sample variance is within 35% of theory");
}

} // namespace

int main(int argc, char** argv)
{
        std::vector<sketch_test::TestGroup> groups;

        groups.emplace_back("Linear Counting")
            .add("formula", test_formula)
            .add("small streams", test_exact_on_small_streams)
            .add("different sketches differ", test_different_sketches_are_not_equal)
            .add("repeats", test_repeats_do_not_change_state)
            .add("reordering", test_reordering_does_not_change_state)
            .add("merge equals union", test_merge_equals_union)
            .add("overlapping merge", test_overlapping_merge)
            .add("self merge", test_merge_self)
            .add("merge rejects different sizes", test_merge_rejects_different_sizes)
            .add("full bitmap", test_full_bitmap_has_no_estimate)
            .add("memory bits", test_memory_bits)
            .add("zero memory throws", test_zero_memory_throws)
            .add("one empty bit", test_one_empty_bit)
            .add("variance", test_variance_matches_theory);

        return sketch_test::run_all(
            argc, argv, std::span<const sketch_test::TestGroup>(groups));
}
