#include "hash.hpp"
#include "linear_counting.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

static constexpr std::size_t kBits = 8192;
static constexpr std::uint64_t kSeed = 1;

static int failures = 0;

static void check(bool ok, const char* what)
{
        if (!ok) {
                std::cerr << "FAIL: " << what << '\n';
                ++failures;
        }
}

static LinearCounting build(const std::vector<std::uint64_t>& values, std::size_t m = kBits)
{
        LinearCounting sketch(m);
        for (std::uint64_t v : values)
                sketch.update(hash64(v, kSeed));
        return sketch;
}

static std::vector<std::uint64_t> range(std::uint64_t start, std::uint64_t stop)
{
        std::vector<std::uint64_t> values;
        for (std::uint64_t v = start; v < stop; ++v)
                values.push_back(v);
        return values;
}

// With bits 0..n-1 set, the estimate must equal -m ln((m - n) / m) exactly.
static void test_formula()
{
        LinearCounting sketch(kBits);
        for (std::uint64_t h = 0; h < 100; ++h)
                sketch.update(h);
        const double expected = -8192.0 * std::log((8192.0 - 100.0) / 8192.0);
        const auto est = sketch.estimate();
        check(est.has_value() && std::abs(*est - expected) < 1e-9, "estimate matches -m ln(V)");
}

static void test_exact_on_small_streams()
{
        check(build({ }).estimate() == 0.0, "empty stream estimates 0");

        const std::vector<std::vector<std::uint64_t>> streams = {
                { 10, 20, 10, 30, 20 },
                range(0, 20),
                { 7, 7, 7, 7 },
        };
        for (const auto& stream : streams) {
                const std::unordered_set<std::uint64_t> distinct(stream.begin(), stream.end());
                const auto est = build(stream).estimate();
                check(est.has_value() && std::llround(*est) == std::ssize(distinct),
                    "small stream rounds to the exact distinct count");
        }
}

// The tests below rely on ==, so it must also tell different sketches apart.
static void test_different_sketches_are_not_equal()
{
        check(!(build({ 1 }) == build({ 2 })), "different bitmaps compare unequal");
        check(!(LinearCounting(64) == LinearCounting(128)), "different m compare unequal");
        check(!(LinearCounting(100) == LinearCounting(120)), "same word count, different m");
}

static void test_repeats_do_not_change_state()
{
        check(build({ 1, 2, 3, 4, 5 }) == build({ 1, 1, 2, 3, 3, 3, 4, 5, 5 }),
            "repeats leave the bitmap unchanged");
}

static void test_reordering_does_not_change_state()
{
        auto values = range(0, 1000);
        const auto forward = build(values);
        std::reverse(values.begin(), values.end());
        check(forward == build(values), "reordering leaves the bitmap unchanged");
}

static void test_merge_equals_union()
{
        auto merged = build(range(0, 500));
        merged.merge(build(range(500, 1000)));
        const auto direct = build(range(0, 1000));
        check(merged == direct, "merge equals the sketch of the union");
        check(merged.estimate() == direct.estimate(), "merge estimates the same as the union");
}

static void test_merge_rejects_different_sizes()
{
        // 100 and 120 both fit in two words, so the check must compare m itself.
        const std::size_t sizes[][2] = { { 64, 128 }, { 100, 120 } };
        for (const auto& [a, b] : sizes) {
                LinearCounting left(a);
                const LinearCounting right(b);
                bool threw = false;
                try {
                        left.merge(right);
                } catch (const std::invalid_argument&) {
                        threw = true;
                }
                check(threw, "merging sketches of different m throws");
        }
}

// m = 100 leaves 28 unused bits in the last word, which must not count as empty.
static void test_full_bitmap_has_no_estimate()
{
        const std::size_t sizes[] = { 64, 100 };
        for (std::size_t m : sizes)
                check(!build(range(0, 10000), m).estimate().has_value(),
                    "full bitmap gives nullopt");
}

static void test_memory_bits()
{
        check(LinearCounting(kBits).memory_bits() == kBits, "memory_bits is m");
}

int main()
{
        test_formula();
        test_exact_on_small_streams();
        test_different_sketches_are_not_equal();
        test_repeats_do_not_change_state();
        test_reordering_does_not_change_state();
        test_merge_equals_union();
        test_merge_rejects_different_sizes();
        test_full_bitmap_has_no_estimate();
        test_memory_bits();

        if (failures != 0)
                return 1;
        std::cout << "test_linear_counting passed\n";
        return 0;
}
