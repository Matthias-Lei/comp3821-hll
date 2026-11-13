#include "bloom_filter.hpp"
#include "hash.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
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

static BloomFilter build(const std::vector<std::uint64_t>& values, std::size_t m = kBits)
{
        BloomFilter sketch(m);
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

// If we pass 0..99 to update(), the double hashing sets exactly 100 bits.
// The estimate must equal -(m/k) ln((m - 100) / m) exactly.
static void test_formula()
{
        BloomFilter sketch(kBits);
        for (std::uint64_t h = 0; h < 100; ++h)
                sketch.update(h);

        // m = 8192, k = 3, set_bits = 100
        const double expected = -(8192.0 / 3.0) * std::log((8192.0 - 100.0) / 8192.0);
        const auto est = sketch.estimate();
        check(est.has_value() && std::abs(*est - expected) < 1e-9, "estimate matches -m/k ln(V)");
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

static void test_repeats_do_not_change_state()
{
        check(
            build({ 1, 2, 3, 4, 5 }).estimate() == build({ 1, 1, 2, 3, 3, 3, 4, 5, 5 }).estimate(),
            "repeats leave the estimate unchanged");
}

static void test_reordering_does_not_change_state()
{
        auto values = range(0, 1000);
        const auto forward = build(values).estimate();
        std::reverse(values.begin(), values.end());
        check(forward == build(values).estimate(), "reordering leaves the estimate unchanged");
}

// m = 100 and m = 64 will saturate very quickly with k=3 hashes per element.
static void test_full_bitmap_has_no_estimate()
{
        const std::size_t sizes[] = { 64, 100 };
        for (std::size_t m : sizes)
                check(
                    !build(range(0, 10000), m).estimate().has_value(), "full bitmap gives nullopt");
}

static void test_memory_bits()
{
        check(BloomFilter(kBits).memory_bits() == kBits, "memory_bits is m");
}

int main()
{
        test_formula();
        test_exact_on_small_streams();
        test_repeats_do_not_change_state();
        test_reordering_does_not_change_state();
        test_full_bitmap_has_no_estimate();
        test_memory_bits();

        if (failures != 0)
                return 1;
        std::cout << "test_bloom_filter passed\n";
        return 0;
}
