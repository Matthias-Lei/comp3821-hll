#include "hash.hpp"
#include "hyperloglog.hpp"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::size_t kDefaultBudgetBits = 8192;
constexpr std::uint64_t kHashSeed = 1;

HyperLogLog sketch_of(
    const std::vector<std::uint64_t>& keys, std::size_t budget_bits = kDefaultBudgetBits)
{
        HyperLogLog sketch(budget_bits);
        for (std::uint64_t key : keys) {
                sketch.update(hash64(key, kHashSeed));
        }
        return sketch;
}

bool expect(bool condition, const char* description)
{
        if (!condition) {
                std::cerr << "FAILED: " << description << '\n';
        }
        return condition;
}

std::vector<std::uint64_t> distinct_keys(std::size_t count)
{
        std::vector<std::uint64_t> keys;
        for (std::uint64_t i = 0; i < count; ++i) {
                keys.push_back(i * 7919 + 13);
        }
        return keys;
}

}

int main()
{
        bool all_passed = true;

        all_passed &= expect(
            sketch_of({ }, 8192).register_count() == 1024, "8192 bits gives 1024 registers");
        all_passed &= expect(
            sketch_of({ }, 32768).register_count() == 4096, "32768 bits gives 4096 registers");
        all_passed &= expect(
            sketch_of({ }, 65536).register_count() == 8192, "65536 bits gives 8192 registers");
        all_passed &= expect(
            sketch_of({ }, 131072).register_count() == 16384, "131072 bits gives 16384 registers");
        all_passed
            &= expect(sketch_of({ }, 8192).memory_bits() == 1024 * 6, "memory_bits is m * 6");

        for (std::size_t count : { 1, 2, 5, 10 }) {
                const double estimate = *sketch_of(distinct_keys(count)).estimate();
                all_passed &= expect(
                    std::llround(estimate) == static_cast<long long>(count), "exact small count");
        }
        for (std::size_t count : { 50, 100 }) {
                const double register_count = 1024.0;
                const double load = static_cast<double>(count) / register_count;
                const double expected_sd
                    = std::sqrt(register_count * (std::exp(load) - load - 1.0));
                const double estimate = *sketch_of(distinct_keys(count)).estimate();
                all_passed
                    &= expect(std::abs(estimate - static_cast<double>(count)) <= 3 * expected_sd,
                        "small count within 3 sigma of linear counting");
        }

        std::vector<std::uint64_t> keys = distinct_keys(5000);
        std::vector<std::uint64_t> repeated;
        for (int round = 0; round < 3; ++round) {
                repeated.insert(repeated.end(), keys.begin(), keys.end());
        }
        all_passed &= expect(
            sketch_of(keys).estimate() == sketch_of(repeated).estimate(), "repeats are ignored");

        std::vector<std::uint64_t> reversed(keys.rbegin(), keys.rend());
        all_passed &= expect(
            sketch_of(keys).estimate() == sketch_of(reversed).estimate(), "order is ignored");

        std::vector<std::uint64_t> stream;
        for (std::uint64_t i = 0; i < 200000; ++i) {
                stream.push_back(i * 2654435761ULL);
        }
        const std::size_t half = stream.size() / 2;
        const std::vector<std::uint64_t> first_half(stream.begin(), stream.begin() + half);
        const std::vector<std::uint64_t> second_half(stream.begin() + half, stream.end());

        HyperLogLog merged = sketch_of(first_half);
        merged.merge(sketch_of(second_half));
        const HyperLogLog whole = sketch_of(stream);
        all_passed
            &= expect(merged.estimate() == whole.estimate(), "merge equals whole-stream sketch");

        const double true_count = static_cast<double>(stream.size());
        const double relative_error = std::abs(*merged.estimate() - true_count) / true_count;
        all_passed
            &= expect(relative_error < 4 * 1.04 / std::sqrt(1024.0), "merged error within 4 sigma");

        HyperLogLog mismatched(32768);
        bool merge_threw = false;
        try {
                merged.merge(mismatched);
        } catch (const std::invalid_argument&) {
                merge_threw = true;
        }
        all_passed &= expect(merge_threw, "merge rejects mismatched register counts");

        return all_passed ? 0 : 1;
}
