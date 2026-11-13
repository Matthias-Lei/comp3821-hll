#include "hyperloglog.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::size_t kDefaultBudgetBits = 8192;
constexpr std::uint64_t kHashSeed = 1;

HyperLogLog sketch_of(
    const std::vector<std::uint64_t>& keys, std::size_t budget_bits = kDefaultBudgetBits)
{
        return sketch_test::build_sketch<HyperLogLog>(keys, budget_bits, kHashSeed);
}

std::vector<std::uint64_t> distinct_keys(std::size_t count)
{
        std::vector<std::uint64_t> keys;
        keys.reserve(count);
        for (std::uint64_t i = 0; i < count; ++i)
                keys.push_back(i * 7919 + 13);
        return keys;
}

void test_register_counts(sketch_test::Context& context)
{
        const std::vector<std::uint64_t> empty;

        context.expect(sketch_of(empty, 8192).register_count() == 1024,
            "8192 bits gives 1024 registers");
        context.expect(sketch_of(empty, 32768).register_count() == 4096,
            "32768 bits gives 4096 registers");
        context.expect(sketch_of(empty, 65536).register_count() == 8192,
            "65536 bits gives 8192 registers");
        context.expect(sketch_of(empty, 131072).register_count() == 16384,
            "131072 bits gives 16384 registers");
        context.expect(sketch_of(empty, 8192).memory_bits() == 1024 * 6,
            "memory_bits is m * 6");
}

void test_small_counts(sketch_test::Context& context)
{
        for (std::size_t count : { 1, 2, 5, 10 }) {
                const std::optional<double> estimate
                    = sketch_of(distinct_keys(count)).estimate();
                context.expect(estimate.has_value()
                        && std::llround(*estimate) == static_cast<long long>(count),
                    "exact small count");
        }

        for (std::size_t count : { 50, 100 }) {
                const double register_count = 1024.0;
                const double load = static_cast<double>(count) / register_count;
                const double expected_sd
                    = std::sqrt(register_count * (std::exp(load) - load - 1.0));
                const std::optional<double> estimate
                    = sketch_of(distinct_keys(count)).estimate();
                context.expect(estimate.has_value()
                        && std::abs(*estimate - static_cast<double>(count))
                            <= 3 * expected_sd,
                    "small count within 3 sigma of linear counting");
        }
}

void test_repeats_and_order(sketch_test::Context& context)
{
        const std::vector<std::uint64_t> keys = distinct_keys(5000);
        std::vector<std::uint64_t> repeated;
        repeated.reserve(3 * keys.size());
        for (int round = 0; round < 3; ++round)
                repeated.insert(repeated.end(), keys.begin(), keys.end());

        context.expect(sketch_of(keys).estimate() == sketch_of(repeated).estimate(),
            "repeats are ignored");

        const std::vector<std::uint64_t> reversed(keys.rbegin(), keys.rend());
        context.expect(sketch_of(keys).estimate() == sketch_of(reversed).estimate(),
            "order is ignored");
}

void test_merge(sketch_test::Context& context)
{
        std::vector<std::uint64_t> stream;
        stream.reserve(200000);
        for (std::uint64_t i = 0; i < 200000; ++i)
                stream.push_back(i * 2654435761ULL);

        const std::size_t half = stream.size() / 2;
        const std::vector<std::uint64_t> first_half(stream.begin(), stream.begin() + half);
        const std::vector<std::uint64_t> second_half(stream.begin() + half, stream.end());

        HyperLogLog merged = sketch_of(first_half);
        merged.merge(sketch_of(second_half));
        const HyperLogLog whole = sketch_of(stream);

        context.expect(merged.estimate() == whole.estimate(),
            "merge equals whole-stream sketch");

        const double true_count = static_cast<double>(stream.size());
        const double relative_error = std::abs(*merged.estimate() - true_count) / true_count;
        context.expect(relative_error < 4 * 1.04 / std::sqrt(1024.0),
            "merged error within 4 sigma");

        const HyperLogLog mismatched(32768);
        context.expect_throws<std::invalid_argument>(
            [&merged, &mismatched] { merged.merge(mismatched); },
            "merge rejects mismatched register counts");
}

} // namespace

int main(int argc, char** argv)
{
        std::vector<sketch_test::TestGroup> groups;

        groups.emplace_back("HyperLogLog")
            .add("register counts", test_register_counts)
            .add("small counts", test_small_counts)
            .add("repeats and order", test_repeats_and_order)
            .add("merge", test_merge);

        return sketch_test::run_all(
            argc, argv, std::span<const sketch_test::TestGroup>(groups));
}
