#include "cebs_stream.hpp"
#include "kmv.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kMemoryBits = 8192;
constexpr std::uint64_t kSeed = 1;
constexpr std::size_t kTrials = 1000;
constexpr double kAlpha = 0.05;

void expect_exact_estimate(
    sketch_test::Context& context, std::span<const std::uint64_t> values, std::size_t memory_bits)
{
        const std::uint64_t truth = sketch_test::distinct_count(values);
        const KMinValues sketch = sketch_test::build_sketch<KMinValues>(values, memory_bits, kSeed);
        const std::optional<double> estimate = sketch.estimate();

        context.expect(estimate.has_value() && *estimate == static_cast<double>(truth),
            "estimate equals the exact cardinality when n < k");
}

void expect_unbiased_on_stream(sketch_test::Context& context, std::string_view filename)
{
        const std::string path = sketch_test::stream_path(filename).string();
        const CebsStream stream(path);
        const std::span<const std::uint64_t> values(stream.data(), stream.size());
        const std::uint64_t truth = sketch_test::distinct_count(values);

        const sketch_test::MonteCarlo samples
            = sketch_test::monte_carlo<KMinValues>(values, kMemoryBits, kTrials);
        if (!context.expect(samples.saturated == 0, "KMV must never return nullopt"))
                return;

        const sketch_test::Summary summary = sketch_test::summarize(samples.estimates);
        const double p = sketch_test::two_sided_p_value(
            sketch_test::t_statistic(summary, static_cast<double>(truth)));

        context.note("mean=" + std::to_string(summary.mean) + " sd=" + std::to_string(summary.sd)
            + " p=" + std::to_string(p));
        context.expect(p > kAlpha, "mean estimate is consistent with the true cardinality");
}

void expect_merge_properties(sketch_test::Context& context)
{
        const std::vector<std::uint64_t> all = sketch_test::range(0, 1000);
        const std::vector<std::uint64_t> left = sketch_test::range(0, 500);
        const std::vector<std::uint64_t> right = sketch_test::range(500, 1000);
        const std::vector<std::uint64_t> overlap_left = sketch_test::range(0, 600);
        const std::vector<std::uint64_t> overlap_right = sketch_test::range(400, 1000);

        const KMinValues direct = sketch_test::build_sketch<KMinValues>(all, kMemoryBits, kSeed);

        KMinValues disjoint = sketch_test::build_sketch<KMinValues>(left, kMemoryBits, kSeed);
        const KMinValues right_sketch
            = sketch_test::build_sketch<KMinValues>(right, kMemoryBits, kSeed);
        const KMinValues right_copy = right_sketch;
        disjoint.merge(right_sketch);
        context.expect(disjoint == direct, "disjoint merge equals the union sketch");
        context.expect(right_sketch == right_copy, "merge leaves the source unchanged");

        KMinValues overlapping
            = sketch_test::build_sketch<KMinValues>(overlap_left, kMemoryBits, kSeed);
        overlapping.merge(sketch_test::build_sketch<KMinValues>(overlap_right, kMemoryBits, kSeed));
        context.expect(overlapping == direct, "overlapping merge equals the union sketch");

        KMinValues self = sketch_test::build_sketch<KMinValues>(all, kMemoryBits, kSeed);
        const KMinValues self_copy = self;
        self.merge(self);
        context.expect(self == self_copy, "self-merge is idempotent");

        const KMinValues larger
            = sketch_test::build_sketch<KMinValues>(all, 2 * kMemoryBits, kSeed);
        context.expect_throws<std::invalid_argument>(
            [&disjoint, &larger] {
                    KMinValues copy = disjoint;
                    copy.merge(larger);
            },
            "merge rejects a sketch with a different k");
}

void expect_boundaries(sketch_test::Context& context)
{
        context.expect_throws<std::invalid_argument>([] { (void)KMinValues(0); }, "m = 0 throws");
        context.expect_throws<std::invalid_argument>([] { (void)KMinValues(64); }, "k = 1 throws");
        context.expect_throws<std::invalid_argument>(
            [] { (void)KMinValues(127); }, "m < 128 throws");
        context.expect(KMinValues(128).memory_bits() == 128, "m = 128 gives k = 2");
        context.expect(
            KMinValues(kMemoryBits).memory_bits() == kMemoryBits, "memory_bits is k * 64");
        context.expect(KMinValues(128).estimate() == 0.0, "empty sketch estimates 0");

        const std::vector<std::uint64_t> values = sketch_test::range(0, 100);
        std::vector<std::uint64_t> repeated;
        repeated.reserve(2 * values.size());
        repeated.insert(repeated.end(), values.begin(), values.end());
        repeated.insert(repeated.end(), values.begin(), values.end());
        context.expect(sketch_test::build_sketch<KMinValues>(values, kMemoryBits, kSeed)
                == sketch_test::build_sketch<KMinValues>(repeated, kMemoryBits, kSeed),
            "repeats leave the sketch unchanged");

        KMinValues zero_hash(128); // k = 2
        zero_hash.update(0);
        zero_hash.update(1);
        const std::optional<double> estimate = zero_hash.estimate();
        context.expect(estimate.has_value() && std::isfinite(*estimate),
            "a zero hash does not make the estimate non-finite");
}

} // namespace

int main(int argc, char** argv)
{
        std::vector<sketch_test::TestGroup> groups;

        groups.emplace_back("Estimate is exact when n < k")
            .add("empty sketch",
                [](sketch_test::Context& context) {
                        const std::vector<std::uint64_t> empty;
                        expect_exact_estimate(context, empty, 128);
                })
            .add("n = 20 with k = 21",
                [](sketch_test::Context& context) {
                        const std::vector<std::uint64_t> values = sketch_test::range(0, 20);
                        expect_exact_estimate(context, values, 21 * 64);
                })
            .add("n = 11 with k = 20",
                [](sketch_test::Context& context) {
                        const std::vector<std::uint64_t> values = sketch_test::range(0, 11);
                        expect_exact_estimate(context, values, 20 * 64);
                })
            .add("n = 2 with k = 3", [](sketch_test::Context& context) {
                    const std::vector<std::uint64_t> values = sketch_test::range(0, 2);
                    expect_exact_estimate(context, values, 3 * 64);
            });

        groups.emplace_back("Estimate is unbiased for n > k")
            .add("normal stream",
                [](sketch_test::Context& context) {
                        expect_unbiased_on_stream(
                            context, "normal_observations_seed7427369_m1000.cebs");
                })
            .add("chi-squared stream",
                [](sketch_test::Context& context) {
                        expect_unbiased_on_stream(
                            context, "chi_squared_observations_seed7427369_m1000.cebs");
                })
            .add("beta stream", [](sketch_test::Context& context) {
                    expect_unbiased_on_stream(context, "beta_observations_seed7427369_m1000.cebs");
            });

        groups.emplace_back("Merge equals union")
            .add(
                "disjoint, overlap, self, source unchanged, mismatched k", expect_merge_properties);

        groups.emplace_back("Boundaries")
            .add("constructor and memory boundaries", expect_boundaries);

        return sketch_test::run_all(argc, argv, std::span<const sketch_test::TestGroup>(groups));
}
