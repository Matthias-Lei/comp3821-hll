#pragma once

// Shared harness for sketch tests.
//
// A test file registers named tests with a TestGroup and returns run_all() from main:
//
//     int main(int argc, char** argv)
//     {
//         std::vector<sketch_test::TestGroup> groups;
//         groups.emplace_back("MySketch")
//             .add("empty is exact", [](sketch_test::Context& t) {
//                 MySketch sketch(8192);
//                 t.expect(sketch.estimate() == 0.0, "empty sketch estimates 0");
//             });
//         return sketch_test::run_all(argc, argv, groups);
//     }
//
// Statistical tests use summarize()/t_statistic()/two_sided_p_value(). The p-value uses
// the normal approximation to Student's t: with n independent trials, df = n - 1. Tests
// that use it must use at least 200 trials at alpha = 1e-3 (the t and normal critical
// values differ by about 1.5% at df = 199 and 0.3% at df = 999). If more statistical
// tests are added, use a Bonferroni-corrected alpha.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "hash.hpp"

namespace sketch_test {

inline constexpr std::size_t kDefaultMemoryBits = 8192;
inline constexpr std::uint64_t kDefaultSeed = 0;

// ============================================================================
// test runner
// ============================================================================

class Context {
public:
        explicit Context(bool verbose)
                : verbose_(verbose)
        {
        }

        void begin(std::string_view test_name)
        {
                test_name_ = test_name;
        }

        bool expect(bool ok, std::string_view what)
        {
                if (!ok) {
                        ++failures_;
                        std::cerr << "FAIL [" << test_name_ << "] " << what << '\n';
                }
                return ok;
        }

        bool expect_close(
            double actual, double expected, double tolerance, std::string_view what)
        {
                const bool ok = std::abs(actual - expected) <= tolerance;
                if (!ok) {
                        std::cerr << "  actual=" << actual << " expected=" << expected
                                  << " tolerance=" << tolerance << '\n';
                }
                return expect(ok, what);
        }

        template <class Exception, class F>
        bool expect_throws(F&& f, std::string_view what)
        {
                try {
                        f();
                } catch (const Exception&) {
                        return true;
                } catch (const std::exception& e) {
                        std::cerr << "  unexpected exception: " << e.what() << '\n';
                        return expect(false, what);
                } catch (...) {
                        std::cerr << "  unexpected non-standard exception\n";
                        return expect(false, what);
                }
                return expect(false, what);
        }

        void note(std::string_view message) const
        {
                if (verbose_)
                        std::cout << "  " << message << '\n';
        }

        int failures() const
        {
                return failures_;
        }

        bool verbose() const
        {
                return verbose_;
        }

private:
        bool verbose_;
        int failures_ = 0;
        std::string_view test_name_;
};

struct TestCase {
        std::string name;
        std::function<void(Context&)> body;
};

class TestGroup {
public:
        explicit TestGroup(std::string name)
                : name_(std::move(name))
        {
        }

        // Adds a test and returns *this for chaining.
        TestGroup& add(std::string name, std::function<void(Context&)> body)
        {
                tests_.push_back(TestCase { std::move(name), std::move(body) });
                return *this;
        }

        // Returns the number of tests that had at least one failed check.
        int run(bool verbose = false) const
        {
                int failed = 0;
                for (const TestCase& test : tests_) {
                        Context context(verbose);
                        context.begin(test.name);
                        try {
                                test.body(context);
                        } catch (const std::exception& e) {
                                context.expect(false, std::string("threw: ") + e.what());
                        } catch (...) {
                                context.expect(false, "threw a non-standard exception");
                        }

                        if (context.failures() != 0) {
                                ++failed;
                                std::cout << "FAILED | " << name_ << " / " << test.name << " | "
                                          << context.failures() << " failed check(s)\n";
                        }
                }

                const std::size_t passed
                    = tests_.size() - static_cast<std::size_t>(failed);
                std::cout << "Test group | " << name_ << " | " << passed << " of "
                          << tests_.size() << " tests succeeded.\n";
                return failed;
        }

private:
        std::string name_;
        std::vector<TestCase> tests_;
};

inline int run_all(int argc, char** argv, std::span<const TestGroup> groups)
{
        bool verbose = false;
        for (int i = 1; i < argc; ++i) {
                const std::string_view arg(argv[i]);
                if (arg == "--verbose" || arg == "-v")
                        verbose = true;
        }

        int failed = 0;
        for (const TestGroup& group : groups)
                failed += group.run(verbose);

        if (failed == 0) {
                std::cout << "All tests passed.\n";
        } else {
                std::cout << failed << " test(s) failed.\n";
        }
        return failed == 0 ? 0 : 1;
}

// ============================================================================
// statistics
// ============================================================================

struct Summary {
        double mean;
        double sd;
        std::size_t n;
};

// Sample mean and sample standard deviation (n - 1). Throws for fewer than two
// values or a non-finite value, so a test can never silently compare a NaN.
inline Summary summarize(std::span<const double> values)
{
        if (values.size() < 2)
                throw std::invalid_argument("summarize: need at least two estimates");

        double sum = 0.0;
        for (double value : values) {
                if (!std::isfinite(value))
                        throw std::invalid_argument("summarize: estimate is not finite");
                sum += value;
        }

        const double mean = sum / static_cast<double>(values.size());
        double delta_squared = 0.0;
        for (double value : values) {
                const double delta = value - mean;
                delta_squared += delta * delta;
        }

        const double sd = std::sqrt(delta_squared / static_cast<double>(values.size() - 1));
        return Summary { mean, sd, values.size() };
}

// Normal-approximation t statistic. A zero standard deviation is handled explicitly:
// no evidence against a matching hypothesis (0), infinite evidence against a different
// one (signed infinity), never NaN.
inline double t_statistic(const Summary& summary, double hypothesised_mean)
{
        const double difference = summary.mean - hypothesised_mean;
        if (summary.sd == 0.0) {
                return difference == 0.0
                    ? 0.0
                    : std::copysign(std::numeric_limits<double>::infinity(), difference);
        }
        return difference * std::sqrt(static_cast<double>(summary.n)) / summary.sd;
}

// Two-sided normal p-value. erfc avoids the cancellation in 1 - Phi(|t|).
inline double two_sided_p_value(double t)
{
        return std::erfc(std::abs(t) / std::sqrt(2.0));
}

// Same zero-sd policy as t_statistic.
inline double cohens_d(const Summary& summary, double hypothesised_mean)
{
        const double difference = summary.mean - hypothesised_mean;
        if (summary.sd == 0.0) {
                return difference == 0.0
                    ? 0.0
                    : std::copysign(std::numeric_limits<double>::infinity(), difference);
        }
        return difference / summary.sd;
}

// ============================================================================
// streams and trial seeds
// ============================================================================

// splitmix64. Used to turn a trial index into an independent hash seed, so Monte Carlo
// tests do not rely on the properties of consecutive XXH64 seeds.
inline std::uint64_t seed_for_trial(std::uint64_t i)
{
        std::uint64_t z = i + 0x9e3779b97f4a7c15ULL;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
}

// Repository root, derived from this header's own path. CMake passes absolute source
// paths; the fallback covers a relative __FILE__ when the test runs from build/.
inline std::filesystem::path source_root()
{
        static const std::filesystem::path root = [] {
                const std::filesystem::path header(__FILE__);
                std::filesystem::path candidate
                    = header.parent_path().parent_path().parent_path();
                if (!std::filesystem::exists(candidate / "CMakeLists.txt")) {
                        const std::filesystem::path cwd = std::filesystem::current_path();
                        if (std::filesystem::exists(cwd.parent_path() / "CMakeLists.txt"))
                                candidate = cwd.parent_path();
                }
                return candidate;
        }();
        return root;
}

// Resolves a stream filename against CEBS_STREAM_DIR, or <repo>/data/streams by
// default. Throws with an actionable message if the file is missing.
inline std::filesystem::path stream_path(std::string_view filename)
{
        std::filesystem::path path;
        if (const char* dir = std::getenv("CEBS_STREAM_DIR"); dir != nullptr && *dir != '\0') {
                path = std::filesystem::path(dir)
                    / std::filesystem::path(std::string(filename));
        } else {
                path = source_root() / "data" / "streams"
                    / std::filesystem::path(std::string(filename));
        }

        if (!std::filesystem::exists(path)) {
                throw std::runtime_error("test stream not found: " + path.string()
                    + " (set CEBS_STREAM_DIR to the directory containing the .cebs files)");
        }
        return path;
}

// ============================================================================
// data and sketch helpers
// ============================================================================

inline std::vector<std::uint64_t> range(std::uint64_t start, std::uint64_t stop)
{
        std::vector<std::uint64_t> values;
        if (stop > start)
                values.reserve(static_cast<std::size_t>(stop - start));
        for (std::uint64_t value = start; value < stop; ++value)
                values.push_back(value);
        return values;
}

inline std::vector<std::uint64_t> to_vector(std::span<const std::uint64_t> values)
{
        return std::vector<std::uint64_t>(values.begin(), values.end());
}

inline std::vector<std::uint64_t> concatenate(
    std::span<const std::uint64_t> a, std::span<const std::uint64_t> b)
{
        std::vector<std::uint64_t> values;
        values.reserve(a.size() + b.size());
        values.insert(values.end(), a.begin(), a.end());
        values.insert(values.end(), b.begin(), b.end());
        return values;
}

inline std::uint64_t distinct_count(std::span<const std::uint64_t> values)
{
        const std::unordered_set<std::uint64_t> distinct(values.begin(), values.end());
        return static_cast<std::uint64_t>(distinct.size());
}

template <class SketchType>
SketchType build_sketch(std::span<const std::uint64_t> values, std::size_t memory_bits,
    std::uint64_t seed = kDefaultSeed)
{
        SketchType sketch(memory_bits);
        for (std::uint64_t value : values)
                sketch.update(hash64(value, seed));
        return sketch;
}

struct MonteCarlo {
        std::vector<double> estimates;
        std::size_t saturated = 0; // trials whose estimate() was nullopt
};

template <class SketchType>
MonteCarlo monte_carlo(std::span<const std::uint64_t> values, std::size_t memory_bits,
    std::size_t trials, std::uint64_t seed_base = 0)
{
        if (trials < 2)
                throw std::invalid_argument("monte_carlo: need at least two trials");

        MonteCarlo result;
        result.estimates.reserve(trials);
        for (std::size_t i = 0; i < trials; ++i) {
                SketchType sketch
                    = build_sketch<SketchType>(values, memory_bits, seed_for_trial(seed_base + i));
                const std::optional<double> estimate = sketch.estimate();
                if (!estimate)
                        ++result.saturated;
                else
                        result.estimates.push_back(*estimate);
        }
        return result;
}

// ============================================================================
// deterministic merge properties
// ============================================================================

// Requires SketchType::merge(const SketchType&).
template <class SketchType>
void expect_merge_equals_union(Context& context, std::span<const std::uint64_t> a,
    std::span<const std::uint64_t> b, std::size_t memory_bits, std::uint64_t seed = kDefaultSeed)
{
        SketchType merged = build_sketch<SketchType>(a, memory_bits, seed);
        merged.merge(build_sketch<SketchType>(b, memory_bits, seed));

        const std::vector<std::uint64_t> all = concatenate(a, b);
        const SketchType direct = build_sketch<SketchType>(all, memory_bits, seed);

        context.expect(merged.estimate() == direct.estimate(),
            "merged sketch estimates the same as the direct union");
}

template <class SketchType>
void expect_merge_idempotent(Context& context, std::span<const std::uint64_t> values,
    std::size_t memory_bits, std::uint64_t seed = kDefaultSeed)
{
        SketchType sketch = build_sketch<SketchType>(values, memory_bits, seed);
        const SketchType copy = sketch;
        sketch.merge(sketch);
        context.expect(sketch.estimate() == copy.estimate(), "self-merge is idempotent");
}

} // namespace sketch_test
