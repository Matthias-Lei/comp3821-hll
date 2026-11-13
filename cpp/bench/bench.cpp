#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "bloom_filter.hpp"
#include "cebs_stream.hpp"
#include "dummy_sketch.hpp"
#include "hash.hpp"
#include "hyperloglog.hpp"
#include "kmv.hpp"
#include "linear_counting.hpp"

static std::string g_stream_path;

// Memory budgets from the proposal (1, 4, 8 and 16 KiB) and the hash seeds of the 64 trials.
static const std::vector<int64_t> kBudgets = { 8192, 32768, 65536, 131072 };
static const std::vector<int64_t> kHashSeeds = benchmark::CreateDenseRange(1, 64, 1);
// Splits a stream into two parts for the merge benchmarks. It is not one of the trial seeds.
static constexpr uint64_t kSplitSeed = 0;

// Each process benchmarks one stream, so its true cardinality is counted once and reused.
static uint64_t get_true_n(const std::uint64_t* data, std::size_t n)
{
        static std::optional<uint64_t> cached;
        if (!cached) {
                std::unordered_set<uint64_t> distinct(data, data + n);
                cached = distinct.size();
        }
        return *cached;
}

static void record_accuracy(benchmark::State& state, std::size_t memory_bits, uint64_t seed,
    std::size_t bits_used, uint64_t true_n, std::optional<double> est)
{
        state.counters["true_n"] = benchmark::Counter(static_cast<double>(true_n));
        state.counters["bits"] = benchmark::Counter(static_cast<double>(memory_bits));
        state.counters["bits_used"] = benchmark::Counter(static_cast<double>(bits_used));
        state.counters["seed"] = benchmark::Counter(static_cast<double>(seed));

        // A sketch with no estimate (e.g. a full bitmap) reports saturated = 1 and no error.
        state.counters["saturated"] = benchmark::Counter(est ? 0.0 : 1.0);
        if (!est)
                return;

        const double rel_err
            = std::abs(*est - static_cast<double>(true_n)) / static_cast<double>(true_n);
        state.counters["estimate"] = benchmark::Counter(*est);
        state.counters["rel_err"] = benchmark::Counter(rel_err);
}

// Times building the sketch over the whole stream with the run's hash seed (the second
// argument), then records the estimate. make(memory_bits) returns a new, empty sketch.
template <class MakeSketch>
static void bench_sketch(benchmark::State& state, MakeSketch make)
{
        const std::size_t memory_bits = static_cast<std::size_t>(state.range(0));
        const uint64_t seed = static_cast<uint64_t>(state.range(1));

        CebsStream stream(g_stream_path);
        const std::uint64_t* data = stream.data();
        const std::size_t n = stream.size();

        const uint64_t true_n = get_true_n(data, n);
        volatile std::uint64_t sink = 0;
        for (std::size_t i = 0; i < n; i += 4096 / sizeof(std::uint64_t))
                sink ^= data[i];
        (void)sink;

        std::optional<double> est;
        std::size_t bits_used = 0;
        for (auto _ : state) {
                auto sketch = make(memory_bits);
                for (std::size_t i = 0; i < n; ++i)
                        sketch.update(hash64(data[i], seed));
                est = sketch.estimate();
                bits_used = sketch.memory_bits();
                benchmark::DoNotOptimize(est);
        }

        record_accuracy(state, memory_bits, seed, bits_used, true_n, est);
}

// The stream split by value into two parts with no value in common, and each part's number of
// distinct values. Each process benchmarks one stream, so the split is made once and reused.
struct StreamSplit {
        std::vector<uint64_t> left;
        std::vector<uint64_t> right;
        uint64_t left_n = 0;
        uint64_t right_n = 0;
};

static const StreamSplit& get_split(const std::uint64_t* data, std::size_t n)
{
        static std::optional<StreamSplit> cached;
        if (!cached) {
                StreamSplit split;
                for (std::size_t i = 0; i < n; ++i)
                        (hash64(data[i], kSplitSeed) & 1 ? split.right : split.left)
                            .push_back(data[i]);
                split.left_n
                    = std::unordered_set<uint64_t>(split.left.begin(), split.left.end()).size();
                split.right_n
                    = std::unordered_set<uint64_t>(split.right.begin(), split.right.end()).size();
                cached = std::move(split);
        }
        return *cached;
}

// Mergeability (proposal): sketches the two disjoint parts of the stream, with d1 and d2 distinct
// values, using the run's hash seed and budget, merges them, and times all of that. Records the
// merged estimate of the d1 + d2 values, the part sizes, and whether the merged estimate equals
// that of a sketch built directly from the whole stream.
template <class MakeSketch>
static void bench_merge(benchmark::State& state, MakeSketch make)
{
        const std::size_t memory_bits = static_cast<std::size_t>(state.range(0));
        const uint64_t seed = static_cast<uint64_t>(state.range(1));

        CebsStream stream(g_stream_path);
        const std::uint64_t* data = stream.data();
        const std::size_t n = stream.size();
        const uint64_t true_n = get_true_n(data, n);
        const StreamSplit& split = get_split(data, n);

        const auto build = [&](const std::uint64_t* values, std::size_t count) {
                auto sketch = make(memory_bits);
                for (std::size_t i = 0; i < count; ++i)
                        sketch.update(hash64(values[i], seed));
                return sketch;
        };

        std::optional<double> est;
        std::size_t bits_used = 0;
        for (auto _ : state) {
                auto merged = build(split.left.data(), split.left.size());
                merged.merge(build(split.right.data(), split.right.size()));
                est = merged.estimate();
                bits_used = merged.memory_bits();
                benchmark::DoNotOptimize(est);
        }

        const std::optional<double> direct = build(data, n).estimate();
        record_accuracy(state, memory_bits, seed, bits_used, true_n, est);
        state.counters["merge_left_n"] = benchmark::Counter(static_cast<double>(split.left_n));
        state.counters["merge_right_n"] = benchmark::Counter(static_cast<double>(split.right_n));
        state.counters["merged_equals_direct"] = benchmark::Counter(est == direct ? 1.0 : 0.0);
}

// Registers a sketch benchmark for every memory budget and hash seed, one pass per run, so each
// run's time is a single cold pass. JSON names look like
// BM_LinearCounting/bits:8192/seed:7/iterations:1. bench_sketch reads both arguments, so every
// sketch must be registered with this macro rather than ->Arg(budget).
#define SKETCH_BENCHMARK(function)                                                                 \
        BENCHMARK(function)                                                                        \
            ->ArgsProduct({ kBudgets, kHashSeeds })                                                \
            ->ArgNames({ "bits", "seed" })                                                         \
            ->Iterations(1)                                                                        \
            ->Unit(benchmark::kMillisecond)

static void BM_DummySketch(benchmark::State& state)
{
        bench_sketch(state, [](std::size_t) { return DummySketch(); });
}
// The exact counter gives the same answer for every hash seed, so it runs with seed 1 only.
BENCHMARK(BM_DummySketch)
    ->ArgsProduct({ kBudgets, { 1 } })
    ->ArgNames({ "bits", "seed" })
    ->Unit(benchmark::kMillisecond);

static void BM_LinearCounting(benchmark::State& state)
{
        bench_sketch(state, [](std::size_t memory_bits) { return LinearCounting(memory_bits); });
}
SKETCH_BENCHMARK(BM_LinearCounting);

static void BM_BloomFilter(benchmark::State& state)
{
        bench_sketch(state, [](std::size_t memory_bits) { return BloomFilter(memory_bits); });
}
SKETCH_BENCHMARK(BM_BloomFilter);

static void BM_HyperLogLog(benchmark::State& state)
{
        bench_sketch(state, [](std::size_t memory_bits) { return HyperLogLog(memory_bits); });
}
SKETCH_BENCHMARK(BM_HyperLogLog);

// Merge benchmarks, for every sketch with a merge(). Their names end in "Merge", which the
// results analyser uses to pair them with the sketch's own theory.
static void BM_LinearCountingMerge(benchmark::State& state)
{
        bench_merge(state, [](std::size_t memory_bits) { return LinearCounting(memory_bits); });
}
SKETCH_BENCHMARK(BM_LinearCountingMerge);

static void BM_HyperLogLogMerge(benchmark::State& state)
{
        bench_merge(state, [](std::size_t memory_bits) { return HyperLogLog(memory_bits); });
}
SKETCH_BENCHMARK(BM_HyperLogLogMerge);

static void BM_KMinVals(benchmark::State& state)
{
        bench_sketch(state, [](std::size_t memory_bits) { return KMinValues(memory_bits); });
}
BENCHMARK(BM_KMinVals)
    ->Arg(8192)
    ->Arg(32768)
    ->Arg(65536)
    ->Arg(131072)
    ->Unit(benchmark::kMillisecond);

int main(int argc, char** argv)
{
        if (argc < 2) {
                std::cerr << "usage: " << argv[0] << " <stream.bin> [benchmark flags]\n";
                return 1;
        }

        g_stream_path = argv[1];

        for (int i = 1; i < argc - 1; ++i)
                argv[i] = argv[i + 1];
        --argc;

        benchmark::Initialize(&argc, argv);
        benchmark::RunSpecifiedBenchmarks();
        benchmark::Shutdown();
        return 0;
}
