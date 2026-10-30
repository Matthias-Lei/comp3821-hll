#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_set>

#include "cebs_stream.hpp"
#include "dummy_sketch.hpp"
#include "hash.hpp"
#include "linear_counting.hpp"
#include "kmv.hpp"

static std::string g_stream_path;
static constexpr uint64_t kHashSeed = 1;

static uint64_t get_true_n(const std::uint64_t* data, std::size_t n)
{
        std::unordered_set<uint64_t> distinct(data, data + n);
        return distinct.size();
}

// make(memory_bits) returns a new, empty sketch for that budget.
template <class MakeSketch>
static void run_accuracy(benchmark::State& state, const uint64_t* data, std::size_t n,
    std::size_t memory_bits, std::size_t true_n, MakeSketch make)
{
        auto sketch = make(memory_bits);
        for (std::size_t i = 0; i < n; ++i)
                sketch.update(hash64(data[i], kHashSeed));

        state.counters["true_n"] = benchmark::Counter(static_cast<double>(true_n));
        state.counters["bits"] = benchmark::Counter(static_cast<double>(memory_bits));

        // A sketch with no estimate (e.g. a full bitmap) reports saturated = 1 and no error.
        const std::optional<double> est = sketch.estimate();
        state.counters["saturated"] = benchmark::Counter(est ? 0.0 : 1.0);
        if (!est)
                return;

        const double rel_err
            = std::abs(*est - static_cast<double>(true_n)) / static_cast<double>(true_n);
        state.counters["estimate"] = benchmark::Counter(*est);
        state.counters["rel_err"] = benchmark::Counter(rel_err);
}

// Times building the sketch over the whole stream, then records its accuracy.
template <class MakeSketch>
static void bench_sketch(benchmark::State& state, MakeSketch make)
{
        const std::size_t memory_bits = static_cast<std::size_t>(state.range(0));

        CebsStream stream(g_stream_path);
        const std::uint64_t* data = stream.data();
        const std::size_t n = stream.size();

        const uint64_t true_n = get_true_n(data, n);
        volatile std::uint64_t sink = 0;
        for (std::size_t i = 0; i < n; i += 4096 / sizeof(std::uint64_t))
                sink ^= data[i];
        (void)sink;

        for (auto _ : state) {
                auto sketch = make(memory_bits);
                for (std::size_t i = 0; i < n; ++i)
                        sketch.update(hash64(data[i], kHashSeed));
                std::optional<double> est = sketch.estimate();
                benchmark::DoNotOptimize(est);
        }

        run_accuracy(state, data, n, memory_bits, true_n, make);
}

static void BM_DummySketch(benchmark::State& state)
{
        bench_sketch(state, [](std::size_t) { return DummySketch(); });
}
BENCHMARK(BM_DummySketch)
    ->Arg(8192)
    ->Arg(32768)
    ->Arg(65536)
    ->Arg(131072)
    ->Unit(benchmark::kMillisecond);

static void BM_LinearCounting(benchmark::State& state)
{
        bench_sketch(state, [](std::size_t memory_bits) { return LinearCounting(memory_bits); });
}
BENCHMARK(BM_LinearCounting)
    ->Arg(8192)
    ->Arg(32768)
    ->Arg(65536)
    ->Arg(131072)
    ->Unit(benchmark::kMillisecond);

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
