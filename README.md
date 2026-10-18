# Cardinality Sketches

## Dependencies

- CMake 3.20+
- C++20
- GCC 14 (or Clang 14+)
- Google Benchmark 1.8+

The system Google Benchmark is preferred. If not found, CMake fetches
`v1.9.5` from GitHub via `FetchContent` automatically.

## Build

```sh
cmake -S . -B build
cmake --build build
```

## Run

Run the benchmark:

```sh
./build/benchmarks data/streams/seed{N}_n{M}.cebs --benchmark_format=json
```

Streams are named `seed{N}_n{M}.cebs`, where `M` is the true distinct count.
`data/` is gitignored.
