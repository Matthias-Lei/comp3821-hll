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

Generate a small test stream (later Python side will do this but better):

```sh
mkdir -p data/streams
python -c "import numpy as np; \
  np.array([1,2,3,1,2,4,5,3,6], dtype=np.uint64).tofile('data/streams/seed0_n6.bin')"
```

Run the benchmark:

```sh
./build/benchmarks data/streams/seed0_n6.bin --benchmark_format=json
```

Streams are named `seed{N}_n{M}.bin`, where `M` is the true distinct count.
`data/` is gitignored.
