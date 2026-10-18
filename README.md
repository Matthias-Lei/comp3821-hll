# Cardinality Sketches

## Dependencies

- CMake 3.20+
- C++20
- GCC 14 (or Clang 14+)
- Google Benchmark 1.8+
- Python 3.10+
- Make (for building the C generator shared library)

The system Google Benchmark is preferred. If not found, CMake fetches
`v1.9.5` from GitHub via `FetchContent` automatically.

## Build & Setup

Run the following commands from the repository root.

### 1. Python Environment & Stream Generation

Choose either `pip` or `uv` to set up the environment.

**Using pip:**

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt
```
**Using uv:**
```
uv venv
source .venv/bin/activate
uv pip install -r python/requirements.txt
```

Once the environment is active, build the C generator and generate the test streams:

```
make -C python/
python3 python/create_test_streams.py --make-directories
```

### 2. C++ build

```
cmake -S . -B build
cmake --build build
```

### Run
Run the benchmark against one of the generated .cebs files using Google Benchmark's JSON output:

```
./build/benchmarks data/streams/name_seed{N}.cebs --benchmark_format=json
```
e.g.
```
./build/benchmarks data/streams/bernoulli_observations_seed7427369.cebs --benchmark_format=json
```

Streams are generated in data/streams/ and follow the naming convention name_seed{N}.cebs (or range_python.cebs). The data/ directory is gitignored.

## Shared hash

`cpp/include/hash.hpp` provides `hash64(value, seed)`: XXH64 of the value's 8 bytes, least
significant byte first. CMake fetches xxHash v0.8.4 (BSD 2-Clause) at configure time.

`test_hash` checks known outputs and a fixed-seed chi-square test (1,000,000 keys, 1024 buckets, p > 0.01).

