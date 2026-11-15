"""Run the C++ benchmark on every stream and save one JSON file per stream.

Run from the repository root (in WSL), after building with CMake and creating streams:

    python3 python/run_benchmarks.py
    python3 python/run_benchmarks.py --filter BM_LinearCounting
"""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BINARY = BASE_DIR / "build" / "benchmarks"
DEFAULT_STREAMS_DIR = BASE_DIR / "data" / "streams"
DEFAULT_RESULTS_DIR = BASE_DIR / "data" / "results"

def find_streams(streams_dir: Path) -> list[Path]:
    return sorted(path for path in streams_dir.glob("*.cebs") if path.is_file())

def display_path(path: Path) -> str:
    """The path as typed from the repository root: ./build/benchmarks inside it, absolute outside."""
    try:
        return "./" + path.resolve().relative_to(BASE_DIR).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def benchmark_command(binary: Path, stream: Path, filter_: str | None) -> list[str]:
    command = [str(binary), str(stream), "--benchmark_format=json"]
    if filter_:
        command.append(f"--benchmark_filter={filter_}")
    return command

def run_benchmark(binary: Path, stream: Path, filter_: str | None) -> dict | None:
    """Runs the benchmark on one stream and returns its JSON output, or None if it failed."""
    result = subprocess.run(benchmark_command(binary, stream, filter_), capture_output=True, text=True)
    if result.returncode != 0:
        last_line = (result.stderr.strip().splitlines() or ["no error output"])[-1]
        print(f"skipped {stream.name}: {last_line}", file=sys.stderr)
        return None

    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        print(f"skipped {stream.name}: output is not JSON ({error})", file=sys.stderr)
        return None

    output["command"] = shlex.join(benchmark_command(Path(display_path(binary)), Path(display_path(stream)), filter_))
    return output

def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        json.dump(data, file, indent=2)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY, help="benchmark executable")
    parser.add_argument("--streams", type=Path, default=DEFAULT_STREAMS_DIR, help="directory of .cebs streams")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_DIR, help="directory for the JSON files")
    parser.add_argument("--filter", help="Google Benchmark --benchmark_filter regex, e.g. BM_LinearCounting")
    args = parser.parse_args()

    if not args.binary.is_file():
        print(f"benchmark executable not found: {args.binary} (build with CMake first)", file=sys.stderr)
        return 1
    streams = find_streams(args.streams)
    if not streams:
        print(f"no streams in {args.streams} (run python/create_test_streams.py first)", file=sys.stderr)
        return 1

    saved = 0
    for stream in streams:
        print(f"running {stream.name}", file=sys.stderr)
        output = run_benchmark(args.binary, stream, args.filter)
        if output is not None:
            save_json(output, args.results / f"{stream.stem}.json")
            saved += 1

    print(f"saved {saved} of {len(streams)} streams to {args.results}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())